// SPDX-License-Identifier: MIT
/* Copyright 2015 Advanced Micro Devices, Inc. */
/* Copyright (c) 2025 Valve Corporation */

#include <linux/array_size.h>
#include <linux/bug.h>
#include <linux/debugfs.h>
#include <linux/err.h>
#include <linux/fs.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/rbtree.h>
#include <linux/seq_file.h>
#include <linux/string.h>

#include <drm/drm_print.h>
#include <drm/gpu_scheduler.h>

#include "sched_internal.h"

static __always_inline bool
drm_sched_entity_compare_before(struct rb_node *a, const struct rb_node *b)
{
	struct drm_sched_entity *ea =
		rb_entry((a), struct drm_sched_entity, rb_tree_node);
	struct drm_sched_entity *eb =
		rb_entry((b), struct drm_sched_entity, rb_tree_node);

	return ktime_before(ea->oldest_job_waiting, eb->oldest_job_waiting);
}

static void drm_sched_rq_update_prio(struct drm_sched_rq *rq)
{
	enum drm_sched_priority prio = DRM_SCHED_PRIORITY_INVALID;
	struct rb_node *rb;

	lockdep_assert_held(&rq->lock);

	rb = rb_first_cached(&rq->rb_tree_root);
	if (rb) {
		struct drm_sched_entity *entity =
			rb_entry(rb, typeof(*entity), rb_tree_node);

		/*
		 * The normal locking order is entity then run-queue so taking
		 * the entity lock here would be a locking inversion for the
		 * case when the current head of the run-queue is different from
		 * the one we already have locked. The unlocked read is fine
		 * though, because if the priority had just changed it is no big
		 * deal for our algorithm, but just a transient reachable only
		 * by drivers with userspace dynamic priority changes API. Equal
		 * in effect to the priority change becoming visible a few
		 * instructions later.
		 */
		prio = READ_ONCE(entity->priority);
	}

	rq->head_prio = prio;
}

static void drm_sched_rq_remove_fifo_locked(struct drm_sched_entity *entity,
					    struct drm_sched_rq *rq)
{
	lockdep_assert_held(&entity->lock);
	lockdep_assert_held(&rq->lock);

	if (!RB_EMPTY_NODE(&entity->rb_tree_node)) {
		rb_erase_cached(&entity->rb_tree_node, &rq->rb_tree_root);
		RB_CLEAR_NODE(&entity->rb_tree_node);
		drm_sched_rq_update_prio(rq);
	}
}

static void drm_sched_rq_update_fifo_locked(struct drm_sched_entity *entity,
					    struct drm_sched_rq *rq,
					    ktime_t ts)
{
	/*
	 * Both locks need to be grabbed, one to protect from entity->rq change
	 * for entity from within concurrent drm_sched_entity_select_rq and the
	 * other to update the rb tree structure.
	 */
	lockdep_assert_held(&entity->lock);
	lockdep_assert_held(&rq->lock);

	drm_sched_rq_remove_fifo_locked(entity, rq);

	entity->oldest_job_waiting = ts;

	rb_add_cached(&entity->rb_tree_node, &rq->rb_tree_root,
		      drm_sched_entity_compare_before);
	drm_sched_rq_update_prio(rq);
}

static void drm_sched_rq_remove_inf_locked(struct drm_sched_entity *entity,
					   struct drm_sched_rq *rq)
{
	lockdep_assert_held(&entity->lock);
	lockdep_assert_held(&rq->lock);

	if (!list_empty(&entity->inf_node))
		list_del_init(&entity->inf_node);
}

/*
 * Bounded-LIFO insert (KPP-style, O(1)): 7 inserts at the head (LIFO fast
 * lane for fresh entities) + 1 insert at the tail (forced progress for
 * older entities) per group of 8. Counter wrap forces the tail. The
 * rb-tree is never touched on this path.
 */
static void drm_sched_rq_update_inf_locked(struct drm_sched_entity *entity,
					   struct drm_sched_rq *rq,
					   ktime_t ts)
{
	u32 seq;

	/*
	 * Both locks need to be grabbed, one to protect from entity->rq change
	 * for entity from within concurrent drm_sched_entity_select_rq and the
	 * other to update the inf list structure.
	 */
	lockdep_assert_held(&entity->lock);
	lockdep_assert_held(&rq->lock);

	drm_sched_rq_remove_inf_locked(entity, rq);

	entity->oldest_job_waiting = ts;

	seq = rq->inf_seq++;
	if (seq == (u32)-1 || (seq & 7) == 7) {
		list_add_tail(&entity->inf_node, &rq->inf_list);
		rq->inf_tail_inserts++;
	} else {
		list_add(&entity->inf_node, &rq->inf_list);
		rq->inf_head_inserts++;
	}
}

/**
 * drm_sched_rq_init - initialize a given run queue struct
 * @sched: scheduler instance to associate with this run queue
 * @rq: scheduler run queue
 *
 * Initializes a scheduler runqueue.
 */
void drm_sched_rq_init(struct drm_gpu_scheduler *sched,
		       struct drm_sched_rq *rq)
{
	spin_lock_init(&rq->lock);
	INIT_LIST_HEAD(&rq->entities);
	INIT_LIST_HEAD(&rq->inf_list);
	rq->inf_seq = 0;
	rq->inf_head_inserts = 0;
	rq->inf_tail_inserts = 0;
	rq->inf_select_hits = 0;
	rq->inf_scan_skips = 0;
	rq->inf_enospc_stops = 0;
	rq->rb_tree_root = RB_ROOT_CACHED;
	rq->sched = sched;
	rq->head_prio = DRM_SCHED_PRIORITY_INVALID;
}

/*
 * Core part of the CFS-like algorithm is that the virtual runtime of lower
 * priority tasks should grow quicker than the higher priority ones, so that
 * when we then schedule entities with the aim of keeping their accumulated
 * virtual time balanced, we can approach fair distribution of GPU time.
 *
 * For converting the real GPU time into virtual we pick some multipliers with
 * the idea to achieve the following GPU time distribution:
 *
 *  - Kernel priority gets roughly 2x GPU time compared to high.
 *  - High gets ~4x relative to normal.
 *  - Normal gets ~8x relative to low.
 */
static const unsigned int vruntime_shift[] = {
	[DRM_SCHED_PRIORITY_KERNEL] = 1,
	[DRM_SCHED_PRIORITY_HIGH]   = 2,
	[DRM_SCHED_PRIORITY_NORMAL] = 4,
	[DRM_SCHED_PRIORITY_LOW]    = 7,
};

static ktime_t
drm_sched_rq_get_min_vruntime(struct drm_sched_rq *rq)
{
	ktime_t vruntime = 0;
	struct rb_node *rb;

	lockdep_assert_held(&rq->lock);

	rb = rb_first_cached(&rq->rb_tree_root);
	if (rb) {
		struct drm_sched_entity *entity =
			rb_entry(rb, typeof(*entity), rb_tree_node);
		struct drm_sched_entity_stats *stats = entity->stats;

		spin_lock(&stats->lock);
		vruntime = stats->vruntime;
		spin_unlock(&stats->lock);
	}

	return vruntime;
}

static void
drm_sched_entity_save_vruntime(struct drm_sched_entity *entity,
			       ktime_t min_vruntime)
{
	struct drm_sched_entity_stats *stats = entity->stats;
	ktime_t vruntime;

	spin_lock(&stats->lock);
	vruntime = stats->vruntime;
	if (min_vruntime && vruntime > min_vruntime)
		vruntime = ktime_sub(vruntime, min_vruntime);
	else
		vruntime = 0;
	stats->vruntime = vruntime;
	spin_unlock(&stats->lock);
}

static ktime_t
drm_sched_entity_restore_vruntime(struct drm_sched_entity *entity,
				  ktime_t min_vruntime,
				  enum drm_sched_priority rq_prio)
{
	struct drm_sched_entity_stats *stats = entity->stats;
	struct drm_gpu_scheduler *sched = entity->rq->sched;
	enum drm_sched_priority prio = entity->priority;
	unsigned long avg_us, sched_avg_us;
	ktime_t vruntime;

	BUILD_BUG_ON(DRM_SCHED_PRIORITY_NORMAL < DRM_SCHED_PRIORITY_HIGH);

	spin_lock(&stats->lock);
	vruntime = stats->vruntime;
	avg_us = ewma_drm_sched_avgtime_read(&stats->avg_job_us);
	/*
	 * Unlocked read of the scheduler average is fine since it is just
	 * heuristics and data type is a natural word size.
	 */
	sched_avg_us = ewma_drm_sched_avgtime_read(&sched->avg_job_us);

	/*
	 * Special handling for entities which were picked from the top of the
	 * queue and are now re-joining the top with another one already there.
	 */
	if (!vruntime && rq_prio != DRM_SCHED_PRIORITY_INVALID) {
		if (prio > rq_prio) {
			/*
			 * Lower priority should not overtake higher when re-
			 * joining at the top of the queue so push it back
			 * somewhere behind the "middle" of the run-queue,
			 * proportional to the scheduler and entity average job
			 * durations.
			 */
			vruntime = us_to_ktime((1 + avg_us + sched_avg_us) <<
					       vruntime_shift[prio]);
		} else if (prio < rq_prio) {
			/*
			 * Higher priority can go first.
			 */
			vruntime = -ns_to_ktime(rq_prio - prio);
		} else {
			/* Favour entity with shorter jobs (interactivity). */
			if (avg_us <= sched_avg_us)
				vruntime = -ns_to_ktime(1);
			else
				vruntime = ns_to_ktime(1);
		}
	}

	/*
	 * Restore saved relative position in the queue.
	 */
	vruntime = ktime_add(min_vruntime, vruntime);

	stats->vruntime = vruntime;
	spin_unlock(&stats->lock);

	return vruntime;
}

static ktime_t drm_sched_entity_update_vruntime(struct drm_sched_entity *entity)
{
	struct drm_sched_entity_stats *stats = entity->stats;
	ktime_t runtime, prev;

	spin_lock(&stats->lock);
	prev = stats->prev_runtime;
	runtime = stats->runtime;
	stats->prev_runtime = runtime;
	runtime = ktime_add_ns(stats->vruntime,
			       ktime_to_ns(ktime_sub(runtime, prev)) <<
			       vruntime_shift[entity->priority]);
	stats->vruntime = runtime;
	spin_unlock(&stats->lock);

	return runtime;
}

static ktime_t drm_sched_entity_get_job_ts(struct drm_sched_entity *entity)
{
	return drm_sched_entity_update_vruntime(entity);
}

/**
 * drm_sched_rq_add_entity - add an entity
 * @entity: scheduler entity
 * @ts: submission timestamp
 *
 * Adds a scheduler entity to the run queue.
 *
 * Return: DRM scheduler selected to handle this entity or NULL if entity has
 * been stopped and cannot be submitted to.
 */
struct drm_gpu_scheduler *
drm_sched_rq_add_entity(struct drm_sched_entity *entity, ktime_t ts)
{
	struct drm_gpu_scheduler *sched;
	struct drm_sched_rq *rq;

	/* Add the entity to the run queue */
	spin_lock(&entity->lock);
	if (entity->stopped) {
		spin_unlock(&entity->lock);

		DRM_ERROR("Trying to push to a killed entity\n");
		return NULL;
	}

	rq = entity->rq;
	spin_lock(&rq->lock);
	sched = rq->sched;

	if (list_empty(&entity->list)) {
		atomic_inc(sched->score);
		list_add_tail(&entity->list, &rq->entities);
	}

	if (drm_sched_policy == DRM_SCHED_POLICY_INFINITY) {
		/* Positional order; reuse caller ts, no ktime_get. */
		drm_sched_rq_update_inf_locked(entity, rq, ts);
		goto inf_added;
	}

	if (drm_sched_policy == DRM_SCHED_POLICY_FAIR) {
		ts = drm_sched_rq_get_min_vruntime(rq);
		ts = drm_sched_entity_restore_vruntime(entity, ts,
						       rq->head_prio);
	} else if (drm_sched_policy == DRM_SCHED_POLICY_RR) {
		ts = entity->rr_ts;
	}

	drm_sched_rq_update_fifo_locked(entity, rq, ts);

inf_added:
	spin_unlock(&rq->lock);
	spin_unlock(&entity->lock);

	return sched;
}

/**
 * drm_sched_rq_remove_entity - remove an entity
 * @rq: scheduler run queue
 * @entity: scheduler entity
 *
 * Removes a scheduler entity from the run queue.
 */
void drm_sched_rq_remove_entity(struct drm_sched_rq *rq,
				struct drm_sched_entity *entity)
{
	lockdep_assert_held(&entity->lock);

	if (list_empty(&entity->list))
		return;

	spin_lock(&rq->lock);

	atomic_dec(rq->sched->score);
	list_del_init(&entity->list);

	drm_sched_rq_remove_fifo_locked(entity, rq);
	drm_sched_rq_remove_inf_locked(entity, rq);

	spin_unlock(&rq->lock);
}

static ktime_t
drm_sched_rq_next_rr_ts(struct drm_sched_rq *rq,
			struct drm_sched_entity *entity)
{
	ktime_t ts;

	lockdep_assert_held(&entity->lock);
	lockdep_assert_held(&rq->lock);

	ts = ktime_add_ns(rq->rr_ts, 1);
	entity->rr_ts = ts;
	rq->rr_ts = ts;

	return ts;
}

/**
 * drm_sched_rq_pop_entity - pops an entity
 * @entity: scheduler entity
 *
 * To be called every time after a job is popped from the entity.
 */
void drm_sched_rq_pop_entity(struct drm_sched_entity *entity)
{
	struct drm_sched_job *next_job;
	struct drm_sched_rq *rq;

	/*
	 * Update the entity's location in the min heap according to
	 * the timestamp of the next job, if any.
	 */
	spin_lock(&entity->lock);
	rq = entity->rq;
	spin_lock(&rq->lock);
	next_job = drm_sched_entity_queue_peek(entity);
	if (drm_sched_policy == DRM_SCHED_POLICY_INFINITY) {
		/* Positional order; preserve ts, no ktime_get. */
		ktime_t ts = entity->oldest_job_waiting;

		if (next_job)
			drm_sched_rq_update_inf_locked(entity, rq, ts);
		else
			drm_sched_rq_remove_inf_locked(entity, rq);
		goto inf_done;
	}
	if (next_job) {
		ktime_t ts;

		if (drm_sched_policy == DRM_SCHED_POLICY_FAIR)
			ts = drm_sched_entity_get_job_ts(entity);
		else if (drm_sched_policy == DRM_SCHED_POLICY_FIFO)
			ts = next_job->submit_ts;
		else
			ts = drm_sched_rq_next_rr_ts(rq, entity);

		drm_sched_rq_update_fifo_locked(entity, rq, ts);
	} else {
		drm_sched_rq_remove_fifo_locked(entity, rq);

		if (drm_sched_policy == DRM_SCHED_POLICY_FAIR) {
			ktime_t min_vruntime;

			min_vruntime = drm_sched_rq_get_min_vruntime(rq);
			drm_sched_entity_save_vruntime(entity, min_vruntime);
		}
	}
inf_done:
	spin_unlock(&rq->lock);
	spin_unlock(&entity->lock);
}

/**
 * drm_sched_rq_select_entity - Select an entity which provides a job to run
 * @sched: the gpu scheduler
 * @rq: scheduler run queue to check.
 *
 * Find oldest waiting ready entity.
 *
 * Return an entity if one is found; return an error-pointer (!NULL) if an
 * entity was ready, but the scheduler had insufficient credits to accommodate
 * its job; return NULL, if no ready entity was found.
 */
struct drm_sched_entity *
drm_sched_rq_select_entity(struct drm_gpu_scheduler *sched,
			   struct drm_sched_rq *rq)
{
	struct rb_node *rb;

	spin_lock(&rq->lock);
	if (drm_sched_policy == DRM_SCHED_POLICY_INFINITY) {
		struct drm_sched_entity *entity, *selected = NULL;

		list_for_each_entry(entity, &rq->inf_list, inf_node) {
			if (!drm_sched_entity_is_ready(entity)) {
				rq->inf_scan_skips++;
				continue;
			}
			/* If we can't queue yet, stop at the first
			 * ready entity just like the rb-tree path below.
			 */
			if (!drm_sched_can_queue(sched, entity)) {
				rq->inf_enospc_stops++;
				spin_unlock(&rq->lock);
				return ERR_PTR(-ENOSPC);
			}

			reinit_completion(&entity->entity_idle);
			rq->inf_select_hits++;
			selected = entity;
			break;
		}
		spin_unlock(&rq->lock);

		return selected;
	}
	for (rb = rb_first_cached(&rq->rb_tree_root); rb; rb = rb_next(rb)) {
		struct drm_sched_entity *entity;

		entity = rb_entry(rb, struct drm_sched_entity, rb_tree_node);
		if (drm_sched_entity_is_ready(entity)) {
			/* If we can't queue yet, preserve the current entity in
			 * terms of fairness.
			 */
			if (!drm_sched_can_queue(sched, entity)) {
				spin_unlock(&rq->lock);
				return ERR_PTR(-ENOSPC);
			}

			reinit_completion(&entity->entity_idle);
			break;
		}
	}
	spin_unlock(&rq->lock);

	return rb ? rb_entry(rb, struct drm_sched_entity, rb_tree_node) : NULL;
}

/*
 * Infinity discipline-native stats, folded into the DRM track.
 * Per-rq plain u64 counters increment under rq->lock at existing
 * hook sites with no new locking; the table below sums at read.
 */

#define LABEL_CAP 32
#define NOTE_CAP 48

/* Value column width, pretty values never exceed this. */
#define INF_VAL_W 8

/* Max schedulers summed in one read; beyond this stats stay partial. */
#define INF_MAX_TRACKED 64

static struct drm_gpu_scheduler *inf_tracked[INF_MAX_TRACKED];
static int inf_tracked_nr;
static DEFINE_MUTEX(inf_tracked_lock);
static struct dentry *inf_dbg_dentry;

/* Copy with ~ truncation, ASCII only. */
static void inf_trunc(const char *src, char *dst, size_t cap)
{
	size_t len = strlen(src);

	if (len <= cap) {
		memcpy(dst, src, len + 1);
	} else if (cap > 0) {
		memcpy(dst, src, cap - 1);
		dst[cap - 1] = '~';
		dst[cap] = '\0';
	} else {
		dst[0] = '\0';
	}
}

/*
 * Integer only pretty for counters: plain below 100000,
 * else scaled K/M/B/T/P/E with one decimal; stays within INF_VAL_W chars.
 */
static void inf_pretty(u64 v, char *buf, size_t sz)
{
	const char *suf = "";
	u64 div = 1;
	u64 whole;
	unsigned int tenth;

	if (v < 100000) {
		snprintf(buf, sz, "%llu", (unsigned long long)v);
		return;
	}
	if (v >= 1000000000000000000ULL) {
		suf = "E";
		div = 1000000000000000000ULL;
	} else if (v >= 1000000000000000ULL) {
		suf = "P";
		div = 1000000000000000ULL;
	} else if (v >= 1000000000000ULL) {
		suf = "T";
		div = 1000000000000ULL;
	} else if (v >= 1000000000ULL) {
		suf = "B";
		div = 1000000000ULL;
	} else if (v >= 1000000ULL) {
		suf = "M";
		div = 1000000ULL;
	} else {
		suf = "K";
		div = 1000ULL;
	}
	whole = v / div;
	if (whole >= 10) {
		snprintf(buf, sz, "%llu%s", (unsigned long long)whole, suf);
		return;
	}
	tenth = (unsigned int)((v % div) * 10 / div);
	snprintf(buf, sz, "%llu.%u%s", (unsigned long long)whole, tenth, suf);
}

static void emit_sep(struct seq_file *m, int lw, int vw, int nw)
{
	int i;

	seq_putc(m, '+');
	for (i = 0; i < lw + 2; i++)
		seq_putc(m, '-');
	seq_putc(m, '+');
	for (i = 0; i < vw + 2; i++)
		seq_putc(m, '-');
	seq_putc(m, '+');
	for (i = 0; i < nw + 2; i++)
		seq_putc(m, '-');
	seq_puts(m, "+\n");
}

static int inf_show(struct seq_file *m, void *v)
{
	static const char *const labels[] = {
		"head_inserts",
		"tail_inserts",
		"select_hits",
		"scan_skips",
		"enospc_stops",
	};
	static const char *const notes[] = {
		"latency head",
		"bulk fifo",
		"scheduler effective",
		"scan efficiency",
		"mem-pressure signal",
	};
	u64 vals[ARRAY_SIZE(labels)] = { 0 };
	char lbuf[ARRAY_SIZE(labels)][LABEL_CAP + 1];
	char nbuf[ARRAY_SIZE(labels)][NOTE_CAP + 1];
	char vbuf[ARRAY_SIZE(labels)][INF_VAL_W + 8];
	int lw, nw;
	const int vw = INF_VAL_W;
	int i, j;
	bool first = true;

	mutex_lock(&inf_tracked_lock);
	for (i = 0; i < inf_tracked_nr; i++) {
		struct drm_gpu_scheduler *sched = inf_tracked[i];
		int k;

		if (!sched || !sched->sched_rq)
			continue;
		for (k = DRM_SCHED_PRIORITY_KERNEL; k < sched->num_rqs; k++) {
			struct drm_sched_rq *rq = sched->sched_rq[k];
			u64 h, t, s, sk, e;

			if (!rq)
				continue;
			spin_lock(&rq->lock);
			h = rq->inf_head_inserts;
			t = rq->inf_tail_inserts;
			s = rq->inf_select_hits;
			sk = rq->inf_scan_skips;
			e = rq->inf_enospc_stops;
			spin_unlock(&rq->lock);
			vals[0] += h;
			vals[1] += t;
			vals[2] += s;
			vals[3] += sk;
			vals[4] += e;
		}
	}
	mutex_unlock(&inf_tracked_lock);

	/* One pass: truncate, pretty-print and measure label/note widths. */
	lw = (int)strlen("counter");
	nw = (int)strlen("note");
	for (j = 0; j < (int)ARRAY_SIZE(labels); j++) {
		inf_trunc(labels[j], lbuf[j], LABEL_CAP);
		inf_trunc(notes[j], nbuf[j], NOTE_CAP);
		inf_pretty(vals[j], vbuf[j], sizeof(vbuf[j]));
		if ((int)strlen(lbuf[j]) > lw)
			lw = (int)strlen(lbuf[j]);
		if ((int)strlen(nbuf[j]) > nw)
			nw = (int)strlen(nbuf[j]);
	}

	emit_sep(m, lw, vw, nw);
	seq_printf(m, "| %-*s | %*s | %-*s |\n", lw, "counter", vw, "value", nw, "note");
	emit_sep(m, lw, vw, nw);
	for (j = 0; j < (int)ARRAY_SIZE(labels); j++) {
		char row[LABEL_CAP + NOTE_CAP + INF_VAL_W + 16];
		int slen = lw + vw + nw + 10;
		int rlen;

		snprintf(row, sizeof(row), "| %-*s | %*s | %-*s |",
			 lw, lbuf[j], vw, vbuf[j], nw, nbuf[j]);
		rlen = (int)strlen(row);
		if (first) {
			char bdr[LABEL_CAP + NOTE_CAP + INF_VAL_W + 16];
			int j1 = lw + 3, j2 = lw + vw + 6;
			int j3 = lw + vw + nw + 9;
			int m;

			WARN_ON(slen != rlen);
			/*
			 * Junction check: every '+' in the separator must
			 * sit exactly under a '|' in data rows. Border math
			 * below mirrors emit_sep on purpose, so any drift
			 * between the two trips the WARN instead of
			 * printing a skewed box.
			 */
			memset(bdr, '-', sizeof(bdr));
			bdr[0] = '+';
			bdr[j1] = '+';
			bdr[j2] = '+';
			bdr[j3] = '+';
			bdr[j3 + 1] = '\0';
			for (m = 0; row[m]; m++)
				WARN_ON((bdr[m] == '+') != (row[m] == '|'));
			first = false;
		}
		seq_printf(m, "%s\n", row);
	}
	emit_sep(m, lw, vw, nw);

	return 0;
}

static int inf_open(struct inode *inode, struct file *file)
{
	return single_open(file, inf_show, NULL);
}

static const struct file_operations inf_fops = {
	.owner = THIS_MODULE,
	.open = inf_open,
	.read = seq_read,
	.llseek = seq_lseek,
	.release = single_release,
};

void drm_sched_inf_track(struct drm_gpu_scheduler *sched)
{
	int i;

	mutex_lock(&inf_tracked_lock);
	for (i = 0; i < inf_tracked_nr; i++) {
		if (inf_tracked[i] == sched)
			break;
	}
	if (i == inf_tracked_nr && inf_tracked_nr < INF_MAX_TRACKED)
		inf_tracked[inf_tracked_nr++] = sched;
	if (!inf_dbg_dentry) {
		inf_dbg_dentry = debugfs_create_file("infinity_drm", 0444,
						     NULL, NULL, &inf_fops);
		if (IS_ERR(inf_dbg_dentry))
			inf_dbg_dentry = NULL;
	}
	mutex_unlock(&inf_tracked_lock);
}

void drm_sched_inf_untrack(struct drm_gpu_scheduler *sched)
{
	int i;

	mutex_lock(&inf_tracked_lock);
	for (i = 0; i < inf_tracked_nr; i++) {
		if (inf_tracked[i] == sched)
			break;
	}
	if (i < inf_tracked_nr) {
		inf_tracked_nr--;
		memmove(&inf_tracked[i], &inf_tracked[i + 1],
			(size_t)(inf_tracked_nr - i) * sizeof(inf_tracked[0]));
	}
	if (inf_tracked_nr == 0 && inf_dbg_dentry) {
		debugfs_remove(inf_dbg_dentry);
		inf_dbg_dentry = NULL;
	}
	mutex_unlock(&inf_tracked_lock);
}
