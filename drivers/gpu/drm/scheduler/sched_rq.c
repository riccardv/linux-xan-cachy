// SPDX-License-Identifier: MIT
/* Copyright 2015 Advanced Micro Devices, Inc. */
/* Copyright (c) 2025 Valve Corporation */

#include <linux/rbtree.h>

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
	ktime_t runtime, prev, now;
	s64 delta_ns;

	spin_lock(&stats->lock);
	prev = stats->prev_runtime;
	runtime = stats->runtime;
	stats->prev_runtime = runtime;
	delta_ns = ktime_to_ns(ktime_sub(runtime, prev));

	/* Infinity: EMA climb on the accounted GPU time.  The delta is the
	 * duration of the job(s) completed since the last fold; climbing the
	 * EMA here keeps all gpu_time_* updates in a single lock domain
	 * (stats->lock) -- the completion path and the vruntime fold can no
	 * longer race on the accounting state.
	 */
	if (delta_ns > 0) {
		u64 climb_ns = delta_ns;

		if (climb_ns > INFINITY_GPU_EMA_CLIMB_NS)
			climb_ns = INFINITY_GPU_EMA_CLIMB_NS;

		stats->gpu_time_ema += div64_u64(
			(INFINITY_GPU_EMA_CLIMB_NS - stats->gpu_time_ema) *
			climb_ns * INFINITY_GPU_EMA_ALPHA,
			INFINITY_GPU_EMA_CLIMB_NS * (1ULL << 8));
	}

	/* Infinity: EMA decay and submission-interval tracking.  The gap
	 * since the last fold is the wall-clock idle time of the entity;
	 * decay the EMA by half-lives so a long-idle entity returns to the
	 * fully-interactive state, and record the submission cadence for
	 * job-type awareness.
	 */
	now = ktime_get();
	if (stats->gpu_time_last_active) {
		u64 idle_ns = ktime_to_ns(ktime_sub(now,
					stats->gpu_time_last_active));
		u64 periods = div64_u64(idle_ns,
				INFINITY_GPU_EMA_HALFLIFE_NS);

		if (periods > 63)
			stats->gpu_time_ema = 0;
		else if (periods)
			stats->gpu_time_ema >>= periods;

		stats->gpu_last_submit_interval = idle_ns;
		if (periods)
			atomic64_inc(this_cpu_ptr(&infinity_gpu_idle_compensations));
	} else {
		stats->gpu_last_submit_interval = 0;
	}
	stats->gpu_time_last_active = now;

	/* Infinity: EMA burst penalty.  Busy entities (high EMA) accrue
	 * virtual time faster, so they drift back in the queue and the GPU
	 * stays available to interactive entities.  Compositor-like
	 * entities with fast submission intervals (every 1-8 ms) get the
	 * burst penalty reduced by 25% (the code keeps 75% of it) so the
	 * desktop stays smooth while batch workloads keep the full penalty.
	 */
	if (stats->gpu_time_ema > 0 && delta_ns > 0) {
		u64 ema_pct = div64_u64(stats->gpu_time_ema * 100ULL,
					INFINITY_GPU_EMA_CLIMB_NS);
		s64 boost_ns;

		if (ema_pct > 100)
			ema_pct = 100;

		boost_ns = delta_ns * ema_pct / 100;
		if (stats->gpu_last_submit_interval > 0 &&
		    stats->gpu_last_submit_interval < INFINITY_GPU_FAST_SUBMIT_NS)
			boost_ns -= boost_ns >> 2;

		delta_ns += boost_ns >> 1;
	}

	/* Infinity: CPU-to-GPU cross-scheduler coupling.  Read the owning
	 * task's CPU-side interactivity signals via the entity's
	 * infinity_pid: futex_waiting and a zero CPU EMA mark an
	 * interactive task whose GPU submissions should not be treated as
	 * bulk load.  Each signal halves this fold's vruntime growth, so
	 * the entity drifts toward the front of the queue while its task
	 * stays interactive.  The reduction is capped at 75% (both signals
	 * present) so the fold's vruntime growth never freezes -- a
	 * fully-frozen fold would let an interactive entity monopolize the
	 * queue.
	 *
	 * RCU-safe: pid_task() under rcu_read_lock() returns NULL if the
	 * task has exited, leaving the growth unchanged.
	 */
	if (entity->infinity_pid) {
		struct task_struct *p;
		u32 coupling = 0;

		rcu_read_lock();
		p = pid_task(entity->infinity_pid, PIDTYPE_PID);
		/* Infinity: the infinity_pid resolves to the process group
		 * leader; only fair-class, non-idle-policy leaders
		 * participate in CPU-to-GPU coupling (see
		 * infinity_is_interactive_candidate()).  Non-CFS leaders
		 * keep coupling == 0.
		 */
		if (p && infinity_is_interactive_candidate(p)) {
			if (READ_ONCE(p->infinity.futex_waiting))
				coupling++;
			if (READ_ONCE(p->infinity.ema) == 0)
				coupling++;
		}
		rcu_read_unlock();

		if (coupling) {
			/* Cap the reduction at 75%: with both signals (coupling == 2)
			 * the raw 100% reduction would freeze the fold's vruntime
			 * growth and let an interactive entity monopolize the queue.
			 */
			u32 pct = min(50 * coupling, 75);

			if (delta_ns < 0)
				delta_ns = 0;
			delta_ns = delta_ns * (100 - pct) / 100;
			atomic64_inc(this_cpu_ptr(&infinity_gpu_cpu_coupling_activations));
		}
	}

	runtime = ktime_add_ns(stats->vruntime,
			       delta_ns << vruntime_shift[entity->priority]);
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

	if (drm_sched_policy == DRM_SCHED_POLICY_FAIR) {
		ts = drm_sched_rq_get_min_vruntime(rq);
		ts = drm_sched_entity_restore_vruntime(entity, ts,
						       rq->head_prio);
	} else if (drm_sched_policy == DRM_SCHED_POLICY_RR) {
		ts = entity->rr_ts;
	}

	drm_sched_rq_update_fifo_locked(entity, rq, ts);

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

			/* Infinity: GPU-to-CPU feedback.  Increment
			 * gpu_passovers on the K nearest right-neighbors in
			 * the rbtree (higher vruntime), so their owning
			 * tasks' CPU-side infinity_wakeup() can detect
			 * GPU-side starvation and accelerate EMA decay.
			 * The K nearest entities are the closest to the next
			 * pick (most relevant for starvation feedback);
			 * entities deeper than K are credited on subsequent
			 * selections as the winner moves.  This bounds the
			 * credited work (and the locked atomics) to K
			 * entities per selection; the walk may still pass
			 * over ineligible nodes beyond K, but the expensive
			 * per-entity accounting is bounded.
			 */
			if (entity->infinity_pid) {
				struct rb_node *rb2;
				int credited = 0;

				rcu_read_lock();
				for (rb2 = rb_next(rb); rb2;
				     rb2 = rb_next(rb2)) {
					struct drm_sched_entity *e2;
					struct task_struct *p;

					e2 = rb_entry(rb2, struct drm_sched_entity,
						      rb_tree_node);
					if (!drm_sched_entity_is_ready(e2))
						continue;
					if (!e2->infinity_pid)
						continue;

					p = pid_task(e2->infinity_pid,
						     PIDTYPE_PID);
					/* Infinity: only fair-class leaders
					 * accumulate GPU passovers (see
					 * infinity_is_interactive_candidate()).
					 */
					if (p &&
					    infinity_is_interactive_candidate(p)) {
						atomic_inc(&p->infinity.gpu_passovers);
						atomic64_inc(this_cpu_ptr(
							&infinity_gpu_passover_boosts));
						++credited;
					}
					if (credited >=
					    INFINITY_GPU_PASSOVER_MAX_ENTITIES)
						break;
				}
				rcu_read_unlock();
			}
			break;
		}
	}
	spin_unlock(&rq->lock);

	return rb ? rb_entry(rb, struct drm_sched_entity, rb_tree_node) : NULL;
}
