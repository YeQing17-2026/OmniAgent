"""RL Rollout Worker.

Async rollout worker that bridges the FastAPI proxy server and the
SLIME training loop. Data production is request-driven: external
agents send requests through the proxy, which collects training
samples.

Only activates when model_provider is "vllm" or "sglang".

"""

import asyncio
import atexit
import queue
import structlog
import threading
import time
from typing import Dict, List, Optional, Tuple

logger = structlog.get_logger(__name__)

from .api_server import RLAPIServer


_global_worker: Optional["AsyncRolloutWorker"] = None
_worker_lock = threading.Lock()


def get_global_worker(args, data_buffer) -> "AsyncRolloutWorker":
    """Get or create the singleton AsyncRolloutWorker."""
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            _global_worker = AsyncRolloutWorker(args, data_buffer)
            _global_worker.start()
        return _global_worker


def stop_global_worker():
    """Stop the global rollout worker. Registered as atexit handler."""
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


class AsyncRolloutWorker:
    """Bridges the RLAPIServer proxy and the SLIME training loop.

    The output_queue receives (group_index, [Sample]) tuples from the
    proxy's request handlers. The SLIME trainer drains this queue
    to collect batches of training data.
    """

    def __init__(self, args, data_buffer):
        self.args = args
        self.data_buffer = data_buffer
        self.running = True
        self.output_queue: queue.Queue = queue.Queue(maxsize=100000)
        self.worker_thread: Optional[threading.Thread] = None
        self._submission_enabled = threading.Event()
        self._server = RLAPIServer(
            args=args,
            output_queue=self.output_queue,
            submission_enabled=self._submission_enabled,
        )

    async def continuous_worker_loop(self):
        """Keepalive loop. Data production is request-driven in FastAPI handlers."""
        while self.running:
            await asyncio.sleep(1.0)

    def worker_thread_func(self):
        """Entry point for the daemon async event loop thread."""
        asyncio.run(self.continuous_worker_loop())

    def start(self):
        """Start the proxy server and the async event loop thread."""
        self._server.start()
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(
                target=self.worker_thread_func, daemon=True,
            )
            self.worker_thread.start()

    def stop(self):
        """Stop the rollout worker, proxy server, and async loop."""
        self.running = False
        self._submission_enabled.clear()
        self._server.stop()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

    def pause_submission(self):
        """Pause submission (returns 503 to all requests). Called during weight update."""
        if self._submission_enabled.is_set():
            self._submission_enabled.clear()
            self._server.purge_record_files()
            logger.info("rl_worker submission paused")

    def resume_submission(self):
        """Resume submission (accepts requests again)."""
        if not self._submission_enabled.is_set():
            self._submission_enabled.set()
            logger.info("rl_worker submission resumed")

    def get_completed_groups(self) -> List[Tuple]:
        """Drain completed (group_index, [Sample]) tuples from the output queue."""
        completed: List[Tuple] = []
        while True:
            try:
                completed.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return completed

    def get_queue_size(self) -> int:
        """Return the current size of the output queue."""
        return self.output_queue.qsize()


async def _drain_output_queue(args, worker: AsyncRolloutWorker) -> list:
    """Block until rollout_batch_size groups have been collected.

    Polls the output queue every 50ms. Skips groups where any sample
    was ABORTED. Prints progress every 30 seconds.
    """
    from slime.utils.types import Sample

    target_data_size = args.rollout_batch_size
    data: list = []
    completed_groups: Dict[int, list] = {}
    start = time.time()
    last_progress = start

    while len(data) < target_data_size:
        completed = worker.get_completed_groups()
        if completed:
            last_progress = time.time()
            for group_id, group in completed:
                completed_groups[group_id] = group

        for group_id in list(completed_groups.keys()):
            if len(data) >= target_data_size:
                break
            group = completed_groups.pop(group_id)
            if any(
                sample.status == Sample.Status.ABORTED
                for sample in group
            ):
                continue
            data.append(group)

        if time.time() - last_progress > 30:
            logger.info(
                "rl_worker waiting for samples",
                current=len(data),
                target=target_data_size,
                queue_size=worker.get_queue_size(),
            )
            last_progress = time.time()

        if len(data) < target_data_size:
            await asyncio.sleep(0.05)

    data.sort(
        key=lambda group: group[0].index
        if group and group[0].index is not None
        else -1,
    )
    logger.info(
        "rl_worker drained groups",
        count=len(data),
        elapsed_seconds=round(time.time() - start, 2),
    )
    return data


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """SLIME rollout function entry point.

    Referenced by: --rollout-function-path omniagent.rl.rollout.generate_rollout

    For training:
    1. Resume submission (open proxy to requests)
    2. Drain output queue until batch_size groups collected
    3. Pause submission (block during weight update)
    4. Return collected samples + PRM eval metrics

    For evaluation:
    Delegates to SLIME's eval_rollout.
    """
    from slime.rollout.base_types import RolloutFnTrainOutput
    from slime.rollout.sglang_rollout import eval_rollout
    from slime.utils.async_utils import run

    worker = get_global_worker(args, data_buffer)

    if evaluation:
        eval_output, _ = run(eval_rollout(args, rollout_id))
        return eval_output

    worker._server.reset_eval_scores()
    worker.resume_submission()
    completed_samples = run(_drain_output_queue(args, worker))
    worker.pause_submission()

    extra_metrics = None
    eval_scores = worker._server.drain_eval_scores()
    if eval_scores:
        extra_metrics = {
            "rollout/prm_eval_score": sum(eval_scores) / len(eval_scores),
        }
        logger.info(
            "rl_worker prm_eval_score",
            score=round(extra_metrics['rollout/prm_eval_score'], 4),
            n=len(eval_scores),
        )

    return RolloutFnTrainOutput(samples=completed_samples, metrics=extra_metrics)


atexit.register(stop_global_worker)
