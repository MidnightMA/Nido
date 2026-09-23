"""FIFO task queue for serialized multi-step desktop agent execution in realtime mode."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from nido.logging import get_logger

logger = get_logger("nido.realtime.queue")


@dataclass
class AgentTask:
    """Represents a finalized spoken command queued for desktop execution."""

    task_id: str
    goal: str
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentTaskQueue:
    """Bounded FIFO task queue with serialized worker thread.

    Ensures desktop agent tasks run one at a time sequentially while microphone capture
    and streaming STT continue in parallel.
    """

    def __init__(self, max_pending: int = 8) -> None:
        self.max_pending = max(1, max_pending)
        self._queue: queue.Queue[Optional[AgentTask]] = queue.Queue(maxsize=self.max_pending)
        self._task_counter = 0
        self._lock = threading.Lock()

        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._current_task: Optional[AgentTask] = None

        # Hooks
        self.on_task_queued: Optional[Callable[[AgentTask, int], None]] = None
        self.on_task_started: Optional[Callable[[AgentTask], None]] = None
        self.on_task_completed: Optional[Callable[[AgentTask, Any], None]] = None
        self.on_task_failed: Optional[Callable[[AgentTask, str], None]] = None
        self.on_queue_full: Optional[Callable[[str], None]] = None

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    @property
    def is_executing(self) -> bool:
        with self._lock:
            return self._current_task is not None

    @property
    def current_task(self) -> Optional[AgentTask]:
        with self._lock:
            return self._current_task

    def enqueue(self, goal: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[AgentTask]:
        """Enqueue a new spoken utterance goal. Returns AgentTask or None if queue full."""
        cleaned_goal = goal.strip()
        if not cleaned_goal:
            return None

        with self._lock:
            self._task_counter += 1
            task_id = f"rt-{self._task_counter:04d}"

        task = AgentTask(task_id=task_id, goal=cleaned_goal, metadata=metadata or {})

        try:
            self._queue.put_nowait(task)
            logger.info(f"Queued desktop task {task.task_id}: '{task.goal}' (pending={self._queue.qsize()})")
            if self.on_task_queued:
                try:
                    self.on_task_queued(task, self._queue.qsize())
                except Exception as e:
                    logger.error(f"Error in on_task_queued hook: {e}")
            return task
        except queue.Full:
            logger.warning(f"Task queue full ({self.max_pending} commands). Rejecting: '{task.goal}'")
            if self.on_queue_full:
                try:
                    self.on_queue_full(task.goal)
                except Exception as e:
                    logger.error(f"Error in on_queue_full hook: {e}")
            return None

    def start_worker(self, executor: Callable[[AgentTask], Any]) -> None:
        """Start the background worker thread that serializes execution."""
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            args=(executor,),
            name="RealtimeAgentWorker",
            daemon=True,
        )
        self._worker_thread.start()
        logger.debug("RealtimeAgentWorker thread started.")

    def stop_worker(self) -> None:
        """Stop worker thread and wait for completion."""
        if not self._running:
            return
        self._running = False
        try:
            self._queue.put_nowait(None)  # Sentinel to unblock get()
        except queue.Full:
            pass

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        logger.debug("RealtimeAgentWorker thread stopped.")

    def _worker_loop(self, executor: Callable[[AgentTask], Any]) -> None:
        while self._running:
            try:
                task = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if task is None:
                break

            with self._lock:
                self._current_task = task

            logger.info(f"Executing task {task.task_id}: '{task.goal}'")
            if self.on_task_started:
                try:
                    self.on_task_started(task)
                except Exception as e:
                    logger.error(f"Error in on_task_started hook: {e}")

            try:
                result = executor(task)
                logger.info(f"Completed task {task.task_id}: '{task.goal}'")
                if self.on_task_completed:
                    try:
                        self.on_task_completed(task, result)
                    except Exception as e:
                        logger.error(f"Error in on_task_completed hook: {e}")
            except Exception as e:
                logger.error(f"Task {task.task_id} failed with error: {e}")
                if self.on_task_failed:
                    try:
                        self.on_task_failed(task, str(e))
                    except Exception as err:
                        logger.error(f"Error in on_task_failed hook: {err}")
            finally:
                with self._lock:
                    self._current_task = None
                self._queue.task_done()
