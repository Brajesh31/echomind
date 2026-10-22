"""EchoMind — Background Media Processing Job Manager.

Provides a thread-safe, in-process background job system for long-running video
transcription and vector indexing, tracking granular stages, progress, and errors.
"""

from __future__ import annotations

import datetime
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from backend.models.schemas import JobResponse, JobStatus
from backend.services.security import safe_token

logger = logging.getLogger("echomind.job_manager")


class ProcessingJob:
    """Represents the lifecycle and execution state of a media processing task."""

    def __init__(self, job_id: str, session_id: str, source: str):
        self.job_id = job_id
        self.session_id = session_id
        self.source = source
        self.status = JobStatus.QUEUED
        self.stage = "queued"
        self.progress_percent = 0
        self.message = "Job queued for processing."
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.created_at = now
        self.updated_at = now
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None

    def update_stage(self, status: JobStatus, stage: str, message: str, progress: Optional[int] = None):
        """Update job stage, status, message, and progress percentage."""
        self.status = status
        self.stage = stage
        self.message = message
        if progress is not None:
            self.progress_percent = progress
        self.updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        logger.info("[Job %s] Stage '%s' (%s%%): %s", self.job_id, stage, self.progress_percent, message)

    def complete(self, result: Dict[str, Any]):
        """Mark job as completed with final output payload."""
        self.status = JobStatus.COMPLETED
        self.stage = "ready"
        self.progress_percent = 100
        self.message = "Video intelligence processing completed successfully."
        self.result = result
        self.updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        logger.info("[Job %s] Completed successfully.", self.job_id)

    def fail(self, user_facing_error: str, technical_error: Optional[str] = None):
        """Mark job as failed with sanitized error message."""
        self.status = JobStatus.FAILED
        self.stage = "failed"
        self.message = user_facing_error
        self.error = user_facing_error
        self.updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        logger.error("[Job %s] Failed: %s (technical: %s)", self.job_id, user_facing_error, technical_error)

    def to_schema(self) -> JobResponse:
        """Convert job state to Pydantic JobResponse."""
        return JobResponse(
            job_id=self.job_id,
            session_id=self.session_id,
            status=self.status,
            stage=self.stage,
            progress_percent=self.progress_percent,
            message=self.message,
            created_at=self.created_at,
            updated_at=self.updated_at,
            result=self.result,
            error=self.error,
        )


class JobManager:
    """Coordinates asynchronous processing jobs using an in-process thread pool."""

    def __init__(self, max_workers: int = 2):
        self._jobs: Dict[str, ProcessingJob] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="EchoMind-Job")

    def get_job(self, job_id: str) -> Optional[ProcessingJob]:
        """Retrieve job by its unique ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> List[ProcessingJob]:
        """List all tracked jobs."""
        with self._lock:
            return list(self._jobs.values())

    def create_job(self, source: str, session_id: str) -> ProcessingJob:
        """Instantiate and register a new processing job."""
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        clean_sid = safe_token(session_id)
        job = ProcessingJob(job_id=job_id, session_id=clean_sid, source=source)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def find_active_job_for_source(self, source: str) -> Optional[ProcessingJob]:
        """Return an in-flight job for the same source to guarantee idempotency."""
        with self._lock:
            for job in self._jobs.values():
                if job.source == source and job.status not in (JobStatus.COMPLETED, JobStatus.FAILED):
                    return job
        return None

    def submit_job(
        self,
        user_id: str,
        source_type: str,
        source_input: str,
        worker_func: Optional[Any] = None,
        append: bool = False,
    ) -> str:
        """Submit a background job; returns job_id. Idempotent for in-flight tasks."""
        active = self.find_active_job_for_source(source_input)
        if active:
            logger.info("Found existing active job %s for source '%s'. Reusing.", active.job_id, source_input)
            return active.job_id

        job = self.create_job(source=source_input, session_id=user_id)

        if worker_func:
            def _custom_wrapper():
                try:
                    job.update_stage(JobStatus.PROCESSING, "processing", "Processing media...", progress=10)
                    def progress_cb(status, msg):
                        job.update_stage(status, str(status).lower(), msg)
                    result = worker_func(job.session_id, job.source, progress_cb)
                    job.complete(result)
                except Exception as exc:
                    logger.error("Job %s worker failed: %s", job.job_id, exc)
                    job.fail(user_facing_error=str(exc), technical_error=repr(exc))

            self._executor.submit(_custom_wrapper)
        else:
            self._executor.submit(
                self._execute_job,
                job=job,
                source=source_input,
                session_id=user_id,
                append=append,
                language="english",
                summary_type="comprehensive",
            )
        return job.job_id

    def submit_processing_task(
        self,
        source: str,
        session_id: str,
        append: bool = False,
        language: str = "english",
        summary_type: str = "comprehensive",
    ) -> JobResponse:
        """Register job and dispatch to background thread pool with idempotency check."""
        active = self.find_active_job_for_source(source)
        if active:
            logger.info("Found existing active job %s for source '%s'. Reusing.", active.job_id, source)
            return active.to_schema()

        job = self.create_job(source=source, session_id=session_id)

        # Dispatch execution
        self._executor.submit(
            self._execute_job,
            job=job,
            source=source,
            session_id=session_id,
            append=append,
            language=language,
            summary_type=summary_type,
        )

        return job.to_schema()

    def _execute_job(
        self,
        job: ProcessingJob,
        source: str,
        session_id: str,
        append: bool,
        language: str,
        summary_type: str,
    ):
        """Worker executing the end-to-end video intelligence pipeline."""
        from Audio_Preprocessing.main import run_pipeline

        try:
            job.update_stage(JobStatus.VALIDATING, "validating", "Validating video source and security parameters...", progress=5)

            # Define a progress callback for internal pipeline stages
            def progress_callback(status: JobStatus, stage: str, msg: str, pct: int):
                job.update_stage(status, stage, msg, progress=pct)

            session = run_pipeline(
                source=source,
                transcript=None,
                user_id=session_id,
                append=append,
                language=language,
                summary_type=summary_type,
                progress_callback=progress_callback,
            )

            job.complete(session)

        except ValueError as exc:
            job.fail(user_facing_error=str(exc), technical_error=repr(exc))
        except RuntimeError as exc:
            user_msg = str(exc) if "ffmpeg" in str(exc).lower() or "mistral" in str(exc).lower() else "Video processing failed. Please check the video source."
            job.fail(user_facing_error=user_msg, technical_error=repr(exc))
        except Exception as exc:
            logger.exception("Unexpected exception during job %s execution: %s", job.job_id, exc)
            job.fail(user_facing_error="An unexpected error occurred while processing the video.", technical_error=repr(exc))


# Global singleton job manager
job_manager = JobManager()
