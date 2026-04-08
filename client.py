"""
client.py — Python client for the DevOps Incident Response Simulator.

Provides a clean interface to talk to the environment server
over HTTP. Used by inference.py to interact with the environment
during the agent loop.

Usage:
    from client import DevOpsEnvClient, IncidentAction

    client = DevOpsEnvClient(base_url="http://localhost:8000")

    obs  = client.reset(task_id="task_1")
    result = client.step(IncidentAction(
        action_type="rollback",
        target_service="payment-service",
        reasoning="Saw NullPointerException in logs"
    ))
    state = client.state()
    client.close()
"""

import requests
from typing import Optional

# Re-export models so callers only need to import from client.py
from models import (
    IncidentAction,
    LogObservation,
    StepResult,
    EpisodeState,
)


class DevOpsEnvClient:
    """
    HTTP client for the DevOps Incident Response Simulator.

    Wraps all server endpoints and parses responses into
    typed Pydantic models.
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        """
        Args:
            base_url : URL where the environment server is running.
                       Locally: "http://localhost:8000"
                       On HF Spaces: "https://your-username-devops-simulator.hf.space"
            timeout  : Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        self.session  = requests.Session()

    # ── Core API ─────────────────────────────────────────────────────────────

    def health(self) -> dict:
        """
        Ping the server. Returns {"status": "ok"} if healthy.
        Raises ConnectionError if the server is not reachable.
        """
        try:
            resp = self.session.get(
                f"{self.base_url}/health",
                timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot reach environment server at {self.base_url}. "
                "Make sure the server is running (python server/app.py)."
            )

    def reset(self, task_id: Optional[str] = None) -> LogObservation:
        """
        Start a new episode.

        Args:
            task_id: "task_1" (easy) | "task_2" (medium) | "task_3" (hard)
                     Pass None to let the server pick randomly.

        Returns:
            LogObservation with the initial logs and pipeline status.
        """
        payload = {"task_id": task_id} if task_id else {}
        resp = self.session.post(
            f"{self.base_url}/reset",
            json=payload,
            timeout=self.timeout,
        )
        self._raise_for_status(resp)
        return LogObservation(**resp.json())

    def step(self, action: IncidentAction) -> StepResult:
        """
        Submit a remediation action to the environment.

        Args:
            action: IncidentAction with action_type, target_service, etc.

        Returns:
            StepResult with new observation, reward (0.0–1.0), done flag,
            and human-readable feedback.
        """
        # Support both Pydantic v2 (model_dump) and v1 (dict).
        payload = action.model_dump() if hasattr(action, "model_dump") else action.dict()

        resp = self.session.post(
            f"{self.base_url}/step",
            json=payload,
            timeout=self.timeout,
        )
        self._raise_for_status(resp)
        return StepResult(**resp.json())

    def state(self) -> EpisodeState:
        """
        Get current episode metadata.

        Returns:
            EpisodeState with episode_id, step_count, cumulative_reward, etc.
        """
        resp = self.session.get(
            f"{self.base_url}/state",
            timeout=self.timeout,
        )
        self._raise_for_status(resp)
        return EpisodeState(**resp.json())

    def list_tasks(self) -> list[dict]:
        """
        Get the list of all available tasks with difficulty and description.

        Returns:
            List of task dicts: [{task_id, difficulty, description}, ...]
        """
        resp = self.session.get(
            f"{self.base_url}/tasks",
            timeout=self.timeout,
        )
        self._raise_for_status(resp)
        return resp.json()["tasks"]

    # ── Context manager support ───────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        """Close the underlying HTTP session."""
        self.session.close()

    # ── Helper ───────────────────────────────────────────────────────────────

    def _raise_for_status(self, resp: requests.Response):
        """
        Raise a clear error if the server returned a non-2xx status.
        Includes the server's error message if available.
        """
        if not resp.ok:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise RuntimeError(
                f"Server returned {resp.status_code}: {detail}"
            )


# ---------------------------------------------------------------------------
# Quick manual test — run this file directly to verify the client works
# (server must already be running: python server/app.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("DevOps Env Client — Manual Test")
    print("=" * 60)

    client = DevOpsEnvClient(base_url="http://localhost:8000")

    # 1. Health check
    print("\n[1] Health check...")
    health = client.health()
    print(f"    {health}")

    # 2. List tasks
    print("\n[2] Available tasks...")
    tasks = client.list_tasks()
    for t in tasks:
        print(f"    [{t['difficulty'].upper()}] {t['task_id']}: {t['description'][:60]}...")

    # 3. Run through all 3 tasks with a correct action each time
    correct_answers = {
        "task_1": IncidentAction(
            action_type="rollback",
            target_service="payment-service",
            reasoning="NullPointerException clearly points to payment-service"
        ),
        "task_2": IncidentAction(
            action_type="rollback",
            target_service="api-gateway",
            reasoning="CrashLoopBackOff on api-gateway during deploy"
        ),
        "task_3": IncidentAction(
            action_type="restart",
            target_service="database",
            reasoning="DB deadlock is the root cause of all downstream failures"
        ),
    }

    total_score = 0.0

    for task_id, action in correct_answers.items():
        print(f"\n{'─' * 50}")
        print(f"[Task] {task_id}")

        obs = client.reset(task_id=task_id)
        print(f"  Status  : {obs.pipeline_status}")
        print(f"  Task    : {obs.task_description[:70]}...")
        print(f"  Services: {obs.active_services}")

        result = client.step(action)
        print(f"  Action  : {action.action_type} on '{action.target_service}'")
        print(f"  Reward  : {result.reward}")
        print(f"  Feedback: {result.feedback}")
        print(f"  Done    : {result.done}")

        state = client.state()
        print(f"  Episode : {state.episode_id[:8]}...")
        print(f"  Steps   : {state.step_count}/{state.max_steps}")

        total_score += result.reward

    print(f"\n{'=' * 60}")
    print(f"Total score across all tasks: {total_score:.2f} / 3.00")
    print(f"Average reward: {total_score / 3:.2f}")
    print("=" * 60)

    client.close()