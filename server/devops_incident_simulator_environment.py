"""
server/your_environment.py — Core environment logic.

Implements reset(), step(), and state() for the DevOps Incident
Response Simulator. The environment generates realistic pipeline
logs and evaluates the AI agent's remediation actions.
"""

import sys
import os
import uuid
import random
from typing import Optional

# Make sure models.py (one level up) is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import (
    IncidentAction,
    LogObservation,
    StepResult,
    EpisodeState,
)


# ---------------------------------------------------------------------------
# SCENARIO LIBRARY
# Each scenario is a dict describing a real-world DevOps incident.
# The environment picks one on reset() and the agent must fix it.
# ---------------------------------------------------------------------------

SCENARIOS = {
    # ── EASY (Task 1) ────────────────────────────────────────────────────────
    "task_1": [
        {
            "name": "null_pointer_in_payment_service",
            "difficulty": "easy",
            "root_cause_service": "payment-service",
            "correct_actions": ["rollback", "patch"],
            "logs": [
                "[INFO]  2024-04-01 10:00:01 auth-service        — Request received: POST /login",
                "[INFO]  2024-04-01 10:00:02 auth-service        — User authenticated successfully",
                "[INFO]  2024-04-01 10:00:03 payment-service     — Processing payment for order #8821",
                "[ERROR] 2024-04-01 10:00:04 payment-service     — NullPointerException at PaymentProcessor.java:142",
                "[ERROR] 2024-04-01 10:00:04 payment-service     — Transaction aborted. order_id=8821",
                "[WARN]  2024-04-01 10:00:05 payment-service     — Retry attempt 1/3 failed",
                "[WARN]  2024-04-01 10:00:06 payment-service     — Retry attempt 2/3 failed",
                "[ERROR] 2024-04-01 10:00:07 payment-service     — Max retries exceeded. Service degraded.",
                "[INFO]  2024-04-01 10:00:08 frontend            — Received 500 from payment-service",
            ],
            "task_description": (
                "The payment-service is throwing a NullPointerException and failing "
                "all transactions. Identify the faulty service and fix it."
            ),
        },
        {
            "name": "oom_kill_in_worker_service",
            "difficulty": "easy",
            "root_cause_service": "worker-service",
            "correct_actions": ["restart", "rollback"],
            "logs": [
                "[INFO]  2024-04-01 11:00:01 scheduler           — Triggering daily batch job",
                "[INFO]  2024-04-01 11:00:02 worker-service      — Batch job started, processing 50000 records",
                "[WARN]  2024-04-01 11:00:10 worker-service      — Memory usage at 85%",
                "[WARN]  2024-04-01 11:00:15 worker-service      — Memory usage at 95%",
                "[ERROR] 2024-04-01 11:00:18 worker-service      — OOMKilled: Container exceeded memory limit (512Mi)",
                "[ERROR] 2024-04-01 11:00:18 worker-service      — Process terminated unexpectedly",
                "[INFO]  2024-04-01 11:00:19 scheduler           — Batch job failed. Exit code: 137",
            ],
            "task_description": (
                "The worker-service was OOM-killed during a batch job. "
                "Identify the service and take the correct remediation action."
            ),
        },
    ],

    # ── MEDIUM (Task 2) ──────────────────────────────────────────────────────
    "task_2": [
        {
            "name": "failed_deploy_partial_rollout",
            "difficulty": "medium",
            "root_cause_service": "api-gateway",
            "correct_actions": ["rollback"],
            "logs": [
                "[INFO]  2024-04-01 12:00:01 ci-cd-pipeline      — Starting deployment v2.4.1 → v2.4.2",
                "[INFO]  2024-04-01 12:00:05 ci-cd-pipeline      — Building Docker image... success",
                "[INFO]  2024-04-01 12:00:20 ci-cd-pipeline      — Pushing image to registry... success",
                "[INFO]  2024-04-01 12:00:25 ci-cd-pipeline      — Rolling out to 3 pods (1/3 updated)",
                "[INFO]  2024-04-01 12:00:30 ci-cd-pipeline      — Rolling out to 3 pods (2/3 updated)",
                "[ERROR] 2024-04-01 12:00:35 api-gateway         — CrashLoopBackOff: pod api-gateway-v2.4.2-abc123",
                "[ERROR] 2024-04-01 12:00:36 api-gateway         — Readiness probe failed: HTTP 503",
                "[ERROR] 2024-04-01 12:00:37 api-gateway         — Liveness probe failed: connection refused",
                "[WARN]  2024-04-01 12:00:38 ci-cd-pipeline      — Rollout stalled. 2/3 pods healthy.",
                "[ERROR] 2024-04-01 12:00:40 api-gateway         — New pod failing to start. Env var API_SECRET missing.",
                "[WARN]  2024-04-01 12:00:42 frontend            — Elevated error rate: 34% of requests returning 503",
                "[INFO]  2024-04-01 12:00:45 ci-cd-pipeline      — Deployment FAILED. Manual intervention required.",
            ],
            "task_description": (
                "A deployment of api-gateway v2.4.2 is stuck in CrashLoopBackOff "
                "due to a missing environment variable. The pipeline is partially "
                "deployed. You must rollback to restore service."
            ),
        },
        {
            "name": "config_map_corruption",
            "difficulty": "medium",
            "root_cause_service": "auth-service",
            "correct_actions": ["rollback", "redeploy"],
            "logs": [
                "[INFO]  2024-04-01 13:00:01 ci-cd-pipeline      — ConfigMap update applied: auth-service-config v3",
                "[INFO]  2024-04-01 13:00:02 auth-service        — Reloading configuration...",
                "[ERROR] 2024-04-01 13:00:03 auth-service        — Failed to parse JWT_SECRET: invalid base64 encoding",
                "[ERROR] 2024-04-01 13:00:03 auth-service        — Authentication module failed to initialise",
                "[ERROR] 2024-04-01 13:00:04 auth-service        — All login requests failing: 500 Internal Server Error",
                "[WARN]  2024-04-01 13:00:06 api-gateway         — Auth service health check failing",
                "[ERROR] 2024-04-01 13:00:08 frontend            — Users unable to log in. Auth service unreachable.",
                "[ERROR] 2024-04-01 13:00:10 monitoring          — SLO breach: auth-service availability < 99.9%",
            ],
            "task_description": (
                "A bad ConfigMap update corrupted the auth-service configuration. "
                "Users cannot log in. Identify the service and roll back the config."
            ),
        },
    ],

    # ── HARD (Task 3) ────────────────────────────────────────────────────────
    "task_3": [
        {
            "name": "cascading_db_failure",
            "difficulty": "hard",
            "root_cause_service": "database",
            "correct_actions": ["restart", "rollback"],
            "logs": [
                "[INFO]  2024-04-01 14:00:01 database            — Running scheduled index optimisation",
                "[WARN]  2024-04-01 14:00:05 database            — Lock contention detected on orders table",
                "[ERROR] 2024-04-01 14:00:10 database            — Deadlock detected. Rolling back 12 transactions.",
                "[ERROR] 2024-04-01 14:00:11 database            — Connection pool exhausted: 500/500 connections used",
                "[ERROR] 2024-04-01 14:00:12 payment-service     — DB connection timeout after 30s",
                "[ERROR] 2024-04-01 14:00:12 payment-service     — Failed to process 47 pending transactions",
                "[ERROR] 2024-04-01 14:00:13 order-service       — Cannot fetch order status: DB unavailable",
                "[ERROR] 2024-04-01 14:00:14 order-service       — 503 returned to 1,240 users",
                "[ERROR] 2024-04-01 14:00:15 api-gateway         — Upstream errors from payment-service and order-service",
                "[ERROR] 2024-04-01 14:00:16 frontend            — Dashboard showing errors for all order/payment features",
                "[WARN]  2024-04-01 14:00:18 monitoring          — PagerDuty alert fired: P1 — Multiple services degraded",
                "[ERROR] 2024-04-01 14:00:20 database            — Replica lag: 45 seconds. Data consistency at risk.",
            ],
            "task_description": (
                "A database deadlock caused connection pool exhaustion which cascaded "
                "into payment-service, order-service, api-gateway, and frontend failures. "
                "Identify the ROOT CAUSE service and apply the correct fix."
            ),
        },
    ],
}


# ---------------------------------------------------------------------------
# REWARD LOGIC
# ---------------------------------------------------------------------------

def compute_reward(
    scenario: dict,
    action: IncidentAction,
    step_number: int,
    max_steps: int,
) -> tuple[float, str, bool]:
    """
    Returns (reward, feedback_message, is_done).

    Scoring rubric:
        1.0  — correct service + correct action on first try
        0.75 — correct service + correct action (not first try)
        0.5  — correct service, wrong action type
        0.25 — wrong service but action type is valid
        0.0  — ignore action or completely wrong
    """
    correct_service = scenario["root_cause_service"]
    correct_actions = scenario["correct_actions"]
    agent_service   = action.target_service.lower().strip()
    agent_action    = action.action_type

    right_service = (agent_service == correct_service)
    right_action  = (agent_action in correct_actions)

    if agent_action == "ignore":
        return 0.0, (
            "The agent chose to ignore the incident. There is a real error "
            f"in '{correct_service}' that must be addressed."
        ), False

    if right_service and right_action:
        speed_bonus = 1.0 if step_number == 1 else 0.75
        return speed_bonus, (
            f"Correct! '{agent_action}' on '{agent_service}' is the right fix. "
            + ("First-try bonus applied!" if speed_bonus == 1.0 else "")
        ), True

    if right_service and not right_action:
        return 0.5, (
            f"You identified the right service ('{correct_service}') but "
            f"'{agent_action}' is not the best action here. "
            f"Try: {correct_actions}."
        ), False

    if not right_service and right_action:
        return 0.25, (
            f"'{agent_action}' is a valid action type but '{agent_service}' "
            f"is not the root cause. Look more carefully at the logs."
        ), False

    # Wrong service AND wrong action
    steps_left = max_steps - step_number
    return 0.0, (
        f"'{agent_action}' on '{agent_service}' did not help. "
        f"Re-read the logs carefully. {steps_left} step(s) remaining."
    ), False


# ---------------------------------------------------------------------------
# MAIN ENVIRONMENT CLASS
# ---------------------------------------------------------------------------

class DevOpsEnvironment:
    """
    The DevOps Incident Response Simulator environment.

    Implements the OpenEnv interface:
        reset()        → LogObservation
        step(action)   → StepResult
        state()        → EpisodeState
    """

    MAX_STEPS = 5

    def __init__(self):
        self._episode_id: Optional[str]   = None
        self._scenario:   Optional[dict]  = None
        self._task_id:    Optional[str]   = None
        self._step_count: int             = 0
        self._cumulative_reward: float    = 0.0
        self._is_done: bool               = False

    # ── reset ────────────────────────────────────────────────────────────────

    def reset(self, task_id: Optional[str] = None) -> LogObservation:
        """
        Start a new episode.
        Optionally pass task_id = "task_1" | "task_2" | "task_3"
        to force a specific difficulty. Defaults to random.
        """
        self._episode_id        = str(uuid.uuid4())
        self._step_count        = 0
        self._cumulative_reward = 0.0
        self._is_done           = False

        # Pick task
        if task_id and task_id in SCENARIOS:
            self._task_id = task_id
        else:
            self._task_id = random.choice(list(SCENARIOS.keys()))

        # Pick scenario within that task
        self._scenario = random.choice(SCENARIOS[self._task_id])

        return LogObservation(
            logs=self._scenario["logs"],
            pipeline_status="failed",
            active_services=self._get_active_services(),
            error_detected=True,
            error_service=None,   # Agent must figure this out
            step_number=0,
            task_description=self._scenario["task_description"],
        )

    # ── step ─────────────────────────────────────────────────────────────────

    def step(self, action: IncidentAction) -> StepResult:
        """
        Agent submits an action. Environment evaluates it and returns
        a new observation with reward and feedback.
        """
        if self._is_done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")
        if self._scenario is None:
            raise RuntimeError("No active episode. Call reset() first.")

        self._step_count += 1

        reward, feedback, success = compute_reward(
            scenario=self._scenario,
            action=action,
            step_number=self._step_count,
            max_steps=self.MAX_STEPS,
        )

        self._cumulative_reward += reward

        # Episode ends on success OR hitting max steps
        if success or self._step_count >= self.MAX_STEPS:
            self._is_done = True

        # Build next observation
        if success:
            new_status = "healthy"
            new_logs   = self._scenario["logs"] + [
                f"[INFO]  REMEDIATION — '{action.action_type}' applied to "
                f"'{action.target_service}'. Pipeline restored successfully.",
            ]
        else:
            new_status = "degraded" if self._step_count < self.MAX_STEPS else "failed"
            new_logs   = self._scenario["logs"] + [
                f"[WARN]  Step {self._step_count} — Action '{action.action_type}' "
                f"on '{action.target_service}' did not resolve the incident.",
            ]

        observation = LogObservation(
            logs=new_logs,
            pipeline_status=new_status,
            active_services=self._get_active_services(),
            error_detected=not success,
            error_service=self._scenario["root_cause_service"] if success else None,
            step_number=self._step_count,
            task_description=self._scenario["task_description"],
        )

        return StepResult(
            observation=observation,
            reward=reward,
            done=self._is_done,
            feedback=feedback,
        )

    # ── state ────────────────────────────────────────────────────────────────

    def state(self) -> EpisodeState:
        """
        Returns metadata about the current episode.
        """
        if self._episode_id is None:
            raise RuntimeError("No active episode. Call reset() first.")

        return EpisodeState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            max_steps=self.MAX_STEPS,
            task_id=self._task_id,
            task_difficulty=self._scenario["difficulty"],
            cumulative_reward=self._cumulative_reward,
            is_done=self._is_done,
            scenario_name=self._scenario["name"],
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    def _get_active_services(self) -> list[str]:
        """
        Extract unique service names from the scenario logs.

        Log format: [LEVEL]  date  time  service-name  —  message
        Index:         0      1     2         3         4    5+
        """
        services = set()
        for line in self._scenario["logs"]:
            parts = line.split()
            # Need at least: [LEVEL] date time service-name
            if len(parts) >= 4:
                services.add(parts[3])
        # Remove any stray tokens that are not service names
        non_services = {"—", "REMEDIATION", "Step"}
        return sorted(services - non_services)