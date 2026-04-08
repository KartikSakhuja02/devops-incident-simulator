"""
models.py — Data models for the DevOps Incident Response Simulator.

Defines the Action, Observation, and State types that the AI agent
sends and receives when interacting with the environment.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# ACTION — what the AI agent can do
# ---------------------------------------------------------------------------

class IncidentAction(BaseModel):
    """
    The action the agent takes in response to an incident.

    action_type choices:
        - rollback   : Revert the last deployment to the previous stable version
        - patch      : Apply a code fix to the running service
        - redeploy   : Re-trigger the pipeline with current code
        - restart    : Restart a specific service without code change
        - ignore     : Do nothing (agent thinks there is no real issue)
    """
    action_type: Literal["rollback", "patch", "redeploy", "restart", "ignore"] = Field(
        description="The type of remediation action to take."
    )
    target_service: str = Field(
        description="The name of the service to act on, e.g. 'payment-service'."
    )
    patch_content: Optional[str] = Field(
        default=None,
        description="For 'patch' actions: a short description or snippet of the fix."
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Agent's explanation for why it chose this action."
    )


# ---------------------------------------------------------------------------
# OBSERVATION — what the agent sees after each step
# ---------------------------------------------------------------------------

class LogObservation(BaseModel):
    """
    What the agent observes about the current state of the pipeline.
    Returned by reset() and step().
    """
    logs: list[str] = Field(
        description="List of log lines from the pipeline run."
    )
    pipeline_status: Literal["healthy", "degraded", "failed"] = Field(
        description="Overall health of the pipeline."
    )
    active_services: list[str] = Field(
        description="Names of all services currently running in the pipeline."
    )
    error_detected: bool = Field(
        description="Whether a critical error was detected in the logs."
    )
    error_service: Optional[str] = Field(
        default=None,
        description="Which service is the root cause of the error, if any."
    )
    step_number: int = Field(
        description="Current step number within the episode."
    )
    task_description: str = Field(
        description="Human-readable description of what needs to be fixed."
    )


# ---------------------------------------------------------------------------
# STEP RESULT — wraps observation + reward signal
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """
    The full result returned after the agent takes an action.
    Combines the new observation with a reward and done signal.
    """
    observation: LogObservation
    reward: float = Field(
        ge=0.0, le=1.0,
        description="Reward for the action taken. 0.0 = wrong, 1.0 = perfect."
    )
    done: bool = Field(
        description="Whether the episode has ended (success or max steps reached)."
    )
    feedback: str = Field(
        description="Human-readable feedback on the action taken."
    )


# ---------------------------------------------------------------------------
# EPISODE STATE — metadata tracked by the environment server
# ---------------------------------------------------------------------------

class EpisodeState(BaseModel):
    """
    Internal state the server tracks per episode.
    Returned by state().
    """
    episode_id: str
    step_count: int
    max_steps: int
    task_id: str                     # "task_1", "task_2", or "task_3"
    task_difficulty: Literal["easy", "medium", "hard"]
    cumulative_reward: float
    is_done: bool
    scenario_name: str               # e.g. "null_pointer_in_payment_service"