"""
inference.py — LLM Agent loop for the DevOps Incident Response Simulator.

This is the file the hackathon validators run directly.
It must:
    - Be named inference.py and live in the project ROOT
    - Use the OpenAI client for all LLM calls
    - Read API_BASE_URL, MODEL_NAME, HF_TOKEN from environment variables
    - Complete in under 20 minutes
    - Run on 2 vCPU / 8GB RAM
    - Print scores for all 3 tasks at the end

Usage:
    # Make sure the environment server is running first:
    #   python server/app.py
    #
    # Then in a second terminal:
    #   python inference.py

Environment variables required:
    API_BASE_URL  — e.g. https://router.huggingface.co/v1
    MODEL_NAME    — e.g. meta-llama/Llama-3.1-8B-Instruct
    HF_TOKEN      — your Hugging Face API token
"""

import os
import sys
import json
import time
from datetime import datetime, timezone

from openai import OpenAI
from dotenv import load_dotenv

# Load .env from this project root (where inference.py lives), then fallback to default lookup.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"), override=False)
load_dotenv(override=False)


def normalize_api_base_url(url: str) -> str:
    """Normalize legacy HF inference endpoint to the supported router endpoint."""
    normalized = url.strip()
    normalized = normalized.rstrip("/")

    # HF retired api-inference.huggingface.co; router.huggingface.co is the supported endpoint.
    normalized = normalized.replace("https://api-inference.huggingface.co", "https://router.huggingface.co")
    normalized = normalized.replace("http://api-inference.huggingface.co", "https://router.huggingface.co")

    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"

    return normalized

# ---------------------------------------------------------------------------
# Config — read from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN")
SERVER_URL   = os.environ.get("ENV_SERVER_URL", "http://localhost:8000")

if not HF_TOKEN:
    print("[ERROR] Missing required environment variable: HF_TOKEN")
    print("        Continuing in fail-safe mode (LLM calls will fail gracefully).")
    HF_TOKEN = ""

original_api_base_url = API_BASE_URL
API_BASE_URL = normalize_api_base_url(API_BASE_URL)
if API_BASE_URL != original_api_base_url.rstrip("/"):
    print(f"[INFO] API_BASE_URL updated to supported endpoint: {API_BASE_URL}")

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

# LLM client (OpenAI-compatible, pointing to HF inference)
llm = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# Environment client (talks to our FastAPI server)
sys.path.insert(0, os.path.dirname(__file__))
from client import DevOpsEnvClient
from models import IncidentAction


# ---------------------------------------------------------------------------
# System prompt — tells the LLM how to behave as a DevOps agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert DevOps Site Reliability Engineer (SRE).
You will be given pipeline logs from a production system that has an incident.
Your job is to:
1. Read the logs carefully
2. Identify which service is the ROOT CAUSE of the problem
3. Choose the correct remediation action

You must respond with ONLY a valid JSON object in this exact format:
{
    "action_type": "<one of: rollback, patch, redeploy, restart, ignore>",
    "target_service": "<the name of the root cause service>",
    "reasoning": "<one sentence explaining why you chose this>"
}

Action type guide:
- rollback  : The service had a bad deployment or config change. Revert to previous version.
- patch     : The service has a code bug that can be fixed with a small change.
- redeploy  : The service needs to be redeployed with the current code (e.g. after config fix).
- restart   : The service is in a bad state (OOM, deadlock) and needs a clean restart.
- ignore    : There is no real incident (use this only if logs show everything is healthy).

Rules:
- Target the ROOT CAUSE service, not the services that are affected downstream.
- If multiple services are failing, look for the one that failed FIRST in the timestamps.
- Respond with ONLY the JSON. No explanation, no markdown, no code blocks.
"""


def _ts() -> str:
    """UTC timestamp for structured logs."""
    return datetime.now(timezone.utc).isoformat()


def log_start(task_id: str, model_name: str, api_base_url: str, server_url: str) -> None:
    print(f"[START] task_id={task_id}")


def log_step(
    task_id: str,
    step: int,
    action_type: str,
    target_service: str,
    reward: float,
    done: bool,
    feedback: str,
) -> None:
    compact_feedback = " ".join((feedback or "").split())
    print(
        f"[STEP] task_id={task_id} action_type={action_type} "
        f"target_service={target_service} score={reward:.2f} done={str(done).lower()} "
        f"feedback=\"{compact_feedback}\""
    )


def log_end(task_id: str, final_reward: float, steps_taken: int, success: bool) -> None:
    print(
        f"[END] task_id={task_id} score={final_reward:.2f} "
        f"status={('pass' if success else 'fail')}"
    )


# ---------------------------------------------------------------------------
# Agent logic
# ---------------------------------------------------------------------------

def build_user_prompt(obs) -> str:
    """Build the prompt sent to the LLM for each step."""
    logs_text = "\n".join(obs.logs)
    return f"""TASK: {obs.task_description}

PIPELINE STATUS: {obs.pipeline_status}
ACTIVE SERVICES: {", ".join(obs.active_services)}
STEP: {obs.step_number}

LOGS:
{logs_text}

Analyze the logs and respond with the JSON action."""


def parse_llm_response(response_text: str) -> dict:
    """
    Parse the LLM's JSON response.
    Handles cases where the model wraps the JSON in markdown code blocks.
    """
    text = response_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        text = "\n".join(lines[1:-1]).strip()

    return json.loads(text)


def run_agent_on_task(client: DevOpsEnvClient, task_id: str) -> dict:
    """
    Run the LLM agent on a single task.
    Returns a result dict with task_id, final_reward, steps_taken, success.
    """
    print(f"\n{'─' * 55}")
    print(f"  Task: {task_id.upper()}")
    print(f"{'─' * 55}")
    log_start(task_id, MODEL_NAME, API_BASE_URL, SERVER_URL)

    # Reset environment for this task
    try:
        obs = client.reset(task_id=task_id)
    except Exception as e:
        msg = f"task_reset_failed: {e}"
        print(f"  [ERROR] {msg}")
        log_step(
            task_id=task_id,
            step=0,
            action_type="error",
            target_service="unknown",
            reward=0.01,
            done=True,
            feedback=msg,
        )
        log_end(task_id, 0.01, 0, False)
        return {
            "task_id": task_id,
            "final_reward": 0.01,
            "steps_taken": 0,
            "success": False,
        }
    print(f"  Scenario  : {obs.task_description[:65]}...")
    print(f"  Services  : {obs.active_services}")
    print()

    final_reward = 0.01
    steps_taken  = 0
    success      = False

    # Agent loop — up to 5 steps per task
    for step in range(1, 6):
        print(f"  [Step {step}] Asking LLM...")

        # Build prompt and call LLM
        user_prompt = build_user_prompt(obs)

        try:
            response = llm.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=200,
                temperature=0.1,   # Low temp for deterministic, precise answers
            )
            response_text = response.choices[0].message.content
            print(f"  [Step {step}] LLM raw response: {response_text[:120]}")

        except Exception as e:
            print(f"  [Step {step}] LLM call failed: {e}")
            log_step(
                task_id=task_id,
                step=step,
                action_type="error",
                target_service="unknown",
                reward=0.01,
                done=True,
                feedback=f"llm_call_failed: {e}",
            )
            break

        # Parse LLM response into an action
        try:
            parsed = parse_llm_response(response_text)
            action = IncidentAction(
                action_type    = parsed.get("action_type", "ignore"),
                target_service = parsed.get("target_service", "unknown"),
                reasoning      = parsed.get("reasoning", ""),
            )
        except Exception as e:
            print(f"  [Step {step}] Failed to parse LLM response: {e}")
            print(f"               Raw text was: {response_text}")
            # Default safe action
            action = IncidentAction(
                action_type="ignore",
                target_service="unknown",
                reasoning="Failed to parse LLM response"
            )

        print(f"  [Step {step}] Action : {action.action_type} on '{action.target_service}'")
        print(f"  [Step {step}] Reason : {action.reasoning}")

        # Submit action to environment
        try:
            result = client.step(action)
        except Exception as e:
            msg = f"task_step_failed: {e}"
            print(f"  [Step {step}] ERROR: {msg}")
            log_step(
                task_id=task_id,
                step=step,
                action_type=action.action_type,
                target_service=action.target_service,
                reward=0.01,
                done=True,
                feedback=msg,
            )
            break
        steps_taken   = step
        final_reward  = result.reward
        print(f"  [Step {step}] Reward : {result.reward}")
        print(f"  [Step {step}] Feedback: {result.feedback}")
        log_step(
            task_id=task_id,
            step=step,
            action_type=action.action_type,
            target_service=action.target_service,
            reward=result.reward,
            done=result.done,
            feedback=result.feedback,
        )

        if result.done:
            success = (result.reward > 0.5)
            print(f"  Episode done. Success: {success}")
            obs = result.observation
            break

        # Update observation for next step
        obs = result.observation

        # Small delay to avoid rate limiting on HF inference API
        time.sleep(1)

    log_end(task_id, final_reward, steps_taken, success)

    return {
        "task_id":      task_id,
        "final_reward": final_reward,
        "steps_taken":  steps_taken,
        "success":      success,
    }


# ---------------------------------------------------------------------------
# Main — run all 3 tasks and print final scores
# ---------------------------------------------------------------------------

def main():
    print("DevOps Incident Response Simulator - Inference")

    # Connect to environment server
    client = DevOpsEnvClient(base_url=SERVER_URL)

    # Verify server is up before starting
    try:
        health = client.health()
        print(f"\n  Server health: {health['status']}")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        print("  Continuing with fail-safe scoring output for all tasks.")
        results = [
            {"task_id": "task_1", "final_reward": 0.01, "steps_taken": 0, "success": False},
            {"task_id": "task_2", "final_reward": 0.01, "steps_taken": 0, "success": False},
            {"task_id": "task_3", "final_reward": 0.01, "steps_taken": 0, "success": False},
        ]
        _print_final_scores(results, elapsed=0.0)
        return

    # Run all 3 tasks
    results = []
    start_time = time.time()

    for task_id in ["task_1", "task_2", "task_3"]:
        try:
            result = run_agent_on_task(client, task_id)
        except Exception as e:
            print(f"\n[ERROR] Unexpected task failure for {task_id}: {e}")
            log_step(
                task_id=task_id,
                step=0,
                action_type="error",
                target_service="unknown",
                reward=0.01,
                done=True,
                feedback=f"unexpected_task_exception: {e}",
            )
            log_end(task_id, 0.01, 0, False)
            result = {"task_id": task_id, "final_reward": 0.01, "steps_taken": 0, "success": False}
        results.append(result)

    elapsed = time.time() - start_time

    _print_final_scores(results, elapsed)


def _print_final_scores(results: list[dict], elapsed: float) -> None:
    # ---------------------------------------------------------------------------
    # Final scores report — validators check this output
    # ---------------------------------------------------------------------------
    print("\nFINAL SCORES")

    total_reward = 0.0
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        print(
            f"  [{status}] {r['task_id']:8s} | "
            f"score={r['final_reward']:.2f} | "
            f"success={r['success']}"
        )
        total_reward += r["final_reward"]

    avg_reward = total_reward / len(results)
    print(f"  Avg score    : {avg_reward:.2f}")

    # Keep exit code 0 so validators receive scores even when tasks fail.
    return


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] fatal_exception: {e}")
        # Last-resort fail-safe output to avoid hard crash in validators.
        fallback = [
            {"task_id": "task_1", "final_reward": 0.01, "steps_taken": 0, "success": False},
            {"task_id": "task_2", "final_reward": 0.01, "steps_taken": 0, "success": False},
            {"task_id": "task_3", "final_reward": 0.01, "steps_taken": 0, "success": False},
        ]
        _print_final_scores(fallback, elapsed=0.0)
        sys.exit(0)