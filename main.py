"""
Main orchestrator — Planner + Executor coordination loop.

Entry point for multi-step tasks. For simple single-step tasks,
gui_agent.py can still be run directly via its own main().
"""

import logging
import os
import time

from dotenv import load_dotenv

from planner import Planner, ExecutionPlan
from gui_agent import GUIAgent

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_task(
    user_instruction: str,
    planner: Planner,
    agent: GUIAgent,
) -> bool:
    """
    Full task execution:
      1. Planner generates a one-shot plan (with screenshot for visual grounding)
      2. Executor runs each step independently (clean state each time)
      3. After each step success: independent verify_completion() confirms it
      4. On step failure → replan (max 3 times total)
    Returns True if all steps completed successfully.
    """
    # Feature 1: Give Planner a current screenshot for visual grounding
    try:
        init_img  = agent.screen_capture.capture_screen()
        init_b64  = agent.screen_capture.image_to_base64(init_img)
    except Exception:
        init_b64 = None

    plan = planner.create_plan(user_instruction, screenshot_b64=init_b64)

    prior_state = ""   # Track screen state from previous step

    while not plan.is_complete:
        step = plan.current_step
        logger.info(f"\n{'='*55}")
        logger.info(f"  Step {plan.progress}: {step}")
        logger.info(f"{'='*55}")

        # Each step gets a clean slate — no history bleed between steps
        agent.reset_state(step)

        # Inject global anchors to prevent goal drift inside the step
        agent.set_global_context(
            global_goal=plan.goal,
            current_step=f"Step {plan.progress}: {step}",
            success_criteria=plan.success_criteria,
            prior_state=prior_state,
        )

        success = agent.run(max_duration_seconds=600)

        if success:
            # Feature 2: verify against THIS step's criteria (not overall task)
            step_criteria = plan.current_step_criteria or plan.success_criteria
            verified, evidence = agent.verify_completion(step_criteria)
            if not verified:
                logger.warning(f"[Verify] ❌ Step {plan.progress} not confirmed: {evidence}")
                logger.info(f"[Verify] Retrying step once...")
                agent.reset_state(step)
                agent.set_global_context(
                    global_goal=plan.goal,
                    current_step=f"Step {plan.progress}: {step}",
                    success_criteria=plan.success_criteria,
                    prior_state=prior_state,
                )
                success = agent.run(max_duration_seconds=600)
                if success:
                    verified, evidence = agent.verify_completion(step_criteria)
                    if not verified:
                        logger.warning(f"[Verify] ❌ Retry failed: {evidence}")
                        success = False

        if success:
            logger.info(f"✅ Step {plan.progress} completed")
            # Capture last observation for next step's context
            if agent.action_history:
                last_rec = agent.action_history[-1]
                prior_state = last_rec.observation or last_rec.thought or ""
            plan.advance()
        else:
            logger.warning(f"❌ Step {plan.progress} failed: {step}")
            try:
                plan = planner.replan(
                    plan,
                    failed_step=step,
                    failure_reason="Executor exhausted max_iterations",
                )
                # Fix 2: Inject failed_steps into prior_state for LLM context
                if plan.failed_steps:
                    prior_state += f" [Previous attempts failed on: {', '.join(plan.failed_steps[-2:])}]"
            except RuntimeError as e:
                logger.error(f"Task failed permanently: {e}")
                return False

    logger.info(f"\n{'='*55}")
    logger.info(f"🎉 Task complete: {plan.goal}")
    logger.info(f"   Replanning used: {plan.replan_count} time(s)")
    logger.info(f"{'='*55}")
    return True


def main():
    from cli import (
        console, show_welcome_panel, show_safety_warning, show_config_panel,
        log_info, log_success, log_warning, log_error,
        get_user_instruction, show_action_summary, Command, TaskProgress,
    )
    from config import load_config, validate_api_key, get_api_key_status

    show_welcome_panel()
    show_safety_warning()

    config = load_config()
    show_config_panel({
        "model":          config.model,
        "base_url":       config.base_url,
        "max_iterations": config.max_iterations,
    })

    if not validate_api_key(config):
        log_warning(f"API Key not configured: {get_api_key_status(config)}")
        log_info("Set DASHSCOPE_API_KEY in .env or use --api-key flag")
        console.print()

    os.makedirs("logs", exist_ok=True)
    session_ts = time.strftime("%Y%m%d_%H%M%S")
    log_path   = os.path.join("logs", f"session_{session_ts}.jsonl")

    planner = Planner()
    agent   = GUIAgent(config=config, history_log_path=log_path)  # Fix 5
    logger.info(f"Session history: {log_path}")

    running     = True
    all_history = []

    while running:
        instruction = get_user_instruction("Enter your instruction")
        if instruction is None:
            running = False
            continue

        if Command.is_command(instruction):
            cmd = Command.get_command_type(instruction)
            if cmd == "quit":
                running = False
                console.print("[bold blue]Goodbye![/bold blue]")
            elif cmd == "help":
                from cli import show_help
                show_help()
            elif cmd == "clear":
                console.clear()
            elif cmd == "config":
                show_config_panel({
                    "model":          config.model,
                    "base_url":       config.base_url,
                    "max_iterations": config.max_iterations,
                })
            continue

        console.print()
        log_info(f"Task: {instruction}")
        console.print()

        try:
            with TaskProgress("Executing task..."):
                success = run_task(instruction, planner, agent)

            console.print()
            if success:
                log_success("Task completed!")
            else:
                log_warning("Task failed — check logs for details")

            for rec in agent.action_history:
                all_history.append({
                    "action":     rec.action_type,
                    "details":    rec.thought,
                    "status":     "success" if rec.screen_changed else "failed",
                    "status_tag": rec.status_tag,
                })

        except KeyboardInterrupt:
            console.print()
            log_warning("Task interrupted by user")
        except Exception as e:
            log_error(f"Execution error: {e}")

    console.print()
    if all_history:
        successful = sum(1 for a in all_history if a["status"] == "success")
        failed     = sum(1 for a in all_history if a["status"] == "failed")
        show_action_summary(len(all_history), successful, failed)

    console.print("[bold blue]Thank you for using GUI Agent![/bold blue]")


if __name__ == "__main__":
    main()
