"""
Planner — Stateless task decomposer.

Converts a high-level user goal into an ordered list of atomic steps
for the Executor (gui_agent.py) to execute one at a time.

Design principle: Planner is a one-shot "compiler", NOT a stateful agent.
It generates the full plan upfront; replan() is only called when Executor
exhausts all retries on a single step.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


# ── Prompts ──────────────────────────────────────────────────────────────────

PLANNER_SYSTEM_PROMPT = """You are a task planning expert for a GUI automation agent.
Your job is to decompose a high-level user goal into a precise, ordered list of atomic steps.

Rules:
- Each step must describe the desired END STATE, not just the action.
  Start each step with "Ensure..." so the executor checks the current screen
  before taking unnecessary actions.
  Good: "Ensure the Men > Bags & Backpacks page is displayed (navigate there if not already visible)"
  Bad:  "Click on Men menu"
- Merge steps that happen in the same browser/app context without page reload.
  Good: "Ensure Chrome is open and navigated to linkedin.com"
  Bad:  Step 1 "Open Chrome", Step 2 "Click address bar", Step 3 "Type URL"
- Steps must be concrete enough that a visual agent can verify completion
  by looking at the screen
- Maximum 8 steps; merge related actions aggressively
- Each step MUST include its own success_criteria (visible, unambiguous screen state)
- If a screenshot is provided, use it to understand the CURRENT screen state
  and avoid steps that are already complete

Return pure JSON only — no Markdown, no extra text:
{
  "goal": "restate the user goal clearly",
  "steps": [
    {"description": "step 1 text", "success_criteria": "what the screen shows when step 1 is done"},
    {"description": "step 2 text", "success_criteria": "what the screen shows when step 2 is done"}
  ],
  "success_criteria": "what the screen must show to confirm FULL task completion"
}"""

REPLAN_SYSTEM_PROMPT = """You are a task replanning expert.
The executor failed on a specific step. Analyze the failure and produce a revised plan.

Consider:
- Can the failed step be split into smaller, more atomic steps?
- Is there an alternative path (keyboard shortcut vs mouse click)?
- Do earlier steps need to be repeated first?

Return pure JSON in the same format as the original plan, with per-step success_criteria:
{
  "goal": "...",
  "steps": [
    {"description": "step text", "success_criteria": "visible screen state for this step"},
    ...
  ],
  "success_criteria": "overall task completion criteria"
}"""


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class ExecutionPlan:
    goal: str
    steps: list[dict]       # [{"description": str, "success_criteria": str}, ...]
    success_criteria: str   # overall / final task criteria
    current_step_index: int = 0
    replan_count: int = 0
    failed_steps: list[str] = field(default_factory=list)

    @property
    def current_step(self) -> Optional[str]:
        """Description text of the current step."""
        if self.current_step_index < len(self.steps):
            s = self.steps[self.current_step_index]
            return s["description"] if isinstance(s, dict) else s
        return None

    @property
    def current_step_criteria(self) -> Optional[str]:
        """Per-step success criteria, falls back to overall criteria."""
        if self.current_step_index < len(self.steps):
            s = self.steps[self.current_step_index]
            if isinstance(s, dict):
                return s.get("success_criteria") or self.success_criteria
        return self.success_criteria

    @property
    def is_complete(self) -> bool:
        return self.current_step_index >= len(self.steps)

    @property
    def progress(self) -> str:
        return f"{self.current_step_index + 1}/{len(self.steps)}"

    def advance(self):
        self.current_step_index += 1


# ── Planner class ─────────────────────────────────────────────────────────────

class Planner:
    MAX_REPLAN_ATTEMPTS = 3

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        api_key    = api_key    or os.getenv("DASHSCOPE_API_KEY")
        base_url   = base_url   or os.getenv("PLANNER_BASE_URL") \
                               or os.getenv("BASE_URL",
                                            "https://dashscope.aliyuncs.com/compatible-mode/v1")
        model_name = model_name or os.getenv("PLANNER_MODEL_NAME") \
                               or os.getenv("MODEL_NAME", "qwen-plus")

        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY not set")

        self.client     = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        logger.info(f"Planner initialized — model: {model_name}")

    def create_plan(self, user_task: str, screenshot_b64: Optional[str] = None) -> ExecutionPlan:
        """Generate a step-by-step plan.

        Args:
            user_task: Natural language task description.
            screenshot_b64: Optional base64 data-URI of current screen
                            (e.g. "data:image/jpeg;base64,...").
                            When provided the Planner gains visual grounding
                            and can skip steps already complete on screen.
        """
        logger.info(f"[Planner] Creating plan for: {user_task}")
        if screenshot_b64:
            logger.info("[Planner] Screenshot provided — visual grounding enabled")

        # Build user message — multimodal when screenshot available
        if screenshot_b64:
            user_content = [
                {"type": "text",
                 "text": f"Task: {user_task}\n\nThis is the CURRENT screen state. "
                         "Use it to understand what is already open/visible."},
                {"type": "image_url",
                 "image_url": {"url": screenshot_b64}},
            ]
        else:
            user_content = f"Task: {user_task}"

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=800,
        )
        data = json.loads(response.choices[0].message.content)

        # Normalise steps: support both old (str) and new (dict) formats
        raw_steps = data["steps"]
        steps = []
        for s in raw_steps:
            if isinstance(s, str):
                steps.append({"description": s, "success_criteria": data.get("success_criteria", "")})
            else:
                steps.append(s)

        plan = ExecutionPlan(
            goal=data["goal"],
            steps=steps,
            success_criteria=data["success_criteria"],
        )
        logger.info(f"[Planner] {len(plan.steps)} steps generated:")
        for i, step in enumerate(plan.steps):
            logger.info(f"  {i+1}. {step['description']}")
            logger.info(f"     → {step.get('success_criteria', '')}")
        logger.info(f"  ✅ Overall: {plan.success_criteria}")
        return plan

    def replan(
        self,
        original_plan: ExecutionPlan,
        failed_step: str,
        failure_reason: str,
    ) -> ExecutionPlan:
        if original_plan.replan_count >= self.MAX_REPLAN_ATTEMPTS:
            raise RuntimeError(
                f"Replanning limit ({self.MAX_REPLAN_ATTEMPTS}) reached. "
                f"Cannot complete: {original_plan.goal}"
            )

        logger.warning(
            f"[Planner] Replanning attempt {original_plan.replan_count + 1} "
            f"— failed: '{failed_step}' | reason: {failure_reason}"
        )

        # Fix 4: Normalize steps to dict format for backward compatibility
        # This handles edge cases where old plans (list[str]) are deserialized
        def normalize_step(s):
            if isinstance(s, str):
                return {"description": s, "success_criteria": original_plan.success_criteria}
            return s
        
        completed = [normalize_step(s) for s in original_plan.steps[:original_plan.current_step_index]]
        remaining = [normalize_step(s) for s in original_plan.steps[original_plan.current_step_index:]]

        context = f"""
Original goal: {original_plan.goal}
Completed steps: {json.dumps(completed)}
Failed step: {failed_step}
Failure reason: {failure_reason}
Remaining steps (not yet attempted): {json.dumps(remaining)}
Success criteria: {original_plan.success_criteria}

Generate a revised plan starting from the failed step.
"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": REPLAN_SYSTEM_PROMPT},
                {"role": "user",   "content": context},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=800,
        )
        data     = json.loads(response.choices[0].message.content)

        # Normalise steps format
        raw_steps = data["steps"]
        steps = []
        for s in raw_steps:
            if isinstance(s, str):
                steps.append({"description": s, "success_criteria": data.get("success_criteria", "")})
            else:
                steps.append(s)

        new_plan = ExecutionPlan(
            goal=data["goal"],
            steps=steps,
            success_criteria=data["success_criteria"],
            replan_count=original_plan.replan_count + 1,
            failed_steps=original_plan.failed_steps + [failed_step],
        )
        logger.info(f"[Planner] Revised plan: {len(new_plan.steps)} steps")
        return new_plan
