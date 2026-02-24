"""
GUI Agent - Desktop automation agent based on Qwen-VL multimodal large language model

Core architecture: ReAct (Reasoning and Acting) Loop

Key design decisions:
- pyautogui for mouse/keyboard (proven, precise, handles DPI/easing properly)
- GDI BitBlt for screen capture (fastest, no deadlock on Windows)
- Pure PIL for SSIM (no cv2/skimage dependency)
- DPI awareness declared BEFORE any import that touches Win32 display
"""

import os
import platform
import sys

# ============================================================================
# Step 1: DPI Awareness — MUST run before pyautogui, PIL, ctypes display calls
# ============================================================================
if platform.system() == "Windows":
    import ctypes
    import ctypes.wintypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)   # Per-Monitor DPI (Win 8.1+)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()    # System DPI (Vista+)
        except Exception:
            pass

# ============================================================================
# Step 2: Global imports — pyautogui MUST be imported here at module level,
# BEFORE any threads start (Rich progress, etc.).  Lazy import causes deadlock
# in Git Bash / MSYS2 because pyautogui's Win32 display init blocks the GIL.
# ============================================================================
import pyautogui
import pyperclip

import base64
import hashlib
import io
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageDraw, ImageGrab

# Configure pyautogui safety
pyautogui.FAILSAFE = True       # Move mouse to corner to abort
pyautogui.PAUSE = 0.05          # Small global pause between pyautogui calls

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

class ActionType(Enum):
    CLICK        = "CLICK"
    DOUBLE_CLICK = "DOUBLE_CLICK"
    RIGHT_CLICK  = "RIGHT_CLICK"
    HOVER        = "HOVER"
    DRAG         = "DRAG"
    TYPE         = "TYPE"
    SCROLL       = "SCROLL"
    HOTKEY       = "HOTKEY"
    WAIT         = "WAIT"
    DONE         = "DONE"


@dataclass
class ActionParams:
    x:            Optional[float] = None
    y:            Optional[float] = None
    end_x:        Optional[float] = None
    end_y:        Optional[float] = None
    text:         Optional[str]   = None
    scroll_amount: Optional[int]  = None
    keys:         Optional[str]   = None
    wait_seconds: Optional[float] = None


@dataclass
class AgentResponse:
    thought:      str
    action_type:  ActionType
    action_params: ActionParams
    observation:  str   = ""
    confidence:   float = 0.5
    fallback:     str   = ""


@dataclass
class ActionRecord:
    iteration:       int
    action_type:     str
    action_params:   dict          = field(default_factory=dict)
    thought:         str           = ""
    observation:     str           = ""
    coordinates:     Optional[tuple] = None
    screen_changed:  bool          = False
    similarity_score: float        = 1.0
    wait_elapsed:    float         = 0.0
    error:           Optional[str] = None
    timestamp:       float         = field(default_factory=time.time)
    status_tag:      str           = ""


# ============================================================================
# Screen Capture  (GDI BitBlt — fastest, no deadlock risk)
# ============================================================================

class ScreenCapture:

    def __init__(self):
        self.last_click_pos: Optional[tuple[float, float]] = None

    @staticmethod
    def _gdi_capture() -> Image.Image:
        """Capture full virtual desktop via Win32 GDI BitBlt.

        Preferred over PIL.ImageGrab / mss because:
        - No import-time display-server connection (no deadlock)
        - Handles multi-monitor negative-offset layouts correctly
        - Returns physical pixels (consistent with DPI-aware coordinates)
        """
        if platform.system() != "Windows":
            return ImageGrab.grab(all_screens=True)

        SM_XVIRTUALSCREEN  = 76
        SM_YVIRTUALSCREEN  = 77
        SM_CXVIRTUALSCREEN = 78
        SM_CYVIRTUALSCREEN = 79
        SRCCOPY = 0x00CC0020

        user32 = ctypes.windll.user32
        gdi32  = ctypes.windll.gdi32

        vx = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
        vy = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
        vw = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
        vh = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)

        hdc     = user32.GetDC(0)
        hdc_mem = gdi32.CreateCompatibleDC(hdc)
        hbmp    = gdi32.CreateCompatibleBitmap(hdc, vw, vh)
        gdi32.SelectObject(hdc_mem, hbmp)
        gdi32.BitBlt(hdc_mem, 0, 0, vw, vh, hdc, vx, vy, SRCCOPY)

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize",          ctypes.c_uint32),
                ("biWidth",         ctypes.c_int32),
                ("biHeight",        ctypes.c_int32),
                ("biPlanes",        ctypes.c_uint16),
                ("biBitCount",      ctypes.c_uint16),
                ("biCompression",   ctypes.c_uint32),
                ("biSizeImage",     ctypes.c_uint32),
                ("biXPelsPerMeter", ctypes.c_int32),
                ("biYPelsPerMeter", ctypes.c_int32),
                ("biClrUsed",       ctypes.c_uint32),
                ("biClrImportant",  ctypes.c_uint32),
            ]

        bmi = BITMAPINFOHEADER()
        bmi.biSize        = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.biWidth       = vw
        bmi.biHeight      = -vh   # negative = top-down
        bmi.biPlanes      = 1
        bmi.biBitCount    = 32
        bmi.biCompression = 0

        buf = (ctypes.c_char * (vw * vh * 4))()
        gdi32.GetDIBits(hdc_mem, hbmp, 0, vh, buf, ctypes.byref(bmi), 0)

        gdi32.DeleteObject(hbmp)
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(0, hdc)

        return Image.frombytes("RGBA", (vw, vh), bytes(buf)).convert("RGB")

    def get_screen_resolution(self) -> tuple[int, int]:
        """Physical pixel resolution of the full virtual desktop."""
        if platform.system() == "Windows":
            w = ctypes.windll.user32.GetSystemMetrics(78)   # SM_CXVIRTUALSCREEN
            h = ctypes.windll.user32.GetSystemMetrics(79)   # SM_CYVIRTUALSCREEN
            return (w, h)
        img = ImageGrab.grab(all_screens=True)
        return img.size

    def capture_screen(self) -> Image.Image:
        return self._gdi_capture()

    def image_to_base64(self, image: Image.Image, fmt: str = "JPEG") -> str:
        buf = io.BytesIO()
        image.save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/{fmt.lower()};base64,{b64}"

    def draw_click_indicator_at_pixels(self, image: Image.Image, px: int, py: int) -> None:
        draw = ImageDraw.Draw(image)
        r = 10
        draw.ellipse([px - r, py - r, px + r, py + r], outline="red", width=3)
        c = 15
        draw.line([px - c, py, px + c, py], fill="red", width=2)
        draw.line([px, py - c, px, py + c], fill="red", width=2)

    def compress_image(self, image: Image.Image, max_width: int = 1280) -> Image.Image:
        w, h = image.size
        if w <= max_width:
            return image
        ratio = max_width / w
        return image.resize((max_width, int(h * ratio)), Image.Resampling.LANCZOS)

    def draw_grid_overlay(self, image: Image.Image) -> None:
        w, h = image.size
        draw = ImageDraw.Draw(image, "RGBA")
        gc = (0, 255, 0, 60)
        tc = (255, 0, 0, 255)
        for i in range(1, 10):
            x = int(w * i / 10)
            draw.line([(x, 0), (x, h)], fill=gc, width=2)
            draw.text((x + 4, 8),    f"X:{i/10:.1f}", fill=tc)
            draw.text((x + 4, h-28), f"X:{i/10:.1f}", fill=tc)
        for i in range(1, 10):
            y = int(h * i / 10)
            draw.line([(0, y), (w, y)], fill=gc, width=2)
            draw.text((8,    y + 4), f"Y:{i/10:.1f}", fill=tc)
            draw.text((w-78, y + 4), f"Y:{i/10:.1f}", fill=tc)

    def get_image_hash(self, image: Image.Image) -> str:
        w, h = image.size
        cropped = image.crop((0, int(h * 0.08), w, int(h * 0.92)))
        thumb = cropped.resize((16, 16), Image.Resampling.LANCZOS).convert("L")
        return hashlib.md5(thumb.tobytes()).hexdigest()

    def get_screen_similarity(self, img_a: Image.Image, img_b: Image.Image) -> float:
        """Structural similarity via numpy MAE — fast, no cv2/skimage needed.

        Crops 8% top/bottom (eliminates clock/spinner noise), downscales to
        320×180, converts to grayscale, then computes normalised mean-absolute
        difference mapped to 0.0 (totally different) … 1.0 (identical).

        Uses numpy instead of PIL.getdata() to avoid Pillow ≥11 deprecation
        warning and get ~10× faster computation.
        """
        import numpy as np

        def preprocess(img: Image.Image):
            w, h = img.size
            cropped = img.crop((0, int(h * 0.08), w, int(h * 0.92)))
            small = cropped.resize((320, 180), Image.Resampling.BILINEAR).convert("L")
            return np.array(small, dtype=np.int16)

        a, b = preprocess(img_a), preprocess(img_b)
        if a.shape != b.shape:
            return 0.0
        return 1.0 - float(np.mean(np.abs(a - b)) / 255.0)


# ============================================================================
# LLM Client
# ============================================================================

class QwenClient:

    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info(f"QwenClient initialized — model: {model_name}, base_url: {base_url}")

    def create_chat_completion(
        self,
        messages: list[dict],
        response_format: Optional[dict] = None,
        max_tokens: int = 1000,
        temperature: float = 0.2,
    ) -> str:
        kwargs = {
            "model":       self.model_name,
            "messages":    messages,
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format
        try:
            resp = self.client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except Exception as e:
            logger.error(f"API call failed: {e}")
            raise


# ============================================================================
# Action Executor  (pyautogui backend)
# ============================================================================

class ActionExecutor:
    """Execute GUI actions using pyautogui.

    Why pyautogui instead of hand-rolled ctypes:
    - moveTo uses smooth tweening (not linear lerp) → natural cursor movement
    - Internally uses MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE in one atomic
      SendInput call → no "ghost click at old position" race condition
    - hotkey() sends both vk AND scan codes → works in Chrome, games, etc.
    - Ten years of edge-case fixes (DPI, multi-monitor, UAC focus, etc.)

    The original deadlock was caused by lazy import inside a Rich Progress
    thread.  Importing pyautogui at module top-level (before any threads start)
    eliminates the deadlock entirely.
    """

    def __init__(
        self,
        screen_offset:     tuple[int, int] = (0, 0),
        screen_resolution: tuple[int, int] = (1920, 1080),
        debug_mode:        bool = False,
        on_click_callback  = None,
    ):
        self.screen_offset     = screen_offset
        self.screen_resolution = screen_resolution
        self.debug_mode        = debug_mode
        self.on_click_callback = on_click_callback
        logger.info(
            f"ActionExecutor ready (pyautogui backend). "
            f"Resolution: {screen_resolution}, offset: {screen_offset}. "
            f"Move mouse to corner to abort."
        )

    def _rel_to_sys(self, x: float, y: float) -> tuple[int, int]:
        """Convert 0.0–1.0 relative coords to physical screen pixels."""
        sw, sh = self.screen_resolution
        ox, oy = self.screen_offset
        return int(x * sw) + ox, int(y * sh) + oy

    # ------------------------------------------------------------------ #
    #  Public dispatch                                                     #
    # ------------------------------------------------------------------ #

    def execute(self, response: AgentResponse) -> bool:
        if response.action_type == ActionType.DONE:
            logger.info("Task completion signal received.")
            return True
        dispatch = {
            ActionType.CLICK:        self._click,
            ActionType.DOUBLE_CLICK: self._double_click,
            ActionType.RIGHT_CLICK:  self._right_click,
            ActionType.HOVER:        self._hover,
            ActionType.DRAG:         self._drag,
            ActionType.TYPE:         self._type,
            ActionType.SCROLL:       self._scroll,
            ActionType.HOTKEY:       self._hotkey,
            ActionType.WAIT:         self._wait,
        }
        handler = dispatch.get(response.action_type)
        if not handler:
            logger.warning(f"Unknown action type: {response.action_type}")
            return False
        try:
            return handler(response.action_params)
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return False

    # ------------------------------------------------------------------ #
    #  Individual action handlers                                          #
    # ------------------------------------------------------------------ #

    def _click(self, p: ActionParams) -> bool:
        if p.x is None or p.y is None:
            return False
        sx, sy = self._rel_to_sys(p.x, p.y)
        logger.info(f"CLICK  rel=({p.x:.3f},{p.y:.3f})  sys=({sx},{sy})")
        dur = 0.6 if self.debug_mode else 0.2
        pyautogui.moveTo(sx, sy, duration=dur)
        pyautogui.click()
        if self.on_click_callback:
            self.on_click_callback(p.x, p.y)
        return True

    def _double_click(self, p: ActionParams) -> bool:
        if p.x is None or p.y is None:
            return False
        sx, sy = self._rel_to_sys(p.x, p.y)
        logger.info(f"DCLICK rel=({p.x:.3f},{p.y:.3f})  sys=({sx},{sy})")
        pyautogui.moveTo(sx, sy, duration=0.2)
        pyautogui.doubleClick()
        if self.on_click_callback:
            self.on_click_callback(p.x, p.y)
        return True

    def _right_click(self, p: ActionParams) -> bool:
        if p.x is None or p.y is None:
            return False
        sx, sy = self._rel_to_sys(p.x, p.y)
        logger.info(f"RCLICK rel=({p.x:.3f},{p.y:.3f})  sys=({sx},{sy})")
        pyautogui.moveTo(sx, sy, duration=0.2)
        pyautogui.rightClick()
        if self.on_click_callback:
            self.on_click_callback(p.x, p.y)
        return True

    def _hover(self, p: ActionParams) -> bool:
        if p.x is None or p.y is None:
            return False
        sx, sy = self._rel_to_sys(p.x, p.y)
        logger.info(f"HOVER  rel=({p.x:.3f},{p.y:.3f})  sys=({sx},{sy})")
        pyautogui.moveTo(sx, sy, duration=0.4)
        time.sleep(0.6)
        if self.on_click_callback:
            self.on_click_callback(p.x, p.y)
        return True

    def _drag(self, p: ActionParams) -> bool:
        if None in (p.x, p.y, p.end_x, p.end_y):
            return False
        sx, sy = self._rel_to_sys(p.x, p.y)
        ex, ey = self._rel_to_sys(p.end_x, p.end_y)
        logger.info(f"DRAG   ({sx},{sy}) → ({ex},{ey})")
        pyautogui.moveTo(sx, sy, duration=0.2)
        pyautogui.dragTo(ex, ey, duration=0.7, button="left")
        if self.on_click_callback:
            self.on_click_callback(p.end_x, p.end_y)
        return True

    def _type(self, p: ActionParams) -> bool:
        """Input text via clipboard paste (supports CJK and all Unicode).

        Flow:
          1. Save current clipboard
          2. Copy target text to clipboard
          3. Ctrl+V (or Cmd+V on macOS)
          4. Restore original clipboard after a short delay

        This avoids pyautogui.typewrite() which only handles ASCII, and avoids
        the Win32Input KEYEVENTF_UNICODE path which requires the target window
        to explicitly handle WM_CHAR — many Electron/Chrome apps don't.
        """
        if not p.text:
            return False
        logger.info(f"TYPE   {p.text[:60]!r}")
        old_clip = ""
        try:
            old_clip = pyperclip.paste()
        except Exception:
            pass
        try:
            pyperclip.copy(p.text)
            time.sleep(0.08)
            paste_key = "command" if platform.system() == "Darwin" else "ctrl"
            pyautogui.hotkey(paste_key, "v")
            time.sleep(0.08)
        except Exception as e:
            logger.error(f"Clipboard paste failed: {e}")
            # ASCII fallback
            pyautogui.typewrite(p.text, interval=0.04)
        finally:
            # Restore clipboard after paste has been processed
            time.sleep(0.15)
            try:
                pyperclip.copy(old_clip)
            except Exception:
                pass
        return True

    def _scroll(self, p: ActionParams) -> bool:
        if p.scroll_amount is None:
            return False
        logger.info(f"SCROLL {p.scroll_amount}")
        pyautogui.scroll(p.scroll_amount)
        return True

    def _hotkey(self, p: ActionParams) -> bool:
        if not p.keys:
            return False
        keys = p.keys.lower().replace("+", " ").replace("-", " ").split()
        logger.info(f"HOTKEY {keys}")
        pyautogui.hotkey(*keys)
        return True

    def _wait(self, p: ActionParams) -> bool:
        secs = min(p.wait_seconds or 2.0, 10.0)
        logger.info(f"WAIT   {secs:.1f}s")
        time.sleep(secs)
        return True


# ============================================================================
# Prompt Architecture (3 layers)
# ============================================================================

CORE_PROMPT = """You are a GUI Agent that controls the computer. I provide screenshots with a green coordinate grid, screen resolution, and the task objective. You output the next action as pure JSON (no Markdown).

【Output Format】Return pure JSON, no ```json or other markers:
{
  "observation": "One-line description of the current screen state",
  "reasoning": "Analyze history (especially failures), explain your plan for this step",
  "confidence": 0.9,
  "fallback": "What to try if this action fails",
  "action_type": "CLICK|DOUBLE_CLICK|RIGHT_CLICK|HOVER|DRAG|TYPE|SCROLL|HOTKEY|WAIT|DONE",
  "action_params": { ... }
}

【Coordinate Rules】x/y are 0.0~1.0 relative coordinates. Green grid lines mark X:0.1~0.9 and Y:0.1~0.9. Red scale labels are drawn at edges.
- X:0.0 = left edge, X:1.0 = right edge; Y:0.0 = top, Y:1.0 = bottom
- Browser address bar: Y≈0.05~0.12; Taskbar: Y≈0.96~1.0
- Interpolate between grid lines (X:0.15 = midway between 0.1 and 0.2)
- A red crosshair marks your previous click position

【Action Parameters】
- CLICK / DOUBLE_CLICK / RIGHT_CLICK / HOVER: {"x": 0.5, "y": 0.3}
- DRAG: {"x": 0.2, "y": 0.5, "end_x": 0.8, "end_y": 0.5}
- TYPE: {"text": "hello world"}
- SCROLL: {"scroll_amount": 3}   (positive=up, negative=down)
- HOTKEY: {"keys": "ctrl l"}     (e.g. "win s", "ctrl t", "alt tab", "enter")
- WAIT: {"wait_seconds": 2.0}
- DONE: {}

【Strategy】
- Prefer HOTKEY over mouse for menus, address bars, shortcuts
- Prefer TYPE over clicking individual buttons/keys
- After 2+ consecutive failures: try SCROLL, ESC, Tab, or a completely different approach
- For coordinate misses: adjust by ±0.02 from last attempt

【Common Shortcuts】
"win s" = Windows search | "ctrl l" = address bar | "ctrl t" = new tab
"alt tab" = switch window | "esc" = close dialog | "enter" = confirm
"""

DOMAIN_KNOWLEDGE: dict[str, str] = {
    "youtube": "【YouTube】Navigate via address bar: HOTKEY ctrl+l → TYPE youtube.com → HOTKEY enter",
    "chrome":  "【Chrome】Ctrl+L = focus address bar, Ctrl+T = new tab, Ctrl+W = close tab",
    "excel":   "【Excel】F2 = edit cell, Ctrl+S = save, Ctrl+Z = undo",
    "word":    "【Word】Use Windows search (Win+S) to launch; Ctrl+N = new document",
    "linkedin":"【LinkedIn】Jobs → search keywords → All Filters → Easy Apply checkbox",
}


# ============================================================================
# JSON parsing helper
# ============================================================================

def parse_json_response(content: str) -> dict:
    for attempt in [
        lambda: json.loads(content),
        lambda: json.loads(re.search(r'```(?:json)?\s*({.*?})\s*```', content, re.DOTALL).group(1)),
        lambda: json.loads(re.search(r'({.*})', content, re.DOTALL).group(1)),
    ]:
        try:
            return attempt()
        except Exception:
            pass
    raise ValueError(f"Cannot parse JSON: {content[:200]}")


# ============================================================================
# GUI Agent
# ============================================================================

class GUIAgent:

    def __init__(self, task: str = "", max_iterations: int = 50, debug_mode: bool = False, config=None):
        self.task             = task
        self.max_iterations   = max_iterations
        self.debug_mode       = debug_mode
        self.action_history:  list[ActionRecord] = []
        self.failed_actions:  list[str]          = []
        self.consecutive_no_change: int          = 0
        self.last_parse_error: Optional[str]     = None

        self.screen_capture = ScreenCapture()

        def _on_click(rx: float, ry: float):
            self.screen_capture.last_click_pos = (rx, ry)
        self._on_click = _on_click

        # Executor initialised in run() after resolution is known
        self.action_executor: Optional[ActionExecutor] = None

        load_dotenv()
        if config:
            api_key    = config.api_key
            base_url   = config.base_url
            model_name = config.model
            self.max_iterations = config.max_iterations
        else:
            api_key    = os.getenv("DASHSCOPE_API_KEY")
            base_url   = os.getenv("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
            model_name = os.getenv("MODEL_NAME", "qwen3-vl-plus")

        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY not set")

        self.llm_client = QwenClient(api_key, base_url, model_name)
        logger.info(f"GUIAgent initialised — task: {task[:50] or 'N/A'}")

    # ------------------------------------------------------------------ #
    #  State management                                                    #
    # ------------------------------------------------------------------ #

    def reset_state(self, new_task: str = "") -> None:
        self.task              = new_task
        self.action_history.clear()
        self.failed_actions.clear()
        self.screen_capture.last_click_pos = None
        self.last_parse_error  = None
        self.consecutive_no_change = 0
        self._last_run_task    = new_task
        logger.info(f"State reset. New task: {new_task[:60] or 'N/A'}")

    def _get_screen_offset(self) -> tuple[int, int]:
        """Virtual desktop top-left offset (negative on left-extended multi-monitor)."""
        if platform.system() == "Windows":
            left = ctypes.windll.user32.GetSystemMetrics(76)  # SM_XVIRTUALSCREEN
            top  = ctypes.windll.user32.GetSystemMetrics(77)  # SM_YVIRTUALSCREEN
            return (left, top)
        return (0, 0)

    # ------------------------------------------------------------------ #
    #  Visual stability polling                                            #
    # ------------------------------------------------------------------ #

    def wait_for_visual_stability(
        self,
        before_hash:  str,
        before_image: Image.Image,
        max_wait:     float = 10.0,
        poll_interval: float = 0.8,
        stability_threshold: int = 2,
    ) -> tuple[bool, str, float, float]:
        """Poll until the screen stops changing, then return.

        Two-level detection:
          • "changed"  — compare every frame against the *initial* screenshot
                         (SSIM < 0.98 means the action had visible effect)
          • "stable"   — compare each frame against the *previous* frame
                         (SSIM > 0.995 means animation/cursor settled)

        Using consecutive-frame SSIM for stability (instead of hash equality)
        avoids the cursor-blink false-positive: a blinking caret changes the
        hash every 500 ms and would otherwise keep resetting stable_count
        until the 10 s timeout fires.
        """
        changed          = False
        stable_count     = 0
        elapsed          = 0.0
        similarity_start = 1.0   # similarity vs the very first frame
        last_image       = before_image

        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval

            cur_img = self.screen_capture.capture_screen()

            # ── Has the action had any visible effect? ──────────────
            if not changed:
                similarity_start = self.screen_capture.get_screen_similarity(
                    before_image, cur_img
                )
                if similarity_start < 0.98:
                    changed = True

            # ── Has the animation / transition finished? ────────────
            # Compare consecutive frames; > 0.995 tolerates cursor blink
            # but rejects loading spinners, scrolling, etc.
            sim_consecutive = self.screen_capture.get_screen_similarity(last_image, cur_img)

            if sim_consecutive > 0.995:
                stable_count += 1
                if stable_count >= stability_threshold:
                    verb = "after change" if changed else "no change"
                    logger.info(
                        f"Page stable {verb} ({elapsed:.1f}s, "
                        f"SSIM_start={similarity_start:.4f})"
                    )
                    return changed, "", elapsed, similarity_start
            else:
                stable_count = 0   # still animating

            last_image = cur_img

        # Timeout — return whatever we know
        logger.warning(f"Wait timeout ({max_wait}s), SSIM_start={similarity_start:.4f}")
        return changed, "", elapsed, similarity_start

    # ------------------------------------------------------------------ #
    #  History / blacklist helpers                                         #
    # ------------------------------------------------------------------ #

    def _action_to_str(self, r: AgentResponse) -> str:
        p, at = r.action_params, r.action_type
        f = lambda v: f"{v:.2f}" if v is not None else "None"
        if at in (ActionType.CLICK, ActionType.DOUBLE_CLICK, ActionType.RIGHT_CLICK, ActionType.HOVER):
            return f"{at.value}({f(p.x)},{f(p.y)})"
        if at == ActionType.DRAG:
            return f"DRAG({f(p.x)},{f(p.y)}->{f(p.end_x)},{f(p.end_y)})"
        if at == ActionType.HOTKEY:
            return f"HOTKEY({p.keys})"
        if at == ActionType.TYPE:
            return f"TYPE({(p.text or '')[:15]})"
        return at.value

    @staticmethod
    def _near(action_str: str, cx: float, cy: float, thr: float = 0.05) -> bool:
        m = re.search(r'\(([0-9.]+),([0-9.]+)\)', action_str)
        if m:
            return abs(float(m.group(1)) - cx) < thr and abs(float(m.group(2)) - cy) < thr
        return False

    # ------------------------------------------------------------------ #
    #  Prompt construction                                                 #
    # ------------------------------------------------------------------ #

    def _domain_knowledge(self) -> str:
        return "\n".join(v for k, v in DOMAIN_KNOWLEDGE.items() if k in self.task.lower())

    def _build_prompt(self, b64_image: str, resolution: tuple[int, int]) -> list[dict]:
        w, h = resolution
        system = CORE_PROMPT + ("\n" + self._domain_knowledge() if self._domain_knowledge() else "")

        history = "None"
        if self.action_history:
            lines = []
            for i, rec in enumerate(self.action_history[-5:]):
                line = f"  {i+1}. [{rec.status_tag}] {rec.action_type}"
                if rec.coordinates:
                    line += f" at ({rec.coordinates[0]:.2f},{rec.coordinates[1]:.2f})"
                line += f" | {rec.thought[:60]}"
                if rec.similarity_score < 1.0:
                    line += f" (SSIM={rec.similarity_score:.3f})"
                lines.append(line)
            history = "\n".join(lines)

        extras = ""
        if self.failed_actions:
            bl = "\n".join(f"  - {a}" for a in self.failed_actions[-5:])
            extras += f"\n[⚠️ BLACKLISTED — do NOT repeat]:\n{bl}\n(Switch to a completely different approach!)"
        if self.last_parse_error:
            extras += f"\n\n[❗JSON ERROR]: {self.last_parse_error}\nReturn ONLY pure JSON, no markdown."
        if self.consecutive_no_change >= 2:
            extras += (
                f"\n\n[⚠️ CRITICAL] {self.consecutive_no_change} consecutive actions had NO visible effect! "
                "You MUST try a completely different approach (SCROLL, ESC, HOTKEY, WAIT)."
            )

        user_msg = (
            f"Task: {self.task}\n"
            f"Screen: {w}×{h}{extras}\n\n"
            f"Action history (last 5):\n{history}\n\n"
            "Analyse the screenshot and output your next action as JSON."
        )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": b64_image}},
                {"type": "text",      "text": user_msg},
            ]},
        ]

    # ------------------------------------------------------------------ #
    #  Response parsing                                                    #
    # ------------------------------------------------------------------ #

    def _parse_response(self, content: str) -> AgentResponse:
        data       = parse_json_response(content)
        thought    = data.get("reasoning") or data.get("thought") or ""
        observation = data.get("observation", "")
        confidence = float(data.get("confidence", 0.5))
        fallback   = data.get("fallback", "")

        if not thought and not observation:
            raise ValueError("Response missing 'reasoning'/'thought' and 'observation'")
        if "action_type" not in data:
            raise ValueError("Response missing 'action_type'")

        try:
            action_type = ActionType(data["action_type"].upper())
        except ValueError:
            raise ValueError(f"Invalid action_type: {data['action_type']}")

        pd = data.get("action_params") or {}

        def sf(v):
            if v is None: return None
            if isinstance(v, list): v = v[0] if v else None
            try: return float(v)
            except: return None

        params = ActionParams(
            x=sf(pd.get("x")), y=sf(pd.get("y")),
            end_x=sf(pd.get("end_x")), end_y=sf(pd.get("end_y")),
            text=pd.get("text"),
            scroll_amount=pd.get("scroll_amount"),
            keys=pd.get("keys"),
            wait_seconds=sf(pd.get("wait_seconds")),
        )

        # Recover missing keys for HOTKEY from reasoning text
        if action_type == ActionType.HOTKEY and not params.keys:
            rl = thought.lower()
            shortcuts = {
                ("ctrl","l"): "ctrl l", ("ctrl","t"): "ctrl t",
                ("win","s"):  "win s",  ("win","d"): "win d",
                ("alt","tab"):"alt tab","enter": "enter", "esc": "esc",
            }
            for hint, val in shortcuts.items():
                if isinstance(hint, tuple):
                    if all(k in rl for k in hint):
                        params.keys = val; break
                elif hint in rl:
                    params.keys = val; break
            if not params.keys:
                raise ValueError("HOTKEY missing 'keys'")

        if observation:
            logger.info(f"[Observation] {observation}")
        logger.info(f"[Reasoning] {thought}")
        if confidence < 0.5:
            logger.warning(f"Low confidence {confidence:.2f} — fallback: {fallback}")

        return AgentResponse(
            thought=thought, action_type=action_type, action_params=params,
            observation=observation, confidence=confidence, fallback=fallback,
        )

    # ------------------------------------------------------------------ #
    #  Main loop                                                           #
    # ------------------------------------------------------------------ #

    def run(self) -> bool:
        if not self.task:
            raise ValueError("Task is empty — call reset_state(task) first")

        # Initialise screen geometry + executor on every run() call
        # (resolution can change between tasks if monitors are reconfigured)
        offset     = self._get_screen_offset()
        resolution = self.screen_capture.get_screen_resolution()
        self.action_executor = ActionExecutor(
            screen_offset=offset,
            screen_resolution=resolution,
            debug_mode=self.debug_mode,
            on_click_callback=self._on_click,
        )

        logger.info("=" * 56)
        logger.info(f"GUI Agent starting — task: {self.task}")
        logger.info(f"Resolution: {resolution}, offset: {offset}")
        logger.info("=" * 56)

        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"\n--- Iteration {iteration}/{self.max_iterations} ---")
            try:
                # ── Perceive ──────────────────────────────────────────
                raw_before   = self.screen_capture.capture_screen()
                before_hash  = self.screen_capture.get_image_hash(raw_before)
                logger.info(f"Hash before: {before_hash[:16]}…")

                display = raw_before.copy()
                self.screen_capture.draw_grid_overlay(display)
                if self.screen_capture.last_click_pos:
                    lx, ly = self.screen_capture.last_click_pos
                    self.screen_capture.draw_click_indicator_at_pixels(
                        display,
                        int(lx * display.width),
                        int(ly * display.height),
                    )
                display   = self.screen_capture.compress_image(display)
                b64_image = self.screen_capture.image_to_base64(display)

                # ── Think ─────────────────────────────────────────────
                logger.info(f"Calling {self.llm_client.model_name}…")
                raw_resp = self.llm_client.create_chat_completion(
                    messages=self._build_prompt(b64_image, resolution),
                    response_format={"type": "json_object"},
                )
                response = self._parse_response(raw_resp)
                self.last_parse_error = None

                # ── Blacklist check ───────────────────────────────────
                action_str = self._action_to_str(response)
                if action_str in self.failed_actions:
                    logger.warning(f"⛔ Blocking blacklisted action: {action_str}")
                    self.action_history.append(ActionRecord(
                        iteration=iteration, action_type=action_str,
                        thought="Blocked (known failure)", status_tag="⛔Blocked",
                    ))
                    self.consecutive_no_change += 1
                    time.sleep(0.5)
                    continue

                logger.info(
                    f"[Action] {response.action_type.value} "
                    f"(conf={response.confidence:.2f})"
                )

                # ── Act ───────────────────────────────────────────────
                self.action_executor.execute(response)

                # ── Observe ───────────────────────────────────────────
                changed, after_hash, elapsed, similarity = self.wait_for_visual_stability(
                    before_hash, raw_before
                )

                # ── Update state ──────────────────────────────────────
                if response.action_type == ActionType.WAIT:
                    status_tag = "⏳Wait"
                    self.consecutive_no_change = 0
                elif changed:
                    self.consecutive_no_change = 0
                    p = response.action_params
                    if p.x is not None:
                        self.failed_actions = [
                            fa for fa in self.failed_actions
                            if not self._near(fa, p.x, p.y)
                        ]
                    else:
                        self.failed_actions.clear()
                    status_tag = "✅Changed"
                    logger.info(f"[{status_tag}] SSIM={similarity:.4f}, {elapsed:.1f}s")
                else:
                    self.consecutive_no_change += 1
                    self.failed_actions.append(action_str)
                    status_tag = f"❌NoChange[{action_str}]"
                    logger.warning(f"[{status_tag}] SSIM={similarity:.4f}, {elapsed:.1f}s")

                p = response.action_params
                coords = (p.x, p.y) if p.x is not None else None
                self.action_history.append(ActionRecord(
                    iteration=iteration,
                    action_type=response.action_type.value,
                    action_params={"x": p.x, "y": p.y, "text": p.text,
                                   "keys": p.keys, "scroll_amount": p.scroll_amount},
                    thought=response.thought,
                    observation=response.observation,
                    coordinates=coords,
                    screen_changed=changed,
                    similarity_score=similarity,
                    wait_elapsed=elapsed,
                    status_tag=status_tag,
                ))

                if response.action_type == ActionType.DONE:
                    logger.info("Task completed ✓")
                    return True

            except (ValueError, json.JSONDecodeError) as e:
                msg = str(e)[:150]
                logger.error(f"Parse error: {e}")
                self.last_parse_error = msg
                self.action_history.append(ActionRecord(
                    iteration=iteration, action_type="PARSE_ERROR",
                    thought=f"Format error: {msg}", status_tag="❌ParseFail", error=msg,
                ))
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Iteration error: {e}", exc_info=True)
                time.sleep(0.5)

        logger.warning(f"Max iterations ({self.max_iterations}) reached")
        return False


# ============================================================================
# Entry point
# ============================================================================

def main():
    try:
        from cli import (
            console, show_welcome_panel, show_safety_warning, show_config_panel,
            log_info, log_success, log_warning, log_error,
            get_user_instruction, show_action_history, show_action_summary,
            Command, TaskProgress,
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
            log_warning(f"API key not configured: {get_api_key_status(config)}")

        agent = GUIAgent(config=config)
        running      = True
        all_actions: list[dict] = []

        while running:
            instruction = get_user_instruction("Enter your instruction")
            if instruction is None:
                running = False
                continue

            if Command.is_command(instruction):
                cmd = Command.get_command_type(instruction)
                if cmd == "help":
                    from cli import show_help; show_help()
                elif cmd == "quit":
                    running = False
                elif cmd == "history":
                    show_action_history(all_actions)
                elif cmd == "clear":
                    console.clear()
                elif cmd == "config":
                    show_config_panel({"model": config.model, "base_url": config.base_url,
                                       "max_iterations": config.max_iterations})
                continue

            console.print()
            log_info(f"Task: {instruction}")
            console.print()

            try:
                with TaskProgress("Executing task…"):
                    agent.reset_state(instruction)
                    success = agent.run()
                console.print()
                (log_success if success else log_warning)(
                    "Task completed!" if success else "Task incomplete — check logs."
                )
            except KeyboardInterrupt:
                console.print()
                log_warning("Interrupted by user.")
            except Exception as e:
                console.print()
                log_error(f"Error: {e}")
            finally:
                for rec in agent.action_history:
                    all_actions.append({
                        "action":     rec.action_type,
                        "details":    rec.thought,
                        "status":     "success" if rec.screen_changed else "failed",
                        "status_tag": rec.status_tag,
                    })

        if all_actions:
            ok  = sum(1 for a in all_actions if a["status"] == "success")
            bad = sum(1 for a in all_actions if a["status"] == "failed")
            show_action_summary(len(all_actions), ok, bad)
        console.print("[bold blue]Goodbye![/bold blue]")

    except ImportError:
        # Minimal fallback if cli/config modules are absent
        print("GUI Agent starting (minimal mode — cli module not found).")
        agent = GUIAgent()
        while True:
            try:
                instruction = input("\nInstruction (or quit): ").strip()
                if not instruction or instruction.lower() in ("quit", "exit"):
                    break
                agent.reset_state(instruction)
                agent.run()
            except KeyboardInterrupt:
                print("\nInterrupted.")
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()