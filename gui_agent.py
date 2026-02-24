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
import dataclasses
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
    FOCUS        = "FOCUS"
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
        # P3: Full-screen screenshot cache for FOCUS region reuse
        self._cached_full: Optional[Image.Image] = None
        self._cached_time: float = 0.0

    @staticmethod
    def _gdi_capture() -> Image.Image:
        """Capture full virtual desktop via Win32 GDI BitBlt.

        Preferred over PIL.ImageGrab / mss because:
        - No import-time display-server connection (no deadlock)
        - Handles multi-monitor negative-offset layouts correctly
        - Returns physical pixels (consistent with DPI-aware coordinates)

        Falls back to ImageGrab.grab on GDI failure (UAC prompts, exclusive apps).
        """
        if platform.system() != "Windows":
            return ImageGrab.grab(all_screens=True)
        try:
            return ScreenCapture._gdi_capture_win32()
        except Exception as e:
            logger.warning(f"GDI capture failed ({e}) — falling back to ImageGrab")
            return ImageGrab.grab(all_screens=True)

    @staticmethod
    def _gdi_capture_win32() -> Image.Image:

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
        img = self._gdi_capture()
        self._cached_full = img
        self._cached_time = time.time()
        return img

    def image_to_base64(self, image: Image.Image, fmt: str = "JPEG", quality: int = 70) -> str:
        buf = io.BytesIO()
        save_kwargs = {"format": fmt}
        if fmt.upper() == "JPEG":
            save_kwargs["quality"] = quality
        image.save(buf, **save_kwargs)
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
        # P2-1: Use RGB mode — JPEG discards alpha anyway
        draw = ImageDraw.Draw(image)
        gc = (180, 255, 180)   # pale green, subtle on RGB
        tc = (255, 0, 0)
        for i in range(1, 10):
            x = int(w * i / 10)
            draw.line([(x, 0), (x, h)], fill=gc, width=1)
            draw.text((x + 4, 8),    f"X:{i/10:.1f}", fill=tc)
            draw.text((x + 4, h-28), f"X:{i/10:.1f}", fill=tc)
        for i in range(1, 10):
            y = int(h * i / 10)
            draw.line([(0, y), (w, y)], fill=gc, width=1)
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

    # ── Phase 2 Addition: FOCUS local zoom ─────────────────────────────────────

    FOCUS_SIZE_PX: int = 400

    def capture_focus_region(
        self,
        cx: float,
        cy: float,
        focus_w_px: int = 400,
        focus_h_px: int = 400,
    ) -> tuple[str, int, int]:
        # P3: Reuse cached full screenshot if < 0.5s old
        if (self._cached_full is not None
                and time.time() - self._cached_time < 0.5):
            full_img = self._cached_full
        else:
            full_img = self.capture_screen()
        W, H = full_img.size

        cx_px = cx * W
        cy_px = cy * H
        x1 = max(0, int(cx_px - focus_w_px / 2))
        y1 = max(0, int(cy_px - focus_h_px / 2))
        x2 = min(W, int(cx_px + focus_w_px / 2))
        y2 = min(H, int(cy_px + focus_h_px / 2))

        roi      = full_img.crop((x1, y1, x2, y2))
        actual_w = x2 - x1
        actual_h = y2 - y1

        # Preserve aspect ratio — avoid distorting the crop when near screen
        # edges (actual_w/h may differ from requested 400×400), which would
        # impair the VLM's spatial coordinate reasoning.
        ratio    = min(800 / actual_w, 800 / actual_h)
        new_w    = int(actual_w * ratio)
        new_h    = int(actual_h * ratio)
        roi_enlarged = roi.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self._draw_fine_grid(roi_enlarged, divisions=10)

        return self.image_to_base64(roi_enlarged), actual_w, actual_h

    def _draw_fine_grid(self, image: Image.Image, divisions: int = 10) -> None:
        width, height = image.size
        draw       = ImageDraw.Draw(image)          # Fix 6: RGB mode (no RGBA)
        grid_color = (0, 180, 220)                  # solid cyan
        text_color = (255, 140, 0)                  # solid orange

        for i in range(1, divisions):
            x = int(width * i / divisions)
            draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
            draw.text((x + 2, 5), f"{i/divisions:.1f}", fill=text_color)

        for i in range(1, divisions):
            y = int(height * i / divisions)
            draw.line([(0, y), (width, y)], fill=grid_color, width=1)
            draw.text((5, y + 2), f"{i/divisions:.1f}", fill=text_color)


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
        max_retries: int = 3,
    ) -> str:
        kwargs = {
            "model":       self.model_name,
            "messages":    messages,
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format

        # P1-2: Exponential backoff retry — network errors don't waste iteration quota
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"API call failed after {max_retries} attempts: {e}")
                    raise
                wait = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(f"API retry {attempt+1}/{max_retries} in {wait}s: {e}")
                time.sleep(wait)


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
          1. Capture foreground window handle (focus guard)
          2. Save current clipboard
          3. Copy target text to clipboard
          4. Ctrl+V (or Cmd+V on macOS)
          5. Restore original clipboard after a short delay

        This avoids pyautogui.typewrite() which only handles ASCII, and avoids
        the Win32Input KEYEVENTF_UNICODE path which requires the target window
        to explicitly handle WM_CHAR — many Electron/Chrome apps don't.
        """
        if not p.text:
            return False
        logger.info(f"TYPE   {p.text[:60]!r}")
        
        # Fix 3: Capture foreground window BEFORE any clipboard operations
        pre_hwnd = None
        if platform.system() == "Windows":
            try:
                pre_hwnd = ctypes.windll.user32.GetForegroundWindow()
            except Exception:
                pass
        
        old_clip = ""
        try:
            old_clip = pyperclip.paste()
        except Exception:
            pass
        try:
            pyperclip.copy(p.text)
            time.sleep(0.15)   # P0-1: increased delay for clipboard propagation

            # P0-1: Verify clipboard contents match what we wrote
            try:
                actual = pyperclip.paste()
                if actual != p.text:
                    logger.warning(f"Clipboard mismatch — retrying copy")
                    pyperclip.copy(p.text)
                    time.sleep(0.1)
            except Exception:
                pass

            # P0-1: Focus guard — log warning if foreground window changed
            if platform.system() == "Windows":
                try:
                    hwnd = ctypes.windll.user32.GetForegroundWindow()
                    buf = ctypes.create_unicode_buffer(256)
                    ctypes.windll.user32.GetWindowTextW(hwnd, buf, 256)
                    if buf.value:
                        logger.debug(f"Paste target window: {buf.value[:50]}")
                except Exception:
                    pass

            paste_key = "command" if platform.system() == "Darwin" else "ctrl"

            # Focus guard: abort paste if focus changed after moveTo
            if pre_hwnd is not None:
                cur_hwnd = ctypes.windll.user32.GetForegroundWindow()
                if cur_hwnd != pre_hwnd:
                    logger.warning("Focus changed before paste — aborting TYPE, will retry")
                    return False

            pyautogui.hotkey(paste_key, "v")
            time.sleep(0.15)   # P0-1: increased post-paste delay
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
        amount = int(p.scroll_amount)  # guard against JSON floats like -3.0
        if p.x is not None and p.y is not None:
            sx, sy = self._rel_to_sys(p.x, p.y)
            logger.info(f"SCROLL {amount} at rel=({p.x:.3f},{p.y:.3f}) sys=({sx},{sy})")
            pyautogui.moveTo(sx, sy, duration=0.1)
        else:
            logger.info(f"SCROLL {amount} (at current mouse position)")
        pyautogui.scroll(amount)
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

CORE_PROMPT = """You are a GUI Agent that controls a computer through screenshots and actions.
I provide: a screenshot with a coordinate grid, screen resolution, task objective, and action history.
You output: the next action as pure JSON (no Markdown, no code blocks, no extra text).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【OUTPUT FORMAT — MANDATORY】
Return ONLY this JSON structure, nothing else:
{
  "observation": "One sentence describing what is currently visible on screen",
  "reasoning":   "Why you chose this action; if history has ❌, explain why old approach failed",
  "confidence":  0.9,
  "fallback":    "What to try if this action fails",
  "action_type": "CLICK|DOUBLE_CLICK|RIGHT_CLICK|HOVER|DRAG|TYPE|SCROLL|HOTKEY|WAIT|FOCUS|DONE",
  "action_params": { ... }
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【COORDINATE SYSTEM】
x and y are relative coordinates: 0.0 = left/top edge, 1.0 = right/bottom edge.

The screenshot has a GREEN grid with lines at 0.1 intervals.
RED scale labels appear at the edges.
A RED CROSSHAIR marks your previous click position.

Reading the grid:
- Grid lines divide the screen into 10×10 = 100 equal zones
- Interpolate between lines for precision (e.g., X:0.15 is halfway between 0.1 and 0.2)
- Common regions: browser address bar Y:0.05~0.08 | taskbar Y:0.96~1.0 | tab bar Y:0.03~0.06

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【ACTION REFERENCE】

CLICK / DOUBLE_CLICK / RIGHT_CLICK / HOVER
  action_params: {"x": 0.5, "y": 0.3}
  Use HOVER to trigger dropdowns or tooltips before clicking.
  Use DOUBLE_CLICK to open files/folders.
  Use RIGHT_CLICK for context menus.

DRAG
  action_params: {"x": 0.3, "y": 0.6, "end_x": 0.7, "end_y": 0.6}
  For sliders, scrollbars, resizing, or reordering items.

TYPE
  action_params: {"text": "Hello World"}
  Types via clipboard paste — supports Chinese and all Unicode.
  IMPORTANT: click the input field first, then TYPE.

SCROLL
  action_params: {"scroll_amount": -3, "x": 0.5, "y": 0.5}
  Positive = scroll up, Negative = scroll down. Use ±3 for normal, ±8 for fast.
  IMPORTANT: Always specify x,y to scroll inside the correct container/panel.

HOTKEY
  action_params: {"keys": "ctrl l"}
  Key names: ctrl, alt, shift, win, cmd, tab, enter, esc, space, f1~f12, up, down, left, right
  Separate keys with space: "ctrl shift t", "alt f4", "win s"
  Common shortcuts:
    "ctrl l"     → focus address bar (Chrome)
    "ctrl t"     → new tab
    "ctrl w"     → close tab
    "win s"      → Windows search
    "alt tab"    → switch window
    "esc"        → cancel / close dialog
    "enter"      → confirm
    "ctrl a"     → select all
    "ctrl c/v"   → copy / paste

WAIT
  action_params: {"wait_seconds": 2.0}
  Use when page is loading. Max 10 seconds. Never use to stall.

FOCUS  ← 🔍 PRECISION ZOOM
  action_params: {"x": 0.85, "y": 0.06}
  Triggers a high-resolution zoom of the region around (x, y).
  The next turn will show a ZOOMED-IN cyan-grid view of that area.
  Use when: target is visible but too small to click accurately (confidence < 0.65).
  Do NOT use: if already in FOCUS mode, or when target location is clear.

DONE
  action_params: {}
  Only output DONE when you can SEE visible proof the task is complete.
  Required: mention visible evidence in the "observation" field.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【FOCUS MODE — DETAIL】

When you are IN FOCUS MODE, you will see:
  - A CYAN grid (not green) with 10 divisions
  - ORANGE coordinate labels (not red)
  - A zoomed-in view of a specific screen region

In FOCUS MODE:
  ✅ Output CLICK / DOUBLE_CLICK with LOCAL coordinates (0~1 within the crop)
  ✅ Be precise — you are zoomed in, accuracy matters
  ❌ Do NOT output FOCUS again (you are already zoomed in)
  ❌ Do NOT use global coordinates from memory

Example FOCUS → CLICK sequence:
  Turn N:   {"action_type":"FOCUS","action_params":{"x":0.85,"y":0.06},"confidence":0.5,...}
  Turn N+1: {"action_type":"CLICK","action_params":{"x":0.42,"y":0.35},"confidence":0.9,...}
             (these are LOCAL coords within the zoomed crop)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【ANTI-FAILURE STRATEGY】

RULE 1 — Never repeat a blacklisted action.
  If your planned action appears in [⚠️ BLACKLISTED], switch strategy completely.
  Options: different coordinates, keyboard shortcut, SCROLL to reveal, HOVER first, ESC then retry.

RULE 2 — Learn from ❌ history.
  In "reasoning", explicitly state: "Previous CLICK at (x,y) failed because [reason]. New approach: [strategy]."
  Never use vague reasoning like "I will try again."

RULE 3 — Keyboard-first for input.
  For text entry, search boxes, address bars, calculators:
    HOTKEY "ctrl l" → TYPE "url" → HOTKEY "enter"    (browser navigation)
    TYPE "3+3="                                        (calculator — direct keystroke)
  Prefer TYPE over clicking individual keys.

RULE 4 — Escalation ladder for stuck states.
  After 2+ consecutive ❌:
    1. HOTKEY "esc" to dismiss dialogs
    2. SCROLL to reveal hidden elements
    3. HOVER over the area to trigger tooltips/dropdowns
    4. FOCUS to zoom in and re-assess
    5. HOTKEY "alt tab" to check if another window is blocking

RULE 5 — Use FOCUS before giving up on a CLICK.
  If an element is visible but small or densely packed:
    → Output FOCUS at that area first
    → Then CLICK with precise local coordinates in the next turn
  This is more reliable than guessing coordinates on the full screen.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【CONFIDENCE GUIDE】

0.95  Element clearly visible, position certain, no ambiguity
0.80  Element visible, coordinates estimated from grid
0.65  Element partially visible or in dense UI — consider FOCUS
0.50  Location uncertain — USE FOCUS or use fallback strategy
<0.50 Do NOT output CLICK; use FOCUS, SCROLL, or HOTKEY instead

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【COMMON MISTAKES — learn from these failures】

❌ CLICK on input field → TYPE had no effect
  → Fix: Verify cursor is blinking inside the field before using TYPE
❌ SCROLL without x,y → wrong panel scrolled
  → Fix: Always specify x,y coordinates targeting the correct scrollable container
❌ Repeated CLICK at same position 3+ times expecting different result
  → Fix: After 2 failures, try HOTKEY shortcut or completely different coordinates
❌ TYPE in address bar without clearing old text first
  → Fix: HOTKEY "ctrl a" to select all, then TYPE the new URL
❌ DONE without visible evidence on screen
  → Fix: Only DONE when you can cite specific visible proof in observation field

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【COMPLETE EXAMPLES】

Navigate via address bar:
{"observation":"Chrome is open, address bar visible at top","reasoning":"Ctrl+L focuses address bar reliably without needing to click","confidence":0.95,"fallback":"Click directly on address bar","action_type":"HOTKEY","action_params":{"keys":"ctrl l"}}

Type URL after focusing:
{"observation":"Address bar is focused and highlighted","reasoning":"Address bar is active, ready to type","confidence":0.95,"fallback":"Double-click address bar to select all first","action_type":"TYPE","action_params":{"text":"https://linkedin.com"}}

FOCUS when target is small:
{"observation":"Toolbar visible but icons are too small to distinguish","reasoning":"Cannot determine which icon is Settings at this zoom level","confidence":0.5,"fallback":"Try Tab key to cycle through toolbar items","action_type":"FOCUS","action_params":{"x":0.92,"y":0.05}}

CLICK inside FOCUS mode (cyan grid visible):
{"observation":"FOCUS mode: zoomed toolbar shows gear icon clearly at local ~0.35, 0.48","reasoning":"In FOCUS mode with cyan grid; gear icon at local coordinates 0.35, 0.48","confidence":0.9,"fallback":"HOTKEY alt+s for settings shortcut","action_type":"CLICK","action_params":{"x":0.35,"y":0.48}}

HOVER to reveal dropdown:
{"observation":"Navigation bar visible, Jobs menu item present","reasoning":"Hover first to reveal sub-menu before clicking","confidence":0.85,"fallback":"Click directly on Jobs","action_type":"HOVER","action_params":{"x":0.22,"y":0.06}}

DONE with evidence:
{"observation":"Calculator shows 6912 which equals 1234+5678. Task complete.","reasoning":"Result is clearly displayed on screen","confidence":0.98,"fallback":"N/A","action_type":"DONE","action_params":{}}
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

def focus_to_global(
    local_x:   float,
    local_y:   float,
    focus_cx:  float,
    focus_cy:  float,
    focus_w_px: int,
    focus_h_px: int,
    screen_w:  int,
    screen_h:  int,
) -> tuple[float, float]:
    offset_x_px = (local_x - 0.5) * focus_w_px
    offset_y_px = (local_y - 0.5) * focus_h_px
    center_x_px = focus_cx * screen_w
    center_y_px = focus_cy * screen_h
    global_x = max(0.0, min(1.0, (center_x_px + offset_x_px) / screen_w))
    global_y = max(0.0, min(1.0, (center_y_px + offset_y_px) / screen_h))
    return global_x, global_y



def _safe_int(v) -> Optional[int]:
    """Convert scroll_amount to int, handling Unicode minus and floats."""
    if v is None:
        return None
    try:
        # normalise full-width minus (\u2212) and en-dash to ASCII minus
        return int(str(v).replace('\u2212', '-').replace('\uff0d', '-').replace('\u2013', '-').strip().split('.')[0])
    except (TypeError, ValueError):
        logger.warning(f"scroll_amount could not be converted to int: {v!r}")
        return None


# Per-action-type threshold for consecutive-no-change CRITICAL alert.
# WAIT is effectively exempt (set to large number) because WAIT implies waiting.
_NO_CHANGE_THRESHOLDS: dict = {
    "SCROLL":       4,
    "TYPE":         3,
    "HOTKEY":       2,
    "WAIT":         99,
    "_default":     2,
}


def _no_change_threshold(action_type) -> int:
    if action_type is None:
        return _NO_CHANGE_THRESHOLDS["_default"]
    key = action_type.value if hasattr(action_type, 'value') else str(action_type)
    return _NO_CHANGE_THRESHOLDS.get(key, _NO_CHANGE_THRESHOLDS["_default"])


def parse_json_response(content: str) -> dict:
    for attempt in [
        lambda: json.loads(content),
        lambda: json.loads(re.search(r'```(?:json)?\s*({.*?})\s*```', content, re.DOTALL).group(1)),
        lambda: json.loads(re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', content, re.DOTALL).group(1)),
        lambda: json.loads(re.search(r'({.*?})', content, re.DOTALL).group(1)),
    ]:
        try:
            return attempt()
        except Exception:
            pass
    raise ValueError(f"Cannot parse JSON: {content[:200]}")


# ============================================================================
# Feature 2: DONE Verification prompt
# ============================================================================

VERIFY_PROMPT = """You are a task completion verifier for a GUI automation agent.
Look at the provided screenshot and the success criteria.

Answer ONLY with pure JSON — no Markdown, no extra text:
{"verified": true, "evidence": "one-sentence description of what you see confirming completion"}
or
{"verified": false, "evidence": "one-sentence description of what is missing or wrong"}

Do NOT suggest any actions. Just verify what is visible on screen now."""


# ============================================================================
# GUI Agent
# ============================================================================

class GUIAgent:

    def __init__(self, task: str = "", max_iterations: int = 50, debug_mode: bool = False,
                 config=None, history_log_path: Optional[str] = None):
        self.task             = task
        self.max_iterations   = max_iterations
        self.debug_mode       = debug_mode
        self.action_history:  list[ActionRecord] = []
        self.failed_action_counts: dict[str, int] = {}
        self.consecutive_no_change: int          = 0
        self.last_parse_error: Optional[str]     = None
        self.milestone_summaries: list[str]      = []  # key state changes
        self.history_log_path: Optional[str]     = history_log_path  # Fix 5: init param

        # FOCUS state — initialised here so run() never hits AttributeError
        # even if reset_state() is never called.
        self._focus_mode:  bool            = False
        self._focus_cx:    Optional[float] = None
        self._focus_cy:    Optional[float] = None
        self._focus_w_px:  int             = 400
        self._focus_h_px:  int             = 400
        self._global_goal:      str = ""
        self._current_step:     str = ""
        self._success_criteria: str = ""
        self._prior_state:      str = ""

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
        self.failed_action_counts.clear()
        self.milestone_summaries.clear()
        # Note: history_log_path is intentionally NOT cleared — keep logging the same session file
        self.screen_capture.last_click_pos = None
        self.last_parse_error  = None
        self.consecutive_no_change = 0
        self._last_run_task    = new_task
        self._focus_mode:  bool            = False
        self._focus_cx:    Optional[float] = None
        self._focus_cy:    Optional[float] = None
        self._focus_w_px:  int             = 400
        self._focus_h_px:  int             = 400
        self._global_goal:      str = ""
        self._current_step:     str = ""
        self._success_criteria: str = ""
        self._prior_state:      str = ""
        logger.info(f"State reset. New task: {new_task[:60] or 'N/A'}")

    def set_global_context(
        self, global_goal: str, current_step: str, success_criteria: str,
        prior_state: str = "",
    ) -> None:
        self._global_goal      = global_goal
        self._current_step     = current_step
        self._success_criteria = success_criteria
        self._prior_state      = prior_state

    def _enable_focus_mode(self, cx: float, cy: float) -> None:
        self._focus_mode = True
        self._focus_cx   = cx
        self._focus_cy   = cy
        self._focus_w_px = 400
        self._focus_h_px = 400
        logger.info(f"[FOCUS] Enabled at ({cx:.3f}, {cy:.3f})")

    def _disable_focus_mode(self) -> None:
        self._focus_mode = False
        self._focus_cx   = None
        self._focus_cy   = None

    def _get_screen_offset(self) -> tuple[int, int]:
        """Virtual desktop top-left offset (negative on left-extended multi-monitor)."""
        if platform.system() == "Windows":
            left = ctypes.windll.user32.GetSystemMetrics(76)  # SM_XVIRTUALSCREEN
            top  = ctypes.windll.user32.GetSystemMetrics(77)  # SM_YVIRTUALSCREEN
            return (left, top)
        return (0, 0)

    # ------------------------------------------------------------------ #
    #  Feature 3: JSONL history persistence                               #
    # ------------------------------------------------------------------ #

    def _log_record(self, rec: ActionRecord) -> None:
        """Append record to in-memory list and (if configured) to JSONL file."""
        self.action_history.append(rec)
        if self.history_log_path:
            try:
                row = dataclasses.asdict(rec)
                # tuple coords not JSON-serialisable by default
                if row.get("coordinates") is not None:
                    row["coordinates"] = list(row["coordinates"])
                with open(self.history_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.warning(f"JSONL write failed: {e}")

    # ------------------------------------------------------------------ #
    #  Feature 2: DONE verification                                       #
    # ------------------------------------------------------------------ #

    def verify_completion(self, success_criteria: str) -> tuple[bool, str]:
        """Capture a fresh screenshot and ask LLM if success_criteria is met.

        Returns:
            (verified, evidence) where evidence is a one-sentence explanation.
        """
        try:
            img = self.screen_capture.capture_screen()
            b64 = self.screen_capture.image_to_base64(img, quality=60)
            messages = [
                {"role": "system", "content": VERIFY_PROMPT},
                {"role": "user", "content": [
                    {"type": "text",
                     "text": f"Success criteria: {success_criteria}"},
                    {"type": "image_url",
                     "image_url": {"url": b64}},
                ]},
            ]
            raw = self.llm_client.create_chat_completion(
                messages=messages,
                max_tokens=200,
                temperature=0.0,
                max_retries=2,
            )
            data = json.loads(raw)
            verified = bool(data.get("verified", False))
            evidence = data.get("evidence", "(no evidence)")
            return verified, evidence
        except Exception as e:
            logger.warning(f"verify_completion failed: {e} — assuming success")
            return True, "(verification unavailable)"
    # ------------------------------------------------------------------ #
    #  Visual stability polling                                           #
    # ------------------------------------------------------------------ #

    def wait_for_visual_stability(
        self,
        before_hash:  str,
        before_image: Image.Image,
        max_wait:     float = 10.0,
        poll_interval: float = 0.8,
        stability_threshold: int = 2,
        change_threshold: float = 0.98,
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

        # P1-1: Adaptive intervals — fast first checks, slower later
        adaptive_intervals = [0.2, 0.3, 0.5]
        poll_idx = 0

        while elapsed < max_wait:
            if poll_idx < len(adaptive_intervals):
                interval = adaptive_intervals[poll_idx]
                poll_idx += 1
            else:
                interval = poll_interval
            time.sleep(interval)
            elapsed += interval

            cur_img  = self.screen_capture.capture_screen()
            cur_hash = self.screen_capture.get_image_hash(cur_img)

            # ── Has the action had any visible effect? ──────────────
            if not changed:
                # Hash fast-check: catches micro changes (typed chars, caret
                # movement) that are invisible to SSIM — skips SSIM computation
                # entirely when the hash already differs.
                if before_hash and cur_hash != before_hash:
                    changed = True
                    similarity_start = self.screen_capture.get_screen_similarity(
                        before_image, cur_img
                    )
                else:
                    similarity_start = self.screen_capture.get_screen_similarity(
                        before_image, cur_img
                    )
                    if similarity_start < change_threshold:
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
        if at == ActionType.SCROLL:
            return f"SCROLL({p.scroll_amount})"
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

        # P1-3: Milestones + recent actions for long-task context
        history = "None"
        if self.action_history:
            parts = []
            if self.milestone_summaries:
                ms = " → ".join(self.milestone_summaries[-8:])
                parts.append(f"  [Milestones]: {ms}")
            lines = []
            for i, rec in enumerate(self.action_history[-5:]):
                line = f"  Step {i+1}: {rec.action_type}"
                if rec.coordinates:
                    line += f" ({rec.coordinates[0]:.2f}, {rec.coordinates[1]:.2f})"
                line += f"\n    Reason: {rec.thought[:70]}"
                result = rec.status_tag or ("\u2705 Screen changed" if rec.screen_changed else "\u274c No change")
                if rec.similarity_score < 1.0:
                    result += f" (SSIM={rec.similarity_score:.3f}, {rec.wait_elapsed:.1f}s)"
                line += f"\n    Result: {result}"
                lines.append(line)
            parts.extend(lines)
            history = "\n".join(parts)

        extras = ""
        blocked = [k for k, v in self.failed_action_counts.items() if v >= 3]
        if blocked:
            bl = "\n".join(f"  - {a} (failed {self.failed_action_counts[a]}x)" for a in blocked[-5:])
            extras += f"\n[⚠️ BLACKLISTED — do NOT repeat]:\n{bl}\n(Switch to a completely different approach!)\n"
        # Also warn about actions that failed 1-2 times (caution, not blocked)
        caution = [k for k, v in self.failed_action_counts.items() if 1 <= v < 3]
        if caution:
            cl = ", ".join(caution[-3:])
            extras += f"\n[⚠️ CAUTION — these actions failed recently]: {cl}"
            extras += (
                "\n[⚠️ CONFIDENCE RULE]: For actions that failed before, lower confidence:\n"
                "  - Failed once → cap confidence at 0.70\n"
                "  - Failed twice → cap at 0.50 AND switch to FOCUS or HOTKEY instead of CLICK"
            )
        if self.last_parse_error:
            extras += f"\n\n[❗JSON ERROR]: {self.last_parse_error}\nReturn ONLY pure JSON, no markdown."
        # Fix 4: per-action-type threshold using last known action type
        _last_at = self.action_history[-1].action_type if self.action_history else None
        if self.consecutive_no_change >= _no_change_threshold(_last_at):
            extras += (
                f"\n\n[⚠️ CRITICAL] {self.consecutive_no_change} consecutive actions had NO visible effect! "
                "You MUST try a completely different approach (SCROLL, ESC, HOTKEY, WAIT)."
            )

        global_context = ""
        if getattr(self, "_global_goal", ""):
            global_context = (
                f"\n[Global Goal]: {self._global_goal}"
                f"\n[Current Step]: {self._current_step}"
                f"\n[Success Criteria]: {self._success_criteria}"
            )
            if getattr(self, "_prior_state", ""):
                global_context += f"\n[Prior Step Result]: {self._prior_state}"

        focus_hint = ""
        if getattr(self, "_focus_mode", False):
            focus_hint = (
                "\n\n[⚠️ FOCUS MODE ACTIVE]: You are viewing a ZOOMED-IN region "
                "(cyan grid, orange labels). Coordinates are LOCAL to this crop (0~1). "
                "Output a precise CLICK. Do NOT output FOCUS again."
            )

        user_msg = (
            f"Task: {self.task}{global_context}{focus_hint}\n"
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
            scroll_amount=_safe_int(pd.get("scroll_amount")),
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

    def run(self, max_duration_seconds: Optional[float] = None) -> bool:
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
        if max_duration_seconds:
            logger.info(f"Timeout: {max_duration_seconds}s")
        logger.info("=" * 56)

        _start_time = time.time()

        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"\n--- Iteration {iteration}/{self.max_iterations} ---")

            # Global wall-clock timeout check
            if max_duration_seconds and (time.time() - _start_time) > max_duration_seconds:
                logger.warning(f"Task timeout ({max_duration_seconds}s) reached at iteration {iteration}")
                return False
            try:
                # ── Perceive ──────────────────────────────────────────────
                # Always capture a full-screen baseline for SSIM, even in
                # FOCUS mode (Bug 3 fix: real stability check after action).
                raw_before   = self.screen_capture.capture_screen()
                before_hash  = self.screen_capture.get_image_hash(raw_before)

                if self._focus_mode and self._focus_cx is not None:
                    before_image_b64, actual_w, actual_h = \
                        self.screen_capture.capture_focus_region(
                            self._focus_cx, self._focus_cy,
                            self._focus_w_px, self._focus_h_px,
                        )
                    self._focus_w_px = actual_w
                    self._focus_h_px = actual_h
                    logger.info(f"[FOCUS] Captured {actual_w}×{actual_h}px region")
                else:
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
                    display          = self.screen_capture.compress_image(display)
                    before_image_b64 = self.screen_capture.image_to_base64(display)

                # ── Think ──────────────────────────────────────────────────
                logger.info(f"Calling {self.llm_client.model_name}…")
                raw_resp = self.llm_client.create_chat_completion(
                    messages=self._build_prompt(before_image_b64, resolution),
                    response_format={"type": "json_object"},
                )
                response = self._parse_response(raw_resp)
                self.last_parse_error = None

                if response.action_type == ActionType.FOCUS:
                    self._enable_focus_mode(response.action_params.x, response.action_params.y)
                    self._log_record(ActionRecord(
                        iteration=iteration, action_type="FOCUS",
                        thought=response.thought,
                        status_tag="🔍Focus",
                    ))  # Fix 3: log FOCUS to history for debuggability
                    continue

                if self._focus_mode:
                    if response.action_type in (
                        ActionType.CLICK, ActionType.DOUBLE_CLICK,
                        ActionType.RIGHT_CLICK, ActionType.HOVER,
                    ):
                        # Translate local FOCUS coords → global screen coords
                        gx, gy = focus_to_global(
                            response.action_params.x, response.action_params.y,
                            self._focus_cx, self._focus_cy,
                            self._focus_w_px, self._focus_h_px,
                            resolution[0], resolution[1],
                        )
                        logger.info(f"[FOCUS] local({response.action_params.x:.3f},{response.action_params.y:.3f}) → global({gx:.3f},{gy:.3f})")
                        response.action_params.x = gx
                        response.action_params.y = gy
                    # Bug 2 fix: ANY action exits FOCUS mode (not just coordinate actions)
                    self._disable_focus_mode()
                    logger.info("[FOCUS] Disabled")

                # ── Blacklist check (counter-based: block after ≥3 failures) ──
                action_str = self._action_to_str(response)
                if self.failed_action_counts.get(action_str, 0) >= 3:
                    logger.warning(f"⛔ Blocking blacklisted action: {action_str} (failed {self.failed_action_counts[action_str]}x)")
                    self._log_record(ActionRecord(
                        iteration=iteration, action_type=action_str,
                        thought="Blocked (known failure)", status_tag="⛔Blocked",
                    ))
                    self.consecutive_no_change += 1
                    time.sleep(0.5)
                    continue

                logger.info(f"[Action] {response.action_type.value} (conf={response.confidence:.2f})")

                # ── Bug 1 fix: DONE early return ───────────────────────────
                if response.action_type == ActionType.DONE:
                    self.action_executor.execute(response)
                    self._log_record(ActionRecord(
                        iteration=iteration,
                        action_type=response.action_type.value,
                        thought=response.thought,
                        observation=response.observation,
                        screen_changed=False,
                        similarity_score=1.0,
                        status_tag="✅Done",
                    ))
                    # P0-2: Clean state on DONE
                    self.failed_action_counts.clear()
                    self.consecutive_no_change = 0
                    logger.info("Task completed ✓")
                    return True

                # ── Act ────────────────────────────────────────────────────
                self.action_executor.execute(response)

                # ── Observe ────────────────────────────────────────────────
                # Bug 3 fix: raw_before is always captured now, even in FOCUS mode
                # TYPE/WAIT use a tighter threshold since they produce tiny pixel diffs
                thresh = 0.995 if response.action_type in (
                    ActionType.TYPE, ActionType.WAIT
                ) else 0.98
                changed, _, elapsed, similarity = self.wait_for_visual_stability(
                    before_hash, raw_before, change_threshold=thresh
                )

                if (not changed
                        and response.action_type == ActionType.CLICK
                        and not self._focus_mode
                        and response.confidence < 0.7):
                    # P2: Only auto-FOCUS when confidence is low (target uncertain)
                    logger.warning(f"CLICK had no effect (conf={response.confidence:.2f}) — auto-triggering FOCUS")
                    self._enable_focus_mode(response.action_params.x, response.action_params.y)
                    continue

                # ── Update state ──────────────────────────────────────
                if response.action_type == ActionType.WAIT:
                    status_tag = "⏳Wait"
                    self.consecutive_no_change = 0
                elif changed:
                    self.consecutive_no_change = 0
                    p = response.action_params
                    if p.x is not None:
                        # Clear nearby coordinate-based failures
                        self.failed_action_counts = {
                            k: v for k, v in self.failed_action_counts.items()
                            if not self._near(k, p.x, p.y)
                        }
                    else:
                        self.failed_action_counts.clear()
                    # Page changed → SCROLL might work again on new content
                    self.failed_action_counts = {
                        k: v for k, v in self.failed_action_counts.items()
                        if not k.startswith("SCROLL")
                    }
                    status_tag = "✅Changed"
                    logger.info(f"[{status_tag}] SSIM={similarity:.4f}, {elapsed:.1f}s")
                    # P1-3: Record milestone for long-task context
                    milestone = f"{response.action_type.value}"
                    if response.observation:
                        milestone += f": {response.observation[:50]}"
                    elif response.thought:
                        milestone += f": {response.thought[:50]}"
                    self.milestone_summaries.append(milestone)
                    # Fix 4: prevent unbounded growth in long tasks
                    if len(self.milestone_summaries) > 30:
                        self.milestone_summaries = self.milestone_summaries[-20:]
                else:
                    self.consecutive_no_change += 1
                    self.failed_action_counts[action_str] = self.failed_action_counts.get(action_str, 0) + 1
                    count = self.failed_action_counts[action_str]
                    status_tag = f"❌NoChange[{action_str}]({count}x)"
                    logger.warning(f"[{status_tag}] SSIM={similarity:.4f}, {elapsed:.1f}s")

                p = response.action_params
                coords = (p.x, p.y) if p.x is not None else None
                self._log_record(ActionRecord(
                    iteration=iteration,
                    action_type=response.action_type.value,
                    action_params={k: v for k, v in {
                        "x": p.x, "y": p.y, "text": p.text,
                        "keys": p.keys, "scroll_amount": p.scroll_amount,
                    }.items() if v is not None},
                    thought=response.thought,
                    observation=response.observation,
                    coordinates=coords,
                    screen_changed=changed,
                    similarity_score=similarity,
                    wait_elapsed=elapsed,
                    status_tag=status_tag,
                ))

            except (ValueError, json.JSONDecodeError) as e:
                msg = str(e)[:150]
                logger.error(f"Parse error: {e}")
                self.last_parse_error = msg
                self._log_record(ActionRecord(
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

        # Fix 1: Enable history logging for standalone gui_agent.py execution
        os.makedirs("logs", exist_ok=True)
        log_path = os.path.join("logs", f"session_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
        agent = GUIAgent(config=config, history_log_path=log_path)
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