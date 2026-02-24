# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GUI Agent is a Python-based desktop automation agent powered by Alibaba Cloud's Qwen-VL multimodal vision-language models. It implements a ReAct (Reasoning + Acting) loop to autonomously execute natural language instructions using mouse/keyboard control.

## Quick Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the agent
python gui_agent.py        # Direct agent mode (single-step tasks)
python main.py             # Multi-step planner + executor mode

# Run tests
python -m pytest test-archive/
```

## Architecture

### Three-Layer Architecture

1. **`main.py`** - Orchestrator: Manages Planner + Agent coordination, handles user input loop
2. **`planner.py`** - Task decomposer: Converts high-level goals into atomic steps using LLM
3. **`gui_agent.py`** - Executor: ReAct loop with screen capture, LLM reasoning, action execution

### Key Modules

- **`config.py`** - Configuration management via dataclass, loads from `.env` or CLI args
- **`cli.py`** - Terminal UI using Rich (colored output, progress spinners, tables)

### Core Design Decisions

- **DPI-aware screen capture** (Windows): Initialized before pyautogui to prevent deadlock
- **GDI BitBlt** for fast screen capture on Windows (falls back to PIL ImageGrab)
- **Relative coordinates** (0.0-1.0) for cross-resolution compatibility
- **10x10 grid system** with visual overlay for LLM coordinate reasoning
- **FOCUS mode**: Zoom-in mechanism for precise clicks on small UI elements
- **Multi-step replanning**: Automatic recovery when steps fail (max 3 replans)

## Configuration

Environment variables (`.env` file, copied from `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `DASHSCOPE_API_KEY` | Alibaba Cloud API Key (required) | - |
| `BASE_URL` | API endpoint | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `MODEL_NAME` | Model to use | `qwen3-vl-plus` |
| `MAX_ITERATIONS` | Max iterations per task | 50 |

Available models: `qwen3-vl-plus` (recommended), `qwen3-vl-flash` (fast), `qwen-vl-max` (premium)

## Supported Actions

| Action | Parameters | Example |
|--------|------------|---------|
| `CLICK` | x, y (0.0-1.0 relative) | `CLICK(0.5, 0.5)` - center screen |
| `DOUBLE_CLICK` | x, y | `DOUBLE_CLICK(0.3, 0.7)` |
| `RIGHT_CLICK` | x, y | `RIGHT_CLICK(0.5, 0.5)` |
| `HOVER` | x, y | `HOVER(0.2, 0.3)` - for dropdowns |
| `DRAG` | x, y, end_x, end_y | `DRAG(0.1, 0.1, 0.9, 0.9)` |
| `TYPE` | text | `TYPE("Hello")` - uses clipboard paste |
| `SCROLL` | scroll_amount, x, y | `SCROLL(-100)` - scroll down |
| `HOTKEY` | keys | `HOTKEY("ctrl l")` - focus address bar |
| `WAIT` | wait_seconds | `WAIT(2.0)` |
| `FOCUS` | x, y | Enables zoom mode for precision |
| `DONE` | - | Task complete |

## Code Style

- **PEP 8** compliant, 4 spaces, max 100 char lines
- **Type hints** required for all function signatures
- **Google-style docstrings** with Args/Returns sections
- Commit messages: `feat(scope): description` (conventional commits)

## Key Implementation Details

### Screen Capture (`gui_agent.py:127-350`)
- Uses Win32 GDI BitBlt for fastest capture on Windows
- DPI awareness must be set BEFORE any pyautogui/PIL imports
- Falls back to PIL ImageGrab if GDI fails

### Action Executor (`gui_agent.py:398-630`)
- TYPE action uses clipboard paste (pyperclip) for CJK/Unicode support
- Clipboard verification and retry logic for reliability
- Focus guard: aborts paste if foreground window changed

### Prompt Architecture (`gui_agent.py:636-812`)
- CORE_PROMPT defines agent behavior, action reference, anti-failure strategies
- DOMAIN_KNOWLEDGE provides app-specific shortcuts (Chrome, Excel, etc.)
- FOCUS mode uses cyan grid with local coordinates (0-1 within crop)

### Verification (`gui_agent.py:1039-1069`)
- `verify_completion()` captures screenshot and asks LLM if success criteria met
- Used after each step in multi-step tasks

## Safety

- PyAutoGUI failsafe: move mouse to screen corner to emergency stop
- Always monitor agent operation
- Ready to interrupt with Ctrl+C

## Testing

Run test suite:
```bash
python -m pytest test-archive/
```

Tests cover: coordinate transformation, integration scenarios, optimization features.
See `test-archive/TEST_REPORT.md` for latest results.
