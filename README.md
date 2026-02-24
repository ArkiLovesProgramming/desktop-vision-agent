# GUI Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A desktop automation agent powered by multimodal vision-language models. Automates computer tasks through visual understanding and precise control.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
cp .env.example .env
```

Edit `.env`:
```ini
DASHSCOPE_API_KEY=sk-your-api-key-here    # Alibaba Cloud DashScope
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_NAME=qwen3-vl-plus
```

**Supported Models:** GUI Agent works with any vision-language model that follows the OpenAI API format.

| Provider | Models |
|----------|--------|
| Alibaba Cloud (DashScope) | `qwen3-vl-plus` (recommended), `qwen3-vl-flash`, `qwen-vl-max` |
| OpenAI | `gpt-4o`, `gpt-4o-mini` |
| Google | `gemini-1.5-pro`, `gemini-1.5-flash` |
| Others | Any model compatible with OpenAI API format |

**Get API Keys:**
- Alibaba Cloud: [Bailian Console](https://bailian.console.aliyun.com/)
- OpenAI: [Platform Console](https://platform.openai.com/)
- Google: [AI Studio](https://aistudio.google.com/)

### 3. Run

```bash
python gui_agent.py
```

Then enter instructions like:
- "Open Notepad"
- "Search for Python tutorials on Google"
- "Minimize all windows"

## Demo

### Basic Task: Open Calculator

<!-- Replace with your video: <video src="path/to/calc.mp4" controls></video> -->
![Open Calculator Demo](path/to/calc.gif)

### Multi-step Task: Web Search

<!-- Replace with your video: <video src="path/to/search.mp4" controls></video> -->
![Web Search Demo](path/to/search.gif)

### Advanced Task: Form Filling

<!-- Replace with your video: <video src="path/to/form.mp4" controls></video> -->
![Form Filling Demo](path/to/form.gif)

## Usage Examples

```python
"Open Windows Calculator"
"Search for Python tutorials on Google"
"Type 'Hello, World!' in Notepad"
"Open browser, visit GitHub, and log in"
```

## Supported Actions

| Action | Description | Example |
|--------|-------------|---------|
| `CLICK` | Click at coordinates (0.0-1.0) | `CLICK(0.5, 0.5)` |
| `TYPE` | Type text via clipboard paste | `TYPE("Hello")` |
| `SCROLL` | Scroll mouse wheel | `SCROLL(-100)` |
| `HOTKEY` | Keyboard shortcuts | `HOTKEY("ctrl+c")` |
| `FOCUS` | Zoom-in for precision | `FOCUS(0.5, 0.5)` |
| `DONE` | Task completed | `DONE` |

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `DASHSCOPE_API_KEY` | API Key (DashScope/OpenAI/etc.) | - |
| `BASE_URL` | API endpoint | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `MODEL_NAME` | Model to use | `qwen3-vl-plus` |
| `MAX_ITERATIONS` | Max iterations per task | `50` |

**Recommended Models:** `qwen3-vl-plus` (best for GUI tasks), `gpt-4o` (general purpose), `qwen3-vl-flash` (fast)

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Perception  │───▶│   Brain     │───▶│   Action    │
│ ScreenCap   │    │  LLM Client │    │ Executor    │
└─────────────┘    └─────────────┘    └─────────────┘
```

Files:
- `gui_agent.py` - Main agent with ReAct loop
- `main.py` - Multi-step planner orchestrator
- `planner.py` - Task decomposition

## Safety

**PyAutoGUI FailSafe:** Move mouse to screen corner to emergency stop.

> - Always monitor agent operations
> - Test in safe environment first
> - Ready to press Ctrl+C to interrupt

## Troubleshooting

**API connection fails:** Check API key and network connection.

**Inaccurate actions:** Use more detailed task descriptions.

**Slow response:** Try a faster model like `qwen3-vl-flash` or `gpt-4o-mini`.

## License

MIT License - See [LICENSE](LICENSE) file for details
