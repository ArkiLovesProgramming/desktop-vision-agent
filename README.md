# GUI Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)

A desktop automation agent powered by Alibaba Cloud's Qwen-VL multimodal vision-language models. GUI Agent implements an intelligent "See-Think-Act" interaction loop to automate computer tasks through visual understanding and precise control.

## Features

- **Visual Perception**: "Sees" the current screen state through screenshots, supports 10x10 grid coordinate positioning
- **Intelligent Decision Making**: Uses Qwen-VL model for reasoning and decision making
- **Precise Control**: Operates mouse and keyboard through PyAutoGUI
- **ReAct Loop**: Continuously executes "Perception-Thinking-Action" cycles until task completion
- **Relative Coordinates**: Uses 0.0-1.0 relative coordinates for cross-resolution compatibility
- **Multi-Monitor Support**: Automatically handles multi-screen offset calculations
- **DPI Aware**: Correctly handles logical vs physical resolution
- **Safety Mechanisms**: PyAutoGUI failsafe for emergency stop

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           GUI Agent Architecture                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │  Perception  │───▶│    Brain     │───▶│    Action    │              │
│  │              │    │              │    │              │              │
│  │  ScreenCap   │    │  QwenClient  │    │  Executor    │              │
│  │  + Grid      │    │  + Prompt    │    │  + PyAutoGUI │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│         │                   │                   │                       │
│         │                   │                   │                       │
│         ▼                   ▼                   ▼                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │  Screenshot  │    │  JSON Parse  │    │ CLICK/TYPE   │              │
│  │  Base64 Enc  │    │  Validation  │    │ SCROLL/HOTKEY│              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                          ┌──────────────────┐
                          │   ReAct Loop     │
                          │  until DONE      │
                          └──────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and add your configuration:

```bash
cp .env.example .env
```

Edit the `.env` file:

```ini
DASHSCOPE_API_KEY=sk-your-api-key-here
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_NAME=qwen3-vl-plus
```

**How to Get API Key:**
1. Log in to [Alibaba Cloud Bailian Console](https://bailian.console.aliyun.com/)
2. Go to API-KEY Management
3. Create a new API Key

### 3. Run the Agent

```bash
python gui_agent.py
```

Then enter your task instructions, such as:
- "Open Notepad"
- "Search for Python in the browser"
- "Minimize all windows"

## Usage Examples

### Basic Examples

```python
# Open Calculator
"Open Windows Calculator"

# Web Search
"Search for Python tutorials on Google"

# Text Input
"Type 'Hello, World!' in Notepad"

# Window Control
"Minimize all windows"
```

### Advanced Examples

```python
# Multi-step Tasks
"Open browser, visit GitHub, and log into my account"

# File Operations
"Open File Explorer and create a new folder"

# System Control
"Adjust system volume to 50%"

# Form Filling
"Enter name, email, and message in the form, then click submit"
```

### Programming Assistant Examples

```python
# Code Execution
"Open VS Code and create a new Python file"

# Debugging Assistance
"Run the current Python script and view the output"
```

## Supported Actions

| Action | Description | Parameters | Example |
|--------|-------------|------------|---------|
| `CLICK` | Click at specified coordinates | x, y (0.0-1.0 relative) | `CLICK(0.5, 0.5)` - Click screen center |
| `TYPE` | Type text via keyboard | text (string to input) | `TYPE("Hello World")` |
| `SCROLL` | Scroll mouse wheel | scroll_amount (negative=down, positive=up) | `SCROLL(-100)` - Scroll down |
| `HOTKEY` | Keyboard shortcuts | keys (shortcut combination) | `HOTKEY("ctrl+c")` - Copy |
| `DONE` | Task completed | none | `DONE` - End task |

## Configuration Options

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DASHSCOPE_API_KEY` | Alibaba Cloud API Key | - | Yes |
| `BASE_URL` | API endpoint | `https://dashscope.aliyuncs.com/compatible-mode/v1` | No |
| `MODEL_NAME` | Model name | `qwen3-vl-plus` | No |
| `MAX_ITERATIONS` | Maximum iterations per task | `50` | No |
| `TEMPERATURE` | Model temperature | `0.7` | No |

### Available Models

| Model | Characteristics | Use Cases |
|-------|-----------------|-----------|
| `qwen3-vl-plus` | Balanced performance and cost | Daily use, recommended |
| `qwen3-vl-flash` | Fast response | Scenarios requiring quick feedback |
| `qwen-vl-max` | Highest performance | Complex tasks, high precision requirements |

## Project Structure

```
GUI-agent/
├── gui_agent.py          # Main program entry point
├── main.py               # Multi-step planner + executor orchestration
├── planner.py            # Task decomposition module
├── cli.py                # CLI interaction module (beautiful terminal output)
├── config.py             # Configuration management
├── requirements.txt      # Python dependencies
├── LICENSE               # MIT License
├── CONTRIBUTING.md       # Contribution guidelines
├── .env.example          # Configuration template
├── .gitignore            # Git ignore rules
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── pull_request_template.md
├── examples/             # Example scripts
└── test-archive/         # Test suite
    ├── test_agent.py
    ├── test_coordinate.py
    ├── test_integration.py
    └── TEST_REPORT.md
```

## Testing

Run the test suite:

```bash
python -m pytest test-archive/
```

View test report:

```bash
cat test-archive/TEST_REPORT.md
```

### Test Results

Latest test date: 2026-02-23

- Coordinate transformation tests: 5/5 passed
- Integration tests: 4/4 passed
- Optimization feature tests: All passed

## Logging Output

The program outputs detailed execution logs:

```
2026-02-23 10:00:00 - INFO - GUI Agent starting
2026-02-23 10:00:00 - INFO - Task objective: Open Notepad
2026-02-23 10:00:01 - INFO - Capturing screen...
2026-02-23 10:00:01 - INFO - Screen resolution: 1920x1080
2026-02-23 10:00:01 - INFO - Calling Qwen-VL model...
2026-02-23 10:00:05 - INFO - [Thinking] I see the current desktop with taskbar at the bottom. To open Notepad, need to click Start menu...
2026-02-23 10:00:05 - INFO - [Action] CLICK
2026-02-23 10:00:05 - INFO - Executing action...
2026-02-23 10:00:05 - INFO - Click coordinates: (100, 1050)
```

## Safety Notice

**PyAutoGUI Built-in FailSafe:** Move the mouse quickly to any of the four screen corners to emergency stop the program.

> **Important Reminders:**
> - Always monitor the Agent's operations
> - Do not run on sensitive data or critical systems
> - Test in a safe environment first
> - Be ready to press Ctrl+C to interrupt

## Limitations and Considerations

1. **Coordinate Positioning Accuracy**: Model estimates coordinates based on vision, may have deviations
2. **Iteration Limit**: Default maximum 15 iterations to prevent runaway execution
3. **Base64 Images**: Uses Data URI format for images; consider temporary file upload if issues occur
4. **Language Support**: Model supports both Chinese and English instructions well
5. **Screen Resolution**: Supports multiple resolutions using relative coordinates for compatibility

## Troubleshooting

### API Connection Failure
- Check if API Key in `.env` file is correct
- Verify network connection is working
- Check Alibaba Cloud account balance

### JSON Parsing Failure
- Model may return non-standard JSON format
- Code includes multiple parsing strategies, but extreme cases may still fail

### Action Execution Inaccuracy
- Prompt emphasizes screen resolution information
- If issues persist, try more detailed task descriptions

### Slow Model Response
- Check network connection
- Try using `qwen3-vl-flash` model for faster response
- Reduce screenshot quality configuration

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code style guidelines
- PR process
- Testing requirements
- Issue and PR templates

### Contributors

<!-- You can add contributors list here -->

## Development Notes

### Adding New Action Types

1. Add new type to `ActionType` enum
2. Add corresponding execution method in `ActionExecutor`
3. Update `SYSTEM_PROMPT` to document the new action

### Modifying System Prompt

The `SYSTEM_PROMPT` variable defines the agent's role and output format requirements, and can be adjusted based on actual needs.

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/gui-agent.git
cd gui-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest test-archive/
```

## Related Links

- [Alibaba Cloud Bailian Console](https://bailian.console.aliyun.com/)
- [Qwen-VL Documentation](https://help.aliyun.com/zh/dashscope/)
- [PyAutoGUI Documentation](https://pyautogui.readthedocs.io/)
- [GitHub Repository](https://github.com/YOUR_USERNAME/gui-agent)

## License

MIT License - See [LICENSE](LICENSE) file for details

---

<div align="center">
  <p>Made with ❤️ using Qwen-VL Vision-Language Models</p>
</div>
