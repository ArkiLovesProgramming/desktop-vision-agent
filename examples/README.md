# GUI Agent Examples

This directory contains example scripts demonstrating how to use GUI Agent for common tasks.

## Examples

### 1. Open Calculator

Opens the Windows Calculator application.

```bash
python examples/open_calculator.py
```

### 2. Web Search

Performs a Google search for "Python tutorials".

```bash
python examples/web_search.py
```

### 3. Type Text

Types text in the active window (make sure to have a text editor open).

```bash
python examples/type_text.py
```

### 4. Minimize All Windows

Minimizes all windows to show the desktop.

```bash
python examples/minimize_windows.py
```

## Running Examples

1. Make sure you have configured your `.env` file with your API key
2. Install all dependencies: `pip install -r requirements.txt`
3. Run any example: `python examples/<script_name>.py`

## Creating Your Own Examples

```python
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gui_agent import GUIAgent, Config

def main():
    # Load configuration
    config = Config()

    # Create agent instance
    agent = GUIAgent(config=config)

    # Your task instruction
    instruction = "Your task here"

    # Run the agent
    result = agent.run(instruction)

    return result

if __name__ == "__main__":
    main()
```

## Tips

- Always monitor the agent's actions
- Be ready to interrupt with Ctrl+C if needed
- Test in a safe environment first
- Use clear and specific instructions
