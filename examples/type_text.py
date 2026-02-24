"""
GUI Agent Example - Type Text

This example demonstrates how to use GUI Agent to type text in an active window.

Usage:
    python examples/type_text.py

Note:
    Make sure to have a text input window active (like Notepad) before running.
"""

import sys
from pathlib import Path

# Add parent directory to path to import gui_agent
sys.path.insert(0, str(Path(__file__).parent.parent))

from gui_agent import GUIAgent, Config


def main():
    """Type 'Hello, World!' in the active window using GUI Agent."""

    # Load configuration
    config = Config()

    # Create agent instance
    agent = GUIAgent(config=config)

    # Task: Type text
    instruction = "在记事本中输入'Hello, World! 这是一个测试。'"

    print(f"Task: {instruction}")
    print("-" * 40)
    print("Note: Please make sure Notepad or a text editor is open and active!")
    print("-" * 40)

    # Run the agent
    result = agent.run(instruction)

    print("-" * 40)
    print(f"Result: {result}")

    return result


if __name__ == "__main__":
    main()
