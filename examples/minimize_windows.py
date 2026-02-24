"""
GUI Agent Example - Minimize All Windows

This example demonstrates how to use GUI Agent to minimize all windows.

Usage:
    python examples/minimize_windows.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import gui_agent
sys.path.insert(0, str(Path(__file__).parent.parent))

from gui_agent import GUIAgent, Config


def main():
    """Minimize all windows using GUI Agent."""

    # Load configuration
    config = Config()

    # Create agent instance
    agent = GUIAgent(config=config)

    # Task: Minimize all windows
    instruction = "Minimize all windows and show the desktop"

    print(f"Task: {instruction}")
    print("-" * 40)

    # Run the agent
    result = agent.run(instruction)

    print("-" * 40)
    print(f"Result: {result}")

    return result


if __name__ == "__main__":
    main()
