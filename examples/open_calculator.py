"""
GUI Agent Example - Open Calculator

This example demonstrates how to use GUI Agent to open the Windows Calculator.

Usage:
    python examples/open_calculator.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import gui_agent
sys.path.insert(0, str(Path(__file__).parent.parent))

from gui_agent import GUIAgent, Config


def main():
    """Open Windows Calculator using GUI Agent."""

    # Load configuration
    config = Config()

    # Create agent instance
    agent = GUIAgent(config=config)

    # Task: Open Calculator
    instruction = "打开 Windows 计算器应用程序"

    print(f"Task: {instruction}")
    print("-" * 40)

    # Run the agent
    result = agent.run(instruction)

    print("-" * 40)
    print(f"Result: {result}")

    return result


if __name__ == "__main__":
    main()
