"""
GUI Agent Example - Web Search

This example demonstrates how to use GUI Agent to perform a web search.

Usage:
    python examples/web_search.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import gui_agent
sys.path.insert(0, str(Path(__file__).parent.parent))

from gui_agent import GUIAgent, Config


def main():
    """Search Python tutorials on Google using GUI Agent."""

    # Load configuration
    config = Config()

    # Create agent instance
    agent = GUIAgent(config=config)

    # Task: Search for Python tutorials
    instruction = "Open browser and search for 'Python tutorials' on Google"

    print(f"Task: {instruction}")
    print("-" * 40)

    # Run the agent
    result = agent.run(instruction)

    print("-" * 40)
    print(f"Result: {result}")

    return result


if __name__ == "__main__":
    main()
