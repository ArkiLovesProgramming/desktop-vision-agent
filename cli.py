"""
GUI Agent - Command Line Interface Module

Provides a professional, colorful terminal interface with:
- Colored output for different log levels
- Progress bars and spinners
- Interactive tables for history
- Rich user input experience
"""

from datetime import datetime
from typing import Optional, List, Dict, Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TaskID
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich.layout import Layout


# Initialize console with emoji support
console = Console()


# ============================================================================
# Logging Functions
# ============================================================================

def log_info(message: str, title: str = "INFO") -> None:
    """Log an informational message with blue styling."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    console.print(f"[dim]{timestamp}[/dim] [bold blue][{title}][/bold blue] {message}")


def log_success(message: str, title: str = "SUCCESS") -> None:
    """Log a success message with green styling."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    console.print(f"[dim]{timestamp}[/dim] [bold green][{title}][/bold green] {message}")


def log_warning(message: str, title: str = "WARNING") -> None:
    """Log a warning message with yellow styling."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    console.print(f"[dim]{timestamp}[/dim] [bold yellow][{title}][/bold yellow] {message}")


def log_error(message: str, title: str = "ERROR") -> None:
    """Log an error message with red styling."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    console.print(f"[dim]{timestamp}[/dim] [bold red][{title}][/bold red] {message}")


def log_debug(message: str, title: str = "DEBUG") -> None:
    """Log a debug message with dim styling."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    console.print(f"[dim]{timestamp}[/dim] [dim][{title}][/dim] {message}")


# ============================================================================
# Panel Displays
# ============================================================================

def show_welcome_panel() -> None:
    """Display the welcome panel with project information."""
    welcome_text = """[bold blue]GUI Agent[/bold blue] - Desktop Automation powered by Qwen-VL

[dim]Turn natural language instructions into mouse and keyboard actions[/dim]

[bold]Quick Commands:[/bold]
  [green]help[/green]     - Show this help message
  [green]history[/green] - View action history
  [green]quit/exit[/green] - Exit the application

[bold]Example Instructions:[/bold]
  - "Open Calculator"
  - "Search for Python tutorials on Google"
  - "Type 'Hello World' in the active window"
  - "Scroll down the page"
"""
    console.print(Panel(
        welcome_text,
        title="[bold blue]Welcome to GUI Agent[/bold blue]",
        border_style="blue",
        padding=(1, 2),
    ))
    console.print()


def show_safety_warning() -> None:
    """Display important safety warnings."""
    warning_text = """[yellow]SAFETY REMINDER:[/yellow]

[bold]Always monitor the agent's actions![/bold]

- The agent can control your mouse and keyboard
- Stay ready to interrupt with Ctrl+C if needed
- Don't run on sensitive data or critical systems
- Test in a safe environment first

[dim]PyAutoGUI failsafe: Move mouse to screen corner to stop[/dim]
"""
    console.print(Panel(
        warning_text,
        title="[bold yellow]Safety Notice[/bold yellow]",
        border_style="yellow",
        padding=(1, 2),
    ))
    console.print()


def show_config_panel(config: Dict[str, Any]) -> None:
    """Display current configuration."""
    config_text = f"""[bold]Model:[/bold] {config.get('model', 'N/A')}
[bold]Base URL:[/bold] {config.get('base_url', 'N/A')}
[bold]Max Iterations:[/bold] {config.get('max_iterations', 50)}
[bold]Screen Capture:[/bold] Grid overlay enabled
"""
    console.print(Panel(
        config_text,
        title="[bold]Current Configuration[/bold]",
        border_style="green",
        padding=(0, 1),
    ))


# ============================================================================
# Progress Indicators
# ============================================================================

class TaskProgress:
    """Context manager for displaying task progress with spinner."""

    def __init__(self, description: str = "Processing"):
        self.description = description
        self.progress: Optional[Progress] = None
        self.task_id: Optional[TaskID] = None

    def __enter__(self):
        self.progress = Progress(
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            console=console,
        )
        self.task_id = self.progress.add_task(self.description, total=None)
        self.progress.start()
        return self

    def __exit__(self, *args):
        if self.progress and self.task_id is not None:
            self.progress.update(self.task_id, completed=100)
            self.progress.stop()


class IterationProgress:
    """Context manager for displaying iteration progress."""

    def __init__(self, current: int, max_iterations: int):
        self.current = current
        self.max = max_iterations

    def __enter__(self):
        self.progress = Progress(
            SpinnerColumn(spinner_name="simpleDotsScrolling"),
            TextColumn(f"[bold cyan]Iteration {self.current}/{self.max}[/bold cyan]"),
            BarColumn(bar_width=30, complete_style="green", finished_style="green"),
            console=console,
        )
        self.task_id = self.progress.add_task("", total=self.max, completed=self.current)
        self.progress.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.progress:
            self.progress.stop()


def create_progress(description: str = "Working...") -> Progress:
    """Create a progress object with spinner."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        console=console,
        transient=True,
    )


# ============================================================================
# Status Display
# ============================================================================

class StatusDisplay:
    """Display real-time status information."""

    def __init__(self):
        self.layout = Layout()
        self._setup_layout()

    def _setup_layout(self):
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="status"),
        )

    def update_status(self, status: str, action: str = "", iteration: int = 0,
                      max_iterations: int = 50) -> None:
        """Update the status display."""
        status_text = Text()
        status_text.append(f"Status: {status}\n", style="bold")
        if action:
            status_text.append(f"Action: {action}\n", style="cyan")
        status_text.append(f"Iteration: {iteration}/{max_iterations}", style="dim")
        self.layout["status"].update(Panel(status_text, title="Current Status"))


# ============================================================================
# Action History Table
# ============================================================================

def show_action_history(history: List[Dict[str, Any]]) -> None:
    """Display action history in a formatted table."""
    if not history:
        console.print("[dim]No actions recorded yet.[/dim]")
        return

    table = Table(
        title="Action History",
        border_style="blue",
        show_header=True,
        header_style="bold magenta",
    )

    table.add_column("#", style="dim", width=4)
    table.add_column("Action", style="cyan", width=12)
    table.add_column("Details", style="green", max_width=50)
    table.add_column("Status", width=10)

    for idx, action_record in enumerate(history, 1):
        action_type = action_record.get('action', 'UNKNOWN').upper()
        details = action_record.get('details', '')
        status = action_record.get('status', 'pending')

        # Color code by status
        if status == 'success':
            status_style = "green"
        elif status == 'failed':
            status_style = "red"
        elif status == 'pending':
            status_style = "yellow"
        else:
            status_style = "dim"

        table.add_row(
            str(idx),
            action_type,
            details[:50] + "..." if len(details) > 50 else details,
            f"[{status_style}]{status}[/{status_style}]",
        )

    console.print()
    console.print(table)
    console.print()


def show_action_summary(total_actions: int, successful: int, failed: int) -> None:
    """Display action summary statistics."""
    summary_text = f"""[bold]Total Actions:[/bold] {total_actions}
[green]Successful:[/green] {successful}
[red]Failed:[/red] {failed}
[dim]Success Rate: {successful/total_actions*100:.1f}%[/dim]""" if total_actions > 0 else "[dim]No actions recorded[/dim]"

    console.print(Panel(
        summary_text,
        title="[bold]Session Summary[/bold]",
        border_style="green",
    ))


# ============================================================================
# User Input Functions
# ============================================================================

class Command:
    """Built-in commands for the CLI."""
    HELP = ['help', 'h', '?']
    QUIT = ['quit', 'exit', 'q']
    HISTORY = ['history', 'hist', 'hst']
    CLEAR = ['clear', 'cls']
    CONFIG = ['config', 'cfg']

    @classmethod
    def is_command(cls, text: str) -> bool:
        """Check if input is a built-in command."""
        text_lower = text.strip().lower()
        return any(text_lower in commands for commands in [
            cls.HELP, cls.QUIT, cls.HISTORY, cls.CLEAR, cls.CONFIG
        ])

    @classmethod
    def get_command_type(cls, text: str) -> Optional[str]:
        """Get the command type from input."""
        text_lower = text.strip().lower()
        if text_lower in cls.HELP:
            return 'help'
        elif text_lower in cls.QUIT:
            return 'quit'
        elif text_lower in cls.HISTORY:
            return 'history'
        elif text_lower in cls.CLEAR:
            return 'clear'
        elif text_lower in cls.CONFIG:
            return 'config'
        return None


def get_user_instruction(prompt_text: str = "Enter your instruction") -> Optional[str]:
    """Get user instruction with styled prompt."""
    try:
        instruction = Prompt.ask(
            f"\n[bold green]{prompt_text}[/bold green]",
            default="",
        )
        return instruction.strip() if instruction else None
    except (KeyboardInterrupt, EOFError):
        return None


def confirm_action(message: str, default: bool = False) -> bool:
    """Ask for user confirmation."""
    return Confirm.ask(f"[yellow]{message}[/yellow]", default=default)


def show_help() -> None:
    """Display help information."""
    help_text = """[bold]Available Commands:[/bold]

[green]help, h, ?[/green]       - Show this help message
[green]quit, exit, q[/green]    - Exit the application
[green]history, hist[/green]    - View action history
[green]clear, cls[/green]       - Clear screen
[green]config, cfg[/green]      - Show current configuration

[bold]Example Instructions:[/bold]

[green]"Open Calculator"[/green]
  Opens the Windows Calculator application

[green]"Search for Python tutorials on Google"[/green]
  Opens browser and searches Google

[green]"Type 'Hello World'"[/green]
  Types the specified text in the active window

[green]"Click the OK button"[/green]
  Clicks on the OK button

[green]"Scroll down"[/green]
  Scrolls the current window down

[dim]Tip: Be specific and descriptive in your instructions.[/dim]
"""
    console.print(Panel(
        help_text,
        title="[bold blue]Help[/bold blue]",
        border_style="blue",
        padding=(1, 2),
    ))


# ============================================================================
# Main CLI Class
# ============================================================================

class CLI:
    """Main CLI controller for GUI Agent."""

    def __init__(self):
        self.console = console
        self.history: List[Dict[str, Any]] = []
        self.running = True

    def start(self) -> None:
        """Start the CLI main loop."""
        show_welcome_panel()
        show_safety_warning()

        while self.running:
            instruction = get_user_instruction("Enter your instruction")

            if instruction is None:
                self.running = False
                continue

            # Check for built-in commands
            if Command.is_command(instruction):
                cmd_type = Command.get_command_type(instruction)
                if cmd_type:
                    self._handle_command(cmd_type)
                continue

            # Process the instruction
            self.on_instruction_received(instruction)

    def _handle_command(self, cmd_type: str) -> None:
        """Handle built-in commands."""
        if cmd_type == 'help':
            show_help()
        elif cmd_type == 'quit':
            self.running = False
            console.print("[bold blue]Goodbye![/bold blue]")
        elif cmd_type == 'history':
            show_action_history(self.history)
        elif cmd_type == 'clear':
            self.console.clear()
        elif cmd_type == 'config':
            console.print("[yellow]Configuration not loaded.[/yellow]")

    def on_instruction_received(self, instruction: str) -> None:
        """Override this method in subclass to handle instructions."""
        # Default implementation - should be overridden
        console.print(f"[dim]Instruction received: {instruction}[/dim]")

    def add_action_to_history(self, action: str, details: str, status: str = 'pending') -> None:
        """Add an action to the history."""
        self.history.append({
            'action': action,
            'details': details,
            'status': status,
            'timestamp': datetime.now().isoformat(),
        })

    def clear_history(self) -> None:
        """Clear the action history."""
        self.history.clear()


# ============================================================================
# Convenience Functions
# ============================================================================

def print_divider(title: str = "") -> None:
    """Print a styled divider."""
    if title:
        console.print(f"\n[bold cyan]{'=' * 20} {title} {'=' * 20}[/bold cyan]\n")
    else:
        console.print(f"\n[dim]{'=' * 50}[/dim]\n")


def print_banner(text: str) -> None:
    """Print a banner with the given text."""
    console.print()
    console.print(Panel(
        f"[bold white]{text}[/bold white]",
        border_style="cyan",
        padding=(0, 2),
    ))
    console.print()


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    # Test the CLI module
    cli = CLI()
    cli.start()
