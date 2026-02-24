"""
GUI Agent - Configuration Module

Centralized configuration management with support for:
- Environment variables via .env file
- Command-line arguments via click
- Default values and validation
"""

import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv
import click


# Default configuration values
DEFAULTS = {
    'model': 'qwen3-vl-plus',
    'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    'max_iterations': 50,
    'timeout': 60,
    'temperature': 0.7,
    'max_tokens': 2048,
    'grid_size': 10,
    'screenshot_quality': 85,
    'screenshot_max_width': 1280,
    'action_delay': 0.5,
    'verify_ssl': True,
}


@dataclass
class Config:
    """Configuration container for GUI Agent."""

    # API Configuration
    api_key: str = ""
    base_url: str = DEFAULTS['base_url']
    model: str = DEFAULTS['model']

    # Agent Configuration
    max_iterations: int = DEFAULTS['max_iterations']
    timeout: int = DEFAULTS['timeout']
    temperature: float = DEFAULTS['temperature']
    max_tokens: int = DEFAULTS['max_tokens']

    # Screen Capture Configuration
    grid_size: int = DEFAULTS['grid_size']
    screenshot_quality: int = DEFAULTS['screenshot_quality']
    screenshot_max_width: int = DEFAULTS['screenshot_max_width']

    # Action Configuration
    action_delay: float = DEFAULTS['action_delay']

    # Advanced Configuration
    verify_ssl: bool = DEFAULTS['verify_ssl']
    debug: bool = False

    # Path configuration
    env_file: str = ".env"

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self):
        """Validate configuration values."""
        if self.max_iterations < 1 or self.max_iterations > 100:
            raise ValueError("max_iterations must be between 1 and 100")

        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("temperature must be between 0 and 2")

        if self.max_tokens < 100 or self.max_tokens > 8192:
            raise ValueError("max_tokens must be between 100 and 8192")

        if self.grid_size < 5 or self.grid_size > 20:
            raise ValueError("grid_size must be between 5 and 20")

        if self.screenshot_quality < 10 or self.screenshot_quality > 100:
            raise ValueError("screenshot_quality must be between 10 and 100")

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'api_key': self.api_key,
            'base_url': self.base_url,
            'model': self.model,
            'max_iterations': self.max_iterations,
            'timeout': self.timeout,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'grid_size': self.grid_size,
            'screenshot_quality': self.screenshot_quality,
            'screenshot_max_width': self.screenshot_max_width,
            'action_delay': self.action_delay,
            'verify_ssl': self.verify_ssl,
            'debug': self.debug,
        }

    def __getitem__(self, key: str):
        """Allow dictionary-style access."""
        return self.to_dict().get(key)

    def get(self, key: str, default=None):
        """Get configuration value with default."""
        return self.to_dict().get(key, default)


class ConfigLoader:
    """Load configuration from various sources."""

    def __init__(self, env_file: Optional[str] = None):
        """Initialize config loader.

        Args:
            env_file: Path to .env file. If None, searches default locations.
        """
        self.env_file = env_file
        self._load_env()

    def _load_env(self):
        """Load environment variables from .env file."""
        if self.env_file:
            env_path = Path(self.env_file)
            if env_path.exists():
                load_dotenv(env_path)
        else:
            # Search for .env in current directory and parent directories
            load_dotenv()

    def load(self) -> Config:
        """Load configuration from environment variables.

        Returns:
            Config object with loaded values.
        """
        return Config(
            api_key=os.getenv('DASHSCOPE_API_KEY', ''),
            base_url=os.getenv('BASE_URL', DEFAULTS['base_url']),
            model=os.getenv('MODEL_NAME', DEFAULTS['model']),
            max_iterations=int(os.getenv('MAX_ITERATIONS', DEFAULTS['max_iterations'])),
            timeout=int(os.getenv('TIMEOUT', DEFAULTS['timeout'])),
            temperature=float(os.getenv('TEMPERATURE', DEFAULTS['temperature'])),
            max_tokens=int(os.getenv('MAX_TOKENS', DEFAULTS['max_tokens'])),
            grid_size=int(os.getenv('GRID_SIZE', DEFAULTS['grid_size'])),
            screenshot_quality=int(os.getenv('SCREENSHOT_QUALITY', DEFAULTS['screenshot_quality'])),
            screenshot_max_width=int(os.getenv('SCREENSHOT_MAX_WIDTH', DEFAULTS['screenshot_max_width'])),
            action_delay=float(os.getenv('ACTION_DELAY', DEFAULTS['action_delay'])),
            verify_ssl=os.getenv('VERIFY_SSL', 'true').lower() == 'true',
            debug=os.getenv('DEBUG', 'false').lower() == 'true',
            env_file=self.env_file or ".env",
        )


def create_click_options():
    """Create click decorator for CLI options.

    Returns:
        Decorator function for click options.
    """
    def decorator(f):
        options = [
            click.option(
                '--api-key', '-k',
                envvar='DASHSCOPE_API_KEY',
                help='DashScope API key (env: DASHSCOPE_API_KEY)',
            ),
            click.option(
                '--base-url', '-u',
                envvar='BASE_URL',
                default=DEFAULTS['base_url'],
                help=f'API base URL (env: BASE_URL, default: {DEFAULTS["base_url"]})',
            ),
            click.option(
                '--model', '-m',
                envvar='MODEL_NAME',
                default=DEFAULTS['model'],
                help=f'Model name (env: MODEL_NAME, default: {DEFAULTS["model"]})',
            ),
            click.option(
                '--max-iterations', '-i',
                type=int,
                default=DEFAULTS['max_iterations'],
                help=f'Maximum iterations per task (default: {DEFAULTS["max_iterations"]})',
            ),
            click.option(
                '--timeout', '-t',
                type=int,
                default=DEFAULTS['timeout'],
                help=f'API timeout in seconds (default: {DEFAULTS["timeout"]})',
            ),
            click.option(
                '--temperature',
                type=float,
                default=DEFAULTS['temperature'],
                help=f'Model temperature (default: {DEFAULTS["temperature"]})',
            ),
            click.option(
                '--grid-size',
                type=int,
                default=DEFAULTS['grid_size'],
                help=f'Grid overlay size (default: {DEFAULTS["grid_size"]})',
            ),
            click.option(
                '--debug', '-d',
                is_flag=True,
                default=False,
                help='Enable debug mode',
            ),
        ]
        for option in reversed(options):
            f = option(f)
        return f
    return decorator


def load_config(
    env_file: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    max_iterations: Optional[int] = None,
    timeout: Optional[int] = None,
    temperature: Optional[float] = None,
    grid_size: Optional[int] = None,
    debug: bool = False,
) -> Config:
    """Load configuration with optional overrides.

    Args:
        env_file: Path to .env file.
        api_key: Override API key.
        base_url: Override base URL.
        model: Override model name.
        max_iterations: Override max iterations.
        timeout: Override timeout.
        temperature: Override temperature.
        grid_size: Override grid size.
        debug: Enable debug mode.

    Returns:
        Config object with loaded values.
    """
    loader = ConfigLoader(env_file)
    config = loader.load()

    # Apply command-line overrides
    if api_key:
        config.api_key = api_key
    if base_url:
        config.base_url = base_url
    if model:
        config.model = model
    if max_iterations is not None:
        config.max_iterations = max_iterations
    if timeout is not None:
        config.timeout = timeout
    if temperature is not None:
        config.temperature = temperature
    if grid_size is not None:
        config.grid_size = grid_size
    if debug:
        config.debug = debug

    return config


def validate_api_key(config: Config) -> bool:
    """Validate that API key is configured.

    Args:
        config: Configuration to validate.

    Returns:
        True if API key is configured, False otherwise.
    """
    if not config.api_key or config.api_key.startswith('sk-your-'):
        return False
    return True


def get_api_key_status(config: Config) -> str:
    """Get API key status message.

    Args:
        config: Configuration to check.

    Returns:
        Status message about API key configuration.
    """
    if not config.api_key:
        return "API key not set"
    elif config.api_key.startswith('sk-your-'):
        return "API key not configured (using placeholder)"
    elif len(config.api_key) < 20:
        return "API key appears invalid (too short)"
    else:
        return f"API key configured ({config.api_key[:10]}...)"


# ============================================================================
# Click CLI Commands
# ============================================================================

@click.group()
def cli():
    """GUI Agent Configuration CLI."""
    pass


@cli.command()
@click.option('--env-file', '-e', default='.env', help='Path to .env file')
def show(env_file: str):
    """Show current configuration."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    config = load_config(env_file=env_file)

    table = Table(title="GUI Agent Configuration", border_style="blue")
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    table.add_column("Source", style="dim")

    config_dict = config.to_dict()
    for key, value in config_dict.items():
        if key == 'api_key' and value:
            value = f"{value[:10]}..." if len(value) > 10 else value
        table.add_row(key, str(value), "env/default")

    console.print(table)

    # Show API key status
    api_status = get_api_key_status(config)
    if validate_api_key(config):
        console.print(f"\n[green]✓[/green] {api_status}")
    else:
        console.print(f"\n[yellow]![/yellow] {api_status}")
        console.print("\n[dim]Set DASHSCOPE_API_KEY in your .env file or use --api-key flag[/dim]")


@cli.command()
def defaults():
    """Show default configuration values."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    table = Table(title="Default Configuration Values", border_style="blue")
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Default Value", style="green")
    table.add_column("Range", style="dim")

    defaults_info = [
        ('model', DEFAULTS['model'], ''),
        ('base_url', DEFAULTS['base_url'], ''),
        ('max_iterations', str(DEFAULTS['max_iterations']), '1-100'),
        ('timeout', str(DEFAULTS['timeout']), ''),
        ('temperature', str(DEFAULTS['temperature']), '0-2'),
        ('max_tokens', str(DEFAULTS['max_tokens']), '100-8192'),
        ('grid_size', str(DEFAULTS['grid_size']), '5-20'),
        ('screenshot_quality', str(DEFAULTS['screenshot_quality']), '10-100'),
        ('screenshot_max_width', str(DEFAULTS['screenshot_max_width']), ''),
        ('action_delay', str(DEFAULTS['action_delay']), ''),
    ]

    for name, value, range_info in defaults_info:
        table.add_row(name, value, range_info)

    console.print(table)


@cli.command()
@click.option('--env-file', '-e', default='.env', help='Path to .env file')
def check(env_file: str):
    """Check configuration and environment."""
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    config = load_config(env_file=env_file)

    issues = []

    # Check API key
    if not validate_api_key(config):
        issues.append("[red]✗[/red] API key not configured")
    else:
        issues.append("[green]✓[/green] API key configured")

    # Check base URL
    if config.base_url:
        issues.append("[green]✓[/green] Base URL configured")
    else:
        issues.append("[yellow]![/yellow] Base URL not configured")

    # Check model
    if config.model:
        issues.append("[green]✓[/green] Model configured")
    else:
        issues.append("[yellow]![/yellow] Model not configured")

    status_text = "\n".join(issues)

    if not any("✗" in issue for issue in issues):
        panel = Panel(
            f"[green]Configuration looks good![/green]\n\n{status_text}",
            title="Configuration Check",
            border_style="green",
        )
    else:
        panel = Panel(
            f"[yellow]Configuration issues found:[/yellow]\n\n{status_text}",
            title="Configuration Check",
            border_style="yellow",
        )

    console.print(panel)


if __name__ == "__main__":
    cli()
