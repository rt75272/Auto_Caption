"""
Colored terminal messages shared by every stage of the program.

All progress reporting goes through this module so the console reads the same
way from start to finish. Colors organize the output by severity:

    cyan   [INFO]     what the program is doing or which model it is using
    green  [OK]       a step finished successfully
    yellow [WARNING]  something was skipped or could not be confirmed
    red    [ERROR]    something failed

Messages are written with tqdm.write rather than print so they appear above
any active progress bar instead of corrupting it. colorama is initialised on
import so the same ANSI colors also work in a Windows console.

Provides:
    format_message: build a colored "[LABEL] message" string.
    info, ok, warning, error: print a message at that severity.
    error_text: format an error as a string, for use in exception messages.

Used by: every module that reports progress to the user.
"""

from colorama import Fore, Style, just_fix_windows_console
from tqdm import tqdm

just_fix_windows_console()


def format_message(color, label, message):
    """
    Build a colored ``[LABEL] message`` line.

    Args:
        color: A colorama Fore constant applied to the bracketed label.
        label: Short severity name such as ``INFO``.
        message: Text that follows the label.

    Returns:
        The formatted string, without a trailing newline.
    """
    return f"{color}[{label}]{Style.RESET_ALL} {message}"


def info(message):
    """
    Print a cyan information message without disturbing progress bars.

    Args:
        message: Text to display after the ``[INFO]`` label.

    Returns:
        None.
    """
    tqdm.write(format_message(Fore.CYAN, "INFO", message))


def ok(message):
    """
    Print a green success message without disturbing progress bars.

    Args:
        message: Text to display after the ``[OK]`` label.

    Returns:
        None.
    """
    tqdm.write(format_message(Fore.GREEN, "OK", message))


def warning(message):
    """
    Print a yellow warning without disturbing progress bars.

    Args:
        message: Text to display after the ``[WARNING]`` label.

    Returns:
        None.
    """
    tqdm.write(format_message(Fore.YELLOW, "WARNING", message))


def error(message):
    """
    Print a red error message without disturbing progress bars.

    Args:
        message: Text to display after the ``[ERROR]`` label.

    Returns:
        None.
    """
    tqdm.write(format_message(Fore.RED, "ERROR", message))


def error_text(message):
    """
    Format an error for use as exception text instead of printing it.

    Args:
        message: Text to display after the ``[ERROR]`` label.

    Returns:
        The red ``[ERROR]`` line as a string.
    """
    return format_message(Fore.RED, "ERROR", message)
