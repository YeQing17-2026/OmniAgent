"""File system utilities with safety checks."""

from pathlib import Path
from typing import Optional


# System directories that are always blocked
_BLOCKED_PREFIXES = (
    Path("/etc"),
    Path("/var"),
    Path("/usr"),
    Path("/bin"),
    Path("/sbin"),
    Path("/System"),
    Path("/Library"),
    Path("/private"),
    Path("/proc"),
    Path("/sys"),
    Path("/dev"),
)


class PathTraversalError(Exception):
    """Raised when path traversal is detected."""
    pass


def safe_path(base_dir: Path, target_path: Path, allow_home: bool = False) -> Path:
    """
    Validate that target_path is within base_dir (prevent path traversal).

    Args:
        base_dir: Base directory (typically work_dir)
        target_path: Target path to validate
        allow_home: If True, also allow paths under the user's home directory
                     (e.g., ~/.omniagent, ~/Documents, ~/Desktop). Only the current
                     user's home is allowed — system dirs remain blocked.

    Returns:
        Resolved absolute path

    Raises:
        PathTraversalError: If path traversal is detected
    """
    home_dir = Path.home().resolve()
    base_dir = base_dir.resolve()

    # If target_path is relative, make it relative to base_dir
    if not target_path.is_absolute():
        target_path = base_dir / target_path

    target_path = target_path.resolve()

    # Always block system directories
    for prefix in _BLOCKED_PREFIXES:
        try:
            target_path.relative_to(prefix)
            raise PathTraversalError(
                f"Access denied: {target_path} is in a system directory"
            )
        except ValueError:
            pass  # Not under this prefix — OK

    # Check if target is within base_dir
    try:
        target_path.relative_to(base_dir)
        return target_path
    except ValueError:
        pass

    # If allow_home, also check home directory
    if allow_home:
        try:
            target_path.relative_to(home_dir)
            return target_path
        except ValueError:
            pass

    raise PathTraversalError(
        f"Path traversal detected: {target_path} is outside {base_dir}"
    )


def read_file(
    file_path: Path,
    base_dir: Optional[Path] = None,
    allow_home: bool = False,
    encoding: str = "utf-8"
) -> str:
    """
    Read file with optional safety check.

    Args:
        file_path: File to read
        base_dir: If provided, validate path is within base_dir
        allow_home: If True, also allow reading files under home directory
        encoding: File encoding

    Returns:
        File contents

    Raises:
        PathTraversalError: If path traversal detected
        FileNotFoundError: If file doesn't exist
    """
    if base_dir:
        file_path = safe_path(base_dir, file_path, allow_home=allow_home)

    with open(file_path, "r", encoding=encoding) as f:
        return f.read()


def write_file(
    file_path: Path,
    content: str,
    base_dir: Optional[Path] = None,
    allow_home: bool = False,
    encoding: str = "utf-8",
    create_dirs: bool = True
) -> None:
    """
    Write file with optional safety check.

    Args:
        file_path: File to write
        content: Content to write
        base_dir: If provided, validate path is within base_dir
        allow_home: If True, also allow writing files under home directory
        encoding: File encoding
        create_dirs: Create parent directories if needed

    Raises:
        PathTraversalError: If path traversal detected
    """
    if base_dir:
        file_path = safe_path(base_dir, file_path, allow_home=allow_home)

    if create_dirs:
        file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding=encoding) as f:
        f.write(content)


def ensure_dir(dir_path: Path, base_dir: Optional[Path] = None, allow_home: bool = False) -> Path:
    """
    Ensure directory exists with optional safety check.

    Args:
        dir_path: Directory to create
        base_dir: If provided, validate path is within base_dir
        allow_home: If True, also allow dirs under home directory

    Returns:
        Resolved directory path

    Raises:
        PathTraversalError: If path traversal detected
    """
    if base_dir:
        dir_path = safe_path(base_dir, dir_path, allow_home=allow_home)

    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path
