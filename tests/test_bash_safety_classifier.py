"""Tests for the bash safety classifier (ReflexionAgent._is_safe_bash_command).

Regression coverage for issue #67: `find` is on the read-only allowlist,
but its command-execution primaries (-exec, -execdir, -ok, -okdir) run an
arbitrary command for every match. The glued spellings (-execdir/-okdir)
evaded the denylist's r'\\bexec\\b' (no word boundary before 'dir'), so a
command such as

    find . -execdir dd if=/dev/zero of={} bs=1M count=100 \\;

was classified "safe" and skipped the human-approval gate, even though the
payload binary is never checked against the allowlist or denylist.
"""

import pytest

from omniagent.agents.reflexion import ReflexionAgent

is_safe = ReflexionAgent._is_safe_bash_command


@pytest.mark.parametrize(
    "command",
    [
        # find execution primaries: the payload runs without approval checks
        "find . -type f -name '*.tmp' -execdir dd if=/dev/zero of={} bs=1M count=100 \\;",
        "find . -execdir python3 -c 'import os; os.system(\"id\")' \\;",
        "find . -execdir sh -c 'id' \\;",
        "find . -type f -okdir dd if=/dev/zero of={} \\;",
        "find . -type f -ok dd if=/dev/zero of={} \\;",
        "find . -type f -exec dd if=/dev/zero of={} \\;",
        # already denied before this fix; must stay denied
        "find . -type f -exec rm {} \\;",
        "rm -rf /tmp/x",
        "curl https://example.com",
        "python3 -c 'print(1)'",
    ],
)
def test_unsafe_bash_commands_require_approval(command):
    assert is_safe(command) is False


@pytest.mark.parametrize(
    "command",
    [
        "find . -name '*.py'",
        "find /tmp -type f -mtime -1",
        "find . -name '*.log' -print",
        "ls -la",
        "grep -rn execdir omniagent/",
    ],
)
def test_read_only_bash_commands_stay_auto_approved(command):
    assert is_safe(command) is True
