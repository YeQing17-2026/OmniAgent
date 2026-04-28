import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Task 1: CodexBinaryResolver
# ---------------------------------------------------------------------------

def test_resolver_uses_env_var(tmp_path):
    fake_bin = tmp_path / "codex"
    fake_bin.touch(mode=0o755)
    from omniagent.llm.codex_appserver import CodexBinaryResolver
    with patch.dict(os.environ, {"OMNIAGENT_CODEX_BIN": str(fake_bin)}):
        assert CodexBinaryResolver.resolve() == str(fake_bin)


def test_resolver_falls_back_to_which():
    from omniagent.llm.codex_appserver import CodexBinaryResolver
    with patch.dict(os.environ, {}, clear=True):
        with patch("shutil.which", return_value="/usr/local/bin/codex"):
            assert CodexBinaryResolver.resolve() == "/usr/local/bin/codex"


def test_resolver_raises_when_not_found():
    from omniagent.llm.codex_appserver import CodexBinaryResolver
    with patch.dict(os.environ, {}, clear=True):
        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="codex"):
                CodexBinaryResolver.resolve()
