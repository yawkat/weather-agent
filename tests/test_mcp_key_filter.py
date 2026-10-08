"""McpKeyFilter: which paths reach MCP with and without an access key."""

import os
import tempfile

import pytest
from at.yawk.weatheragent import McpAccessConfig, McpKeyFilter

KEY = "k" * 31 + "-_Z9"


def key_filter(key, key_file):
    return McpKeyFilter(McpAccessConfig([], key, key_file))


@pytest.mark.parametrize("unset", ["", None])
def test_without_key_only_plain_path(unset):
    f = key_filter(unset, unset)
    assert f.allowed("/mcp")
    assert not f.allowed("/mcp/")
    assert not f.allowed("/mcp/" + KEY)


def test_with_key_only_secret_path():
    f = key_filter(KEY, "")
    assert f.allowed("/mcp/" + KEY)
    assert not f.allowed("/mcp")
    assert not f.allowed("/mcp/")
    assert not f.allowed("/mcp/" + KEY[:-1])
    assert not f.allowed("/mcp/" + KEY + "x")
    assert not f.allowed("/mcp/" + KEY + "/")
    assert not f.allowed("/mcpx/" + KEY)


def test_key_from_file_is_stripped():
    with tempfile.NamedTemporaryFile("w", suffix=".key", delete=False) as out:
        out.write(KEY + "\n")
    try:
        assert key_filter("", out.name).allowed("/mcp/" + KEY)
        with pytest.raises(BaseException, match="not both"):
            key_filter(KEY, out.name)
    finally:
        os.unlink(out.name)


def test_empty_key_file_fails_closed():
    """A placeholder or failed decryption leaves an empty credential: that must not open MCP."""
    with tempfile.NamedTemporaryFile("w", suffix=".key", delete=False) as out:
        out.write(" \n")
    try:
        with pytest.raises(BaseException, match="access key must be"):
            key_filter("", out.name)
    finally:
        os.unlink(out.name)


@pytest.mark.parametrize("key, key_file", [(" ", ""), ("\n", ""), ("", " "), ("", "\t")])
def test_whitespace_settings_fail_closed(key, key_file):
    """Only empty settings (the defaults) mean "no key"; anything else must be a valid key."""
    with pytest.raises(BaseException, match="access key must be|cannot read"):
        key_filter(key, key_file)


@pytest.mark.parametrize("key", ["short", "x" * 31, KEY + "/", KEY + "%2F", "ü" * 40])
def test_weak_or_unsafe_keys_rejected(key):
    with pytest.raises(BaseException, match="access key must be"):
        key_filter(key, "")
