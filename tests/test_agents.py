"""Tests for agent roles and graph logic."""

import pytest
from agents.graph import _extract_code


def test_extract_code_python():
    md = """Here's some code:
```python
print("hello")
```
More text."""
    lang, code = _extract_code(md)
    assert lang == "python"
    assert code == 'print("hello")'


def test_extract_code_r():
    md = """R code example:
```r
print("hello from R")
```
End."""
    lang, code = _extract_code(md)
    assert lang == "r"
    assert code == 'print("hello from R")'


def test_extract_code_unknown():
    md = """Generic code:
```
some code
```
"""
    lang, code = _extract_code(md)
    assert lang == "unknown"
    assert code == "some code"


def test_extract_code_no_block():
    md = "No code block here."
    lang, code = _extract_code(md)
    assert lang == "unknown"
    assert code == "No code block here."