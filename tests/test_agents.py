"""Tests for agent roles and graph logic."""

import pytest
from agents.graph import _extract_code
from agents.llm import LLMConfig, OllamaLLM, ModelSpec, ModelSelectionStrategy


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


def test_model_selection():
    """Test model selection logic."""
    models = [
        ModelSpec(name="qwen2.5:7b", priority=1, min_vram_gb=6),
        ModelSpec(name="qwen2.5:3b", priority=2, min_vram_gb=4),
    ]
    cfg = LLMConfig(
        model="qwen2.5:7b",
        models=models,
        selection_strategy=ModelSelectionStrategy.COMPLEXITY_BASED
    )
    llm = OllamaLLM(cfg)

    # Test simple task
    selected = llm.select_models("simple")
    assert len(selected) == 2
    assert selected[0].name == "qwen2.5:3b"  # Smaller first for simple

    # Test complex task
    selected = llm.select_models("complex")
    assert len(selected) == 2
    assert selected[0].name == "qwen2.5:7b"  # Larger first for complex


def test_task_complexity_estimation():
    """Test task complexity estimation."""
    cfg = LLMConfig(model="test")
    llm = OllamaLLM(cfg)

    assert llm.estimate_task_complexity("Compute mean") == "simple"
    assert llm.estimate_task_complexity("Analyze complex strategy with optimization") == "complex"
    assert llm.estimate_task_complexity("This is a longer task with more words to make it medium complexity") == "medium"