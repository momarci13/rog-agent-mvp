"""Task conversation processing and refinement logic.

Handles:
- User message routing (Q&A, refinement, iteration)
- Artifact re-execution with code edits
- Task branching and variation creation
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

from agents.graph import RunState, Message
from agents.llm import OllamaLLM
from agents.problem_decoder import decode_problem
from rag.hybrid import LiteHybridRAG
from tools.sandbox import run_code_sync
from tools import task_storage


def process_user_message(
    task_id: str,
    message_content: str,
    llm: OllamaLLM,
    rag: LiteHybridRAG,
    iteration: int = 0,
) -> tuple[str, list[dict]]:
    """Process a user message and generate assistant response.
    
    Args:
        task_id: ID of the task to process
        message_content: User's message text
        llm: Language model instance
        rag: RAG system instance
        iteration: Which iteration this belongs to
    
    Returns:
        (assistant_response_text, new_artifacts)
    """
    # Load task
    task = task_storage.load_task(task_id)
    if not task:
        raise ValueError(f"Task {task_id} not found")
    
    # Add user message to task
    user_msg = Message(
        role="user",
        content=message_content,
        timestamp=datetime.utcnow(),
        iteration=iteration,
    )
    task.messages.append(user_msg)
    
    # Retrieve relevant context from RAG
    docs = rag.retrieve(message_content, k=3)
    doc_context = "\n".join([d.get("text", "") for d in docs])
    
    # Determine response type based on message content
    response_type = _classify_message(message_content)
    
    # Generate response
    if response_type == "refinement":
        response, artifacts = _handle_refinement(
            task, message_content, doc_context, llm
        )
    elif response_type == "question":
        response, artifacts = _handle_question(
            task, message_content, doc_context, llm
        )
    else:  # new_iteration
        response, artifacts = _handle_new_iteration(
            task, message_content, doc_context, llm
        )
    
    # Add assistant response
    assistant_msg = Message(
        role="assistant",
        content=response,
        timestamp=datetime.utcnow(),
        iteration=iteration,
    )
    task.messages.append(assistant_msg)
    
    # Update task with new artifacts
    if artifacts:
        task.artifacts.extend(artifacts)
    
    # Save updated task
    task_storage.save_task(task)
    
    return response, artifacts


def re_execute_artifact(
    task_id: str,
    artifact_id: str,
    edited_code: str,
    artifact_index: int = -1,
) -> dict:
    """Re-execute an artifact with edited code.
    
    Args:
        task_id: Task ID
        artifact_id: Artifact ID (for reference)
        edited_code: Modified code to execute
        artifact_index: Index in artifacts list (-1 for last)
    
    Returns:
        {
            "status": "success" | "error",
            "stdout": str,
            "stderr": str,
            "returncode": int,
            "execution_time": float,
        }
    """
    task = task_storage.load_task(task_id)
    if not task:
        raise ValueError(f"Task {task_id} not found")
    
    if artifact_index < 0:
        artifact_index = len(task.artifacts) + artifact_index
    
    if artifact_index < 0 or artifact_index >= len(task.artifacts):
        raise ValueError(f"Artifact index {artifact_index} out of range")
    
    artifact = task.artifacts[artifact_index]
    
    # Execute code
    result = run_code_sync(
        code=edited_code,
        timeout_s=30,
        mem_mb=512,
    )
    
    # Create new artifact version with results
    new_artifact = {
        "id": f"{artifact_id}_v{len(task.artifacts)}",
        "type": artifact.get("type", "ds"),
        "payload": result,
        "raw": edited_code,
        "iteration": task.iterations,
        "edited": True,
        "edited_at": datetime.utcnow().isoformat(),
    }
    
    # Append to artifacts
    task.artifacts.append(new_artifact)
    
    # Add message to thread
    msg = Message(
        role="system",
        content=f"User edited and re-ran artifact {artifact_id}. Return code: {result['returncode']}",
        timestamp=datetime.utcnow(),
        iteration=task.iterations,
    )
    task.messages.append(msg)
    
    # Save
    task_storage.save_task(task)
    
    return result


def branch_task(
    task_id: str,
    branch_name: Optional[str] = None,
    from_iteration: int = 0,
) -> str:
    """Branch a task to create a variation.
    
    Args:
        task_id: Original task ID
        branch_name: Name for the branch
        from_iteration: Which iteration to branch from (0 = start fresh)
    
    Returns:
        new_task_id
    """
    new_task_id = task_storage.branch_task(
        task_id,
        branch_name=branch_name,
        from_iteration=from_iteration,
    )
    
    # Log branching action
    task = task_storage.load_task(new_task_id)
    if task:
        msg = Message(
            role="system",
            content=f"Branch created from task {task_id}" + (f" (branch: {branch_name})" if branch_name else ""),
            timestamp=datetime.utcnow(),
            iteration=0,
        )
        task.messages.append(msg)
        task_storage.save_task(task)
    
    return new_task_id


def _classify_message(content: str) -> str:
    """Classify user message type.
    
    Returns: "question" | "refinement" | "new_iteration"
    """
    content_lower = content.lower()
    
    # Refinement keywords
    refinement_keywords = [
        "refine", "improve", "adjust", "modify", "change",
        "fix", "fix the", "update", "edit", "tweak",
        "try again", "better", "more", "less",
    ]
    
    # Question keywords
    question_keywords = [
        "what", "how", "why", "where", "when",
        "can you", "could you", "explain", "understand",
        "tell me", "show me", "interpret",
    ]
    
    # New iteration keywords
    iteration_keywords = [
        "run again", "execute again", "retry", "restart",
        "new run", "from scratch", "start over",
    ]
    
    if any(kw in content_lower for kw in iteration_keywords):
        return "new_iteration"
    elif any(kw in content_lower for kw in refinement_keywords):
        return "refinement"
    elif any(kw in content_lower for kw in question_keywords):
        return "question"
    else:
        # Default to question for most things
        return "question"


def _handle_question(
    task: RunState,
    message: str,
    doc_context: str,
    llm: OllamaLLM,
) -> tuple[str, list[dict]]:
    """Handle a clarification/information question.
    
    Returns (response_text, artifacts).
    """
    from agents import roles
    
    prompt = f"""You are assisting with a data analysis task.

Original task: {task.task}

Task type: {task.task_type}

Current iteration: {task.iterations}

User question: {message}

Relevant documentation:
{doc_context}

Provide a clear, concise answer to the user's question in the context of this task.
If you can provide code improvements or suggestions, include them.
"""
    
    response = llm.invoke(prompt)
    
    return response, []


def _handle_refinement(
    task: RunState,
    message: str,
    doc_context: str,
    llm: OllamaLLM,
) -> tuple[str, list[dict]]:
    """Handle a refinement request.
    
    Returns (response_text, artifacts).
    """
    from agents import roles
    
    # Extract last artifact for context
    last_artifact = task.artifacts[-1] if task.artifacts else None
    last_artifact_context = ""
    if last_artifact:
        last_artifact_context = f"\nLast artifact type: {last_artifact.get('type')}\n"
        if last_artifact.get('raw'):
            last_artifact_context += f"Last code:\n{last_artifact['raw']}\n"
    
    prompt = f"""You are assisting with a data analysis task refinement.

Original task: {task.task}

Task type: {task.task_type}

Current iteration: {task.iterations}

Refinement request: {message}

{last_artifact_context}

Relevant documentation:
{doc_context}

Provide a concise response acknowledging the refinement request.
Suggest what changes you would make to improve the analysis.
"""
    
    response = llm.invoke(prompt)
    
    return response, []


def _handle_new_iteration(
    task: RunState,
    message: str,
    doc_context: str,
    llm: OllamaLLM,
) -> tuple[str, list[dict]]:
    """Handle request to start a new iteration.
    
    Returns (response_text, artifacts).
    """
    prompt = f"""You are assisting with restarting a data analysis task.

Original task: {task.task}

Task type: {task.task_type}

Previous iterations: {task.iterations}

User request: {message}

Relevant documentation:
{doc_context}

Acknowledge the request to restart and summarize what we'll do in the new iteration.
"""
    
    response = llm.invoke(prompt)
    
    return response, []
