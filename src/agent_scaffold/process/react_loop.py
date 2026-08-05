"""ReAct-style agent business loop.

The loop follows this shape:

1. Build a prompt with user input, known tools, session context, and memories.
2. Trim or assemble the context before each LLM call.
3. Ask the LLM what to do next.
4. If the LLM returns one or more tool calls, execute them and append observations.
5. Repeat until the LLM returns a final answer or max steps is reached.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

from agent_scaffold.context_trim import (
    ContextTrimRequest,
    ContextTrimmer,
    PassthroughContextTrimmer,
)
from agent_scaffold.llm_call import LLMClient, Message
from agent_scaffold.memory import MemoryStore
from agent_scaffold.session import SessionManager, SessionStatus
from agent_scaffold.tool_call import ToolCallRequest, ToolCallResult, ToolExecutor
from agent_scaffold.tool_register import ToolRegistry
from agent_scaffold.trace import log_event


@dataclass(slots=True)
class ReActConfig:
    """Runtime options for the ReAct loop."""

    max_steps: int = 6
    memory_limit: int = 5
    session_history_limit: int = 12
    save_user_input_to_memory: bool = False
    save_final_answer_to_memory: bool = False
    trace_log_path: str | None = "logs/react_trace.jsonl"
    system_prompt: str = (
        "You are a ReAct agent. Solve the user's task by reasoning, using tools "
        "when helpful, and returning a final answer.\n\n"
        "When you need a tool, respond with JSON only:\n"
        '{"thought": "short reasoning", "action": "tool_name", "arguments": {}}\n\n'
        "When you are done, respond with JSON only:\n"
        '{"thought": "short reasoning", "final_answer": "answer to user"}'
    )


@dataclass(slots=True)
class ReActStep:
    """One model/tool iteration in a ReAct run."""

    index: int
    thought: str = ""
    action: str | None = None
    tool_call_id: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    observation: Any = None
    final_answer: str | None = None
    raw_response: str = ""
    tool_result: ToolCallResult | None = None
    tool_results: list[ToolCallResult] = field(default_factory=list)


@dataclass(slots=True)
class ReActResult:
    """Final result returned by the ReAct loop."""

    answer: str
    session_id: str | None
    steps: list[ReActStep]
    completed: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class ReActAgent:
    """A small ReAct agent process built on the scaffold modules."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        memory_store: MemoryStore | None = None,
        session_manager: SessionManager | None = None,
        context_trimmer: ContextTrimmer | None = None,
        config: ReActConfig | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._tool_registry = tool_registry
        self._tool_executor = ToolExecutor(tool_registry)
        self._memory_store = memory_store
        self._session_manager = session_manager
        self._context_trimmer = context_trimmer or PassthroughContextTrimmer()
        self._config = config or ReActConfig()

    def run(
        self,
        *,
        user_id: str,
        query: str,
        session_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ReActResult:
        if not user_id:
            raise ValueError("user_id cannot be empty.")
        if not query:
            raise ValueError("query cannot be empty.")

        session_id = self._ensure_session(session_id, user_id)
        if self._memory_store and self._config.save_user_input_to_memory:
            self._memory_store.save(user_id, query, metadata={"type": "user_input"})

        messages = self._build_initial_messages(user_id, query, session_id)
        steps: list[ReActStep] = []
        answer = ""
        completed = False
        tools = self._tool_registry.as_llm_tools()

        for index in range(1, self._config.max_steps + 1):
            context = self._context_trimmer.trim(
                ContextTrimRequest(
                    messages=messages,
                    tools=tools,
                    session_id=session_id,
                    user_id=user_id,
                    query=query,
                    step_index=index,
                    metadata=metadata or {},
                )
            )
            llm_messages = context.messages
            log_event(
                self._config.trace_log_path,
                "llm_request",
                session_id=session_id,
                step_index=index,
                data={
                    "provider": provider,
                    "model": model,
                    "messages": [self._message_to_dict(message) for message in llm_messages],
                    "tools": tools,
                    "context_trim": context.metadata,
                    "metadata": metadata or {},
                },
            )
            response = self._llm_client.call(
                llm_messages,
                provider=provider,
                model=model,
                tools=tools,
                metadata=metadata,
            )
            log_event(
                self._config.trace_log_path,
                "llm_response",
                session_id=session_id,
                step_index=index,
                data={
                    "content": response.content,
                    "tool_calls": response.tool_calls,
                    "usage": response.usage,
                    "metadata": response.metadata,
                    "model": response.model,
                },
            )

            parsed = self._parse_response(response.content, response.tool_calls)
            step_tool_calls = self._parsed_tool_calls(parsed, session_id, index)
            first_tool_call = step_tool_calls[0] if step_tool_calls else {}
            step = ReActStep(
                index=index,
                thought=parsed.get("thought", ""),
                action=first_tool_call.get("name"),
                tool_call_id=first_tool_call.get("id"),
                arguments=first_tool_call.get("args", {}),
                tool_calls=step_tool_calls,
                final_answer=parsed.get("final_answer"),
                raw_response=response.content,
            )
            steps.append(step)
            messages.append(
                Message(
                    role="assistant",
                    content=response.content,
                    metadata={"tool_calls": self._assistant_tool_calls(step.tool_calls)},
                )
            )

            if step.final_answer is not None:
                answer = step.final_answer
                completed = True
                break

            if step.tool_calls:
                for tool_call in step.tool_calls:
                    tool_request = ToolCallRequest(
                        name=tool_call["name"],
                        arguments=tool_call.get("args", {}),
                        call_id=tool_call.get("id"),
                    )
                    log_event(
                        self._config.trace_log_path,
                        "tool_request",
                        session_id=session_id,
                        step_index=index,
                        data={
                            "name": tool_request.name,
                            "arguments": tool_request.arguments,
                            "call_id": tool_request.call_id,
                            "metadata": tool_request.metadata,
                        },
                    )
                    tool_result = self._tool_executor.call_request(tool_request)
                    log_event(
                        self._config.trace_log_path,
                        "tool_response",
                        session_id=session_id,
                        step_index=index,
                        data={
                            "name": tool_result.name,
                            "success": tool_result.success,
                            "result": tool_result.result,
                            "error": tool_result.error,
                            "call_id": tool_result.call_id,
                            "elapsed_ms": tool_result.elapsed_ms,
                            "metadata": tool_result.metadata,
                        },
                    )
                    step.tool_results.append(tool_result)
                    if step.tool_result is None:
                        step.tool_result = tool_result
                    messages.append(
                        Message(
                            role="tool",
                            name=tool_result.name,
                            tool_call_id=tool_result.call_id,
                            content=self._format_observation(tool_result),
                        )
                    )
                step.observation = [
                    result.result if result.success else result.error
                    for result in step.tool_results
                ]
                if len(step.observation) == 1:
                    step.observation = step.observation[0]
                continue

            answer = response.content
            completed = True
            break

        if not completed:
            answer = "Agent stopped because it reached the maximum number of steps."

        if self._memory_store and self._config.save_final_answer_to_memory:
            self._memory_store.save(user_id, answer, metadata={"type": "final_answer"})
        self._save_session_history(session_id, query, answer, completed)
        self._finish_session(session_id, completed)
        log_event(
            self._config.trace_log_path,
            "agent_finish",
            session_id=session_id,
            data={
                "answer": answer,
                "completed": completed,
                "max_steps": self._config.max_steps,
            },
        )

        return ReActResult(
            answer=answer,
            session_id=session_id,
            steps=steps,
            completed=completed,
            metadata={"max_steps": self._config.max_steps},
        )

    def _ensure_session(self, session_id: str | None, user_id: str) -> str | None:
        if not self._session_manager:
            return session_id
        if session_id:
            session = self._session_manager.get(session_id)
            if session.status in {
                SessionStatus.COMPLETED,
                SessionStatus.FAILED,
                SessionStatus.PAUSED,
            }:
                self._session_manager.transition(session_id, SessionStatus.ACTIVE)
            self._session_manager.set_context(session_id, "user_id", user_id)
            return session_id
        session = self._session_manager.create(context={"user_id": user_id})
        return session.id

    def _finish_session(self, session_id: str | None, completed: bool) -> None:
        if not self._session_manager or not session_id:
            return
        status = SessionStatus.COMPLETED if completed else SessionStatus.FAILED
        self._session_manager.transition(session_id, status)

    def _build_initial_messages(
        self,
        user_id: str,
        query: str,
        session_id: str | None,
    ) -> list[Message]:
        tool_text = self._format_tools()
        memory_text = self._format_memories(user_id, query)
        session_text = self._format_session(session_id)
        history_messages = self._session_history_messages(session_id)
        current_time = datetime.now().isoformat(timespec="seconds")

        return [
            Message(
                role="system",
                content=(
                    f"{self._config.system_prompt}\n\n"
                    f"Current system time: {current_time}\n\n"
                    f"Available tools:\n{tool_text}\n\n"
                    f"Relevant memories:\n{memory_text}\n\n"
                    f"Session context:\n{session_text}"
                ),
            ),
            *history_messages,
            Message(role="user", content=query),
        ]

    def _session_history_messages(self, session_id: str | None) -> list[Message]:
        if not self._session_manager or not session_id:
            return []
        history = self._session_manager.get_history(
            session_id,
            limit=self._config.session_history_limit,
        )
        messages: list[Message] = []
        for item in history:
            if item.role not in {"user", "assistant"}:
                continue
            if not item.content:
                continue
            messages.append(Message(role=item.role, content=item.content))
        return messages

    def _save_session_history(
        self,
        session_id: str | None,
        query: str,
        answer: str,
        completed: bool,
    ) -> None:
        if not self._session_manager or not session_id:
            return
        self._session_manager.append_history(
            session_id,
            "user",
            query,
            metadata={"type": "user_input"},
        )
        self._session_manager.append_history(
            session_id,
            "assistant",
            answer,
            metadata={"type": "final_answer", "completed": completed},
        )

    def _format_tools(self) -> str:
        specs = self._tool_registry.list()
        if not specs:
            return "No tools are registered."
        return "\n".join(
            f"- {spec.name}: {spec.description or 'No description.'}"
            for spec in specs
        )

    def _format_memories(self, user_id: str, query: str) -> str:
        if not self._memory_store:
            return "No memory store is configured."
        memories = self._memory_store.search(
            user_id,
            query,
            limit=self._config.memory_limit,
        )
        if not memories:
            return "No relevant memories."
        return "\n".join(f"- {record.content}" for record in memories)

    def _format_session(self, session_id: str | None) -> str:
        if not self._session_manager or not session_id:
            return "No session context."
        session = self._session_manager.get(session_id)
        return json.dumps(
            {"id": session.id, "status": str(session.status), "context": session.context},
            ensure_ascii=False,
        )

    @staticmethod
    def _message_to_dict(message: Message) -> dict[str, Any]:
        return {
            "role": message.role,
            "content": message.content,
            "name": message.name,
            "tool_call_id": message.tool_call_id,
            "metadata": message.metadata,
        }

    @classmethod
    def _parse_response(
        cls,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if tool_calls:
            normalized_calls = [cls._normalize_tool_call(call) for call in tool_calls]
            first_call = normalized_calls[0]
            return {
                "thought": tool_calls[0].get("thought", ""),
                "action": first_call.get("name"),
                "arguments": first_call.get("args", {}),
                "tool_call_id": first_call.get("id"),
                "tool_calls": normalized_calls,
            }

        payload = cls._extract_json_object(content)
        if payload is not None:
            return payload

        final_answer_match = re.search(
            r"final\s*answer\s*:\s*(?P<answer>.+)",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if final_answer_match:
            return {"final_answer": final_answer_match.group("answer").strip()}

        return {"final_answer": content.strip()}

    @staticmethod
    def _normalize_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
        function = tool_call.get("function", tool_call)
        arguments = (
            function.get("arguments")
            or function.get("args")
            or tool_call.get("args")
            or {}
        )
        if isinstance(arguments, str):
            arguments = json.loads(arguments or "{}")
        return {
            "id": tool_call.get("id") or function.get("id"),
            "name": function.get("name") or tool_call.get("name"),
            "args": arguments,
            "type": tool_call.get("type") or "tool_call",
        }

    @classmethod
    def _parsed_tool_calls(
        cls,
        parsed: dict[str, Any],
        session_id: str | None,
        step_index: int,
    ) -> list[dict[str, Any]]:
        raw_calls = parsed.get("tool_calls")
        if raw_calls:
            calls = [cls._normalize_tool_call(call) for call in raw_calls]
        elif parsed.get("action"):
            calls = [
                {
                    "id": parsed.get("tool_call_id"),
                    "name": parsed.get("action"),
                    "args": parsed.get("arguments", {}) or {},
                    "type": "tool_call",
                }
            ]
        else:
            return []

        normalized_calls: list[dict[str, Any]] = []
        for call_index, call in enumerate(calls, start=1):
            call_id = (
                call.get("id")
                or f"call_{session_id or uuid4()}_{step_index}_{call_index}"
            )
            normalized_calls.append(
                {
                    "id": call_id,
                    "name": call.get("name"),
                    "args": call.get("args", {}) or {},
                    "type": call.get("type") or "tool_call",
                }
            )
        return normalized_calls

    @staticmethod
    def _assistant_tool_calls(
        tool_calls: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        return list(tool_calls or [])

    @staticmethod
    def _extract_json_object(content: str) -> dict[str, Any] | None:
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
            stripped = re.sub(r"```$", "", stripped).strip()

        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            candidates = ReActAgent._extract_json_candidates(stripped)
            preferred = [
                item
                for item in candidates
                if isinstance(item, dict)
                and ("action" in item or "final_answer" in item)
            ]
            if preferred:
                return preferred[-1]
            if candidates and isinstance(candidates[-1], dict):
                return candidates[-1]
            return None

        if not isinstance(parsed, dict):
            return None
        return parsed

    @staticmethod
    def _extract_json_candidates(content: str) -> list[Any]:
        candidates: list[Any] = []
        start: int | None = None
        depth = 0
        in_string = False
        escaped = False
        for index, char in enumerate(content):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == "{":
                if depth == 0:
                    start = index
                depth += 1
                continue
            if char == "}":
                if depth == 0:
                    continue
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        candidates.append(json.loads(content[start : index + 1]))
                    except json.JSONDecodeError:
                        pass
                    start = None
        return candidates

    @staticmethod
    def _format_observation(tool_result: ToolCallResult) -> str:
        payload = {
            "tool": tool_result.name,
            "success": tool_result.success,
            "result": tool_result.result,
            "error": tool_result.error,
        }
        return f"Observation: {json.dumps(payload, ensure_ascii=False, default=str)}"
