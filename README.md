# Agent Scaffold

一个用于 agent 开发的 Python 项目脚手架。

## Core Modules

```text
src/agent_scaffold/
  llm_call/       # 大模型执行模块，支持动态注册与替换 provider
  tool_register/  # 工具注册、工具查询、LLM tool schema 导出
  tool_call/      # 本地工具调用，支持同步/异步工具与统一结果封装
  memory/         # 以 user_id 关联的记忆存储与查询
  session/        # 会话状态机、上下文管理、会话数据存取
  context_trim/   # LLM 调用前的上下文裁剪与压缩边界
  process/        # Agent 业务流程编排，包含 ReAct loop
  trace.py        # JSONL 事件追踪日志
```

## Install Dependencies

```powershell
pip install -r requirements.txt
```

## Quick Example

```python
from agent_scaffold.llm_call import LLMClient, Message, MiniMaxLLMProvider
from agent_scaffold.memory import InMemoryStore
from agent_scaffold.tool_call import ToolExecutor
from agent_scaffold.tool_register import ToolRegistry

llm = LLMClient()
llm.register_provider("minimax", MiniMaxLLMProvider(), set_default=True)
response = llm.call([Message(role="user", content="hello")])

registry = ToolRegistry()
registry.register("add", lambda a, b: a + b, description="Add two numbers.")
executor = ToolExecutor(registry)
result = executor.call("add", {"a": 1, "b": 2})

memory = InMemoryStore()
memory.save("user-1", "The user prefers concise answers.")
user_memories = memory.search("user-1", "concise")
```

## ReAct Process Hooks

`ReActAgent` 支持可替换的上下文裁剪器、会话历史和 trace 日志：

```python
from agent_scaffold.context_trim import PassthroughContextTrimmer
from agent_scaffold.process import ReActAgent, ReActConfig

agent = ReActAgent(
    llm_client=llm,
    tool_registry=registry,
    context_trimmer=PassthroughContextTrimmer(),
    config=ReActConfig(trace_log_path="logs/react_trace.jsonl"),
)
```

## Run Tests

```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s tests
```

## MiniMax Provider

MiniMax 走 OpenAI-compatible API。先安装依赖并设置环境变量：

```powershell
pip install -r requirements.txt
$env:MINIMAX_API_KEY="your-api-key"
```

```python
from agent_scaffold.llm_call import LLMClient, Message, MiniMaxLLMProvider

client = LLMClient()
client.register_provider("minimax", MiniMaxLLMProvider(), set_default=True)

response = client.call([
    Message(role="system", content="你是一个有帮助的助手。"),
    Message(role="user", content="现在是几点，请讲一个关于程序员的冷笑话"),
])
print(response.content)
```
