# R.A.I.N.-tools

Python companion package for R.A.I.N. Lab — LangGraph-based tool calling for consistent LLM agent execution.

## Installation

```bash
pip install R.A.I.N.-tools
```

## Usage

```python
from R.A.I.N._tools import create_agent, shell, file_read, file_write

agent = create_agent(
    tools=[shell, file_read, file_write],
    model="local",
    api_key="",
)
```

## CLI

```bash
R.A.I.N.-tools "Your message here"
R.A.I.N.-tools --interactive
```