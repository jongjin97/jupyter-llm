from typing import TypedDict, List
from nbformat import NotebookNode
from src.tools.jupyter_executor import JupyterExecutor

class AgentState(TypedDict):
    task: str
    plan: List[str]
    executed_code: str
    observation: str
    kernel_executor: JupyterExecutor
    notebook_path: str
    notebook: NotebookNode