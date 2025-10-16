from typing import TypedDict, List
from nbformat import NotebookNode
from src.tools.jupyter_executor import JupyterExecutor
from typing import TypedDict, List, Literal

class AgentState(TypedDict):
    """
    LangGraph 에이전트의 전체 상태(메모리)를 정의합니다.
    """
    # 기본 작업 정보
    task: str
    plan: List[str]
    executed_code: str

    # 실행 결과
    stdout: str
    stderr: str

    # 노트북 및 커널 정보
    # kernel_executor: JupyterExecutor
    notebook_path: str
    notebook: NotebookNode

    # 에이전트의 장기 기억
    history: List[str]

    suggested_options: List[str]

    # 라우터가 결정한 작업 종류와 전무가 모드를 저장
    destination: str
    task_type: Literal["file_system", "data_analysis", "visualization", "ml_engineering", "general"]