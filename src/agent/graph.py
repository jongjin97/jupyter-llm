from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import code_generator_node, code_executor_node, option_suggester_node, router_node
from langgraph.checkpoint.memory import MemorySaver
from src.tools.jupyter_executor import JupyterExecutor
from functools import partial # ✨ partial 임포트


def has_critical_error(stderr: str) -> bool:
    """
    stderr 문자열에 해결이 필요한 심각한 오류가 있는지 판단하는 헬퍼 함수.
    단순 알림(notice) 등은 무시합니다.
    """
    if not stderr:
        return False

    # 무시할 문구 목록
    ignore_phrases = [
        "[notice]",
        "A new release of pip is available"
    ]
    if any(phrase in stderr for phrase in ignore_phrases):
        return False

    # 실제 오류를 나타내는 키워드 목록
    error_keywords = [
        "error", "traceback", "exception", "failed", "invalid"
    ]
    # 소문자로 변환하여 비교
    stderr_lower = stderr.lower()
    if any(keyword in stderr_lower for keyword in error_keywords):
        return True

    return False


def after_execution_router(state: AgentState) -> str:
    """
    executor 실행 후, 오류 발생 여부 및 작업 종류에 따라 다음 경로를 결정합니다.
    """
    stderr = state.get("stderr", "")
    print(f"stderr: {stderr}")
    # 1. 심각한 오류가 발생했는지 먼저 확인합니다.
    if has_critical_error(stderr):
        print("🔥 심각한 오류 감지. 수정 계획을 위해 generator로 돌아갑니다.")
        return "fix_error"

    # 2. 오류가 없다면, 기존 로직대로 작업 종류에 따라 분기합니다.
    # if state.get("task_type") == "simple_task":
    #     return "end"
    # else:
    return "end"

def should_continue(state: AgentState) -> str:
    """
    실행 결과를 보고 계속할지, 종료할지 결정하는 조건부 엣지(Conditional Edge) 함수입니다.

    Planner가 'FINISH'를 계획했다면 그래프를 종료('end')하고,
    그렇지 않다면 다음 계획을 위해 다시 planner로 돌아갑니다('continue').
    """
    if state["plan"][-1] == "FINISH":
        return "end"
    else:
        return "continue"

def create_agent_workflow(executor: JupyterExecutor):
    """
    AI 에이전트의 전체 작업 흐름을 정의하는 StateGraph를 생성하고 컴파일.
    """
    # 1. AgentState를 기반으로 그래프 객체를 생성
    workflow = StateGraph(AgentState)

    # partial을 사용하여 state 외에 executor를 추가 인자로 받는 새 함수 작성
    executor_with_tool = partial(code_executor_node, executor=executor)

    # 모든 노드를 등록합니다.
    workflow.add_node("router", router_node)
    workflow.add_node("suggester", option_suggester_node)
    workflow.add_node("generator", code_generator_node)
    workflow.add_node("executor", executor_with_tool)

    workflow.set_entry_point("router")

    # ✨ 라우터의 결정에 따라 흐름을 분기합니다.
    workflow.add_conditional_edges(
        "router",
        lambda state: state.get("destination"),  # state의 'destination' 키 값을 보고 판단
        {
            "simple_task": "generator",  # 'simple_task'이면 바로 generator로
            "complex_task": "suggester"  # 'complex_task'이면 suggester로
        }
    )

    # suggester가 끝나면 generator로 갑니다 (사용자 입력은 main.py에서 처리).
    workflow.add_edge("suggester", "generator")

    # generator가 코드를 만들면 executor가 실행합니다.
    workflow.add_edge("generator", "executor")

    # executor 실행 후, 조건분 라우팅 록직을 적용
    workflow.add_conditional_edges(
        "executor",
        after_execution_router,
        {
            "fix_error": "generator", # 오류 발생 시 -> 수정 코드를 위해 generator로 돌아갑니다.
            "end": END # 종료
        }
    )

    # executor가 실행을 마치면, 다시 라우터로 돌아가 다음 작업을 기다립니다.
    # (더 복잡한 로직에서는 executor가 끝나면 바로 suggester로 갈 수도 있습니다)
    # workflow.add_edge("executor", "router")

    # 인메모리 체크포인터 객체를 생성
    checkpointer = MemorySaver()

    # ✨ 중단 지점은 'suggester'로 유지합니다. (복잡한 작업일 경우에만 멈춤)
    app = workflow.compile(
        checkpointer=checkpointer,
        interrupt_after=["suggester"]
    )
    return app
