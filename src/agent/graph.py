from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import code_generator_node, code_executor_node, option_suggester_node, router_node, error_classifier_node
from langgraph.checkpoint.memory import MemorySaver
from src.tools.jupyter_executor import JupyterExecutor
from functools import partial


# 1차 검사 (Python): stderr에 내용이 있는지 확인
def check_for_stderr(state: AgentState) -> str:
    """
    [Edge] 1차 검사: executor 실행 후, 'stderr'에 내용이 있는지 확인합니다.
    """
    stderr = state.get("stderr", "")
    if not stderr:
        return "no_error"  # stderr가 비어있으면 바로 '오류 없음' 경로로

    # pip 알림처럼 명백히 무시할 수 있는 경고는 AI 호출 없이 바로 처리
    ignore_phrases = ["[notice]", "A new release of pip is available"]
    stderr_lines = stderr.strip().split('\n')
    if all(any(phrase in line for phrase in ignore_phrases) for line in stderr_lines if line.strip()):
        print("✅ 단순 pip 알림 감지. AI 호출 없이 계속합니다.")
        return "no_error"

    return "check_error_critically"  # 그 외 내용이 있으면 'AI 심판'에게 보냄

# 2차 검사 (AI): AI 심판의 결정을 바탕으로 경로 결정
def after_error_classifier_router(state: AgentState) -> str:
    """
    [Edge] 2차 검사: AI 심판의 결정을 바탕으로 경로를 결정합니다.
    """
    if state.get("destination") == "fix_error":
        return "fix_error"
    else:
        # AI가 오류가 아니라고 판단했으므로, 3차 검사로 이동
        return "no_error"

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
    workflow.add_node("error_classifier", error_classifier_node)
    workflow.set_entry_point("router")

    # 라우터의 결정에 따라 흐름을 분기합니다.
    workflow.add_conditional_edges(
        "router",
        lambda state: state.get("destination"),  # state의 'destination' 키 값을 보고 판단
        {
            "simple_task": "generator",  # 'simple_task'이면 바로 generator로
            "complex_task": "suggester"  # 'complex_task'이면 suggester로
        }
    )

    # Executor 실행 후 1차 검사 (Python)
    workflow.add_conditional_edges(
        "executor",
        check_for_stderr,  # 1차 검사
        {
            "check_error_critically": "error_classifier",  # 오류가 의심되면 AI 심판에게
            "no_error": END
        }
    )
    # AI 심판 실행 후 2차 검사 (AI의 결정)
    workflow.add_conditional_edges(
        "error_classifier",
        after_error_classifier_router,  # 2차 검사
        {
            "fix_error": "generator",  # 치명적 오류 -> 수정하러 감
            "no_error": END
        }
    )
    # suggester가 끝나면 generator로 갑니다 (사용자 입력은 main.py에서 처리).
    workflow.add_edge("suggester", "generator")

    # generator가 코드를 만들면 executor가 실행합니다.
    workflow.add_edge("generator", "executor")

    # 인메모리 체크포인터 객체를 생성
    checkpointer = MemorySaver()

    # 중단 지점은 'suggester'로 유지합니다. (복잡한 작업일 경우에만 멈춤)
    app = workflow.compile(
        checkpointer=checkpointer,
        interrupt_after=["suggester"]
    )
    return app
