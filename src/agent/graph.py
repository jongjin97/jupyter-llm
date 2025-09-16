from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import planner_node, code_executor_node

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

def create_agent_workflow():
    """
    AI 에이전트의 전체 작업 흐름을 정의하는 StateGraph를 생성하고 컴파일.
    """
    # 1. AgentState를 기반으로 그래프 객체를 생성
    workflow = StateGraph(AgentState)

    # 2. 그래프에 노드들을 추가
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", code_executor_node)

    # 3. 그래프의 시작점(Entry Point)을 'planner' 노드로 설정
    workflow.set_entry_point("planner")

    # 4. 노드 간의 일반 엣지(Edge)를 추가
    # 'planner' 노드가 끝나면 항상 'executor' 노드로 이동
    workflow.add_edge("planner", "executor")

    # 5. 조건부 엣지(Conditional Edge)를 추가
    # 'executor' 노드가 끝난 후에는 should_continue 함수를 호출하여 그 반환값에 따라 다음 노드를 결정.
    # - 'continue'가 반환되면 -> 'planner' 노드로 이동 (루프)
    # - 'end'가 반환되면 -> 그래프 종료 (END)
    workflow.add_conditional_edges(
        "executor",
        should_continue,
        {
            "continue": "planner",
            "end": END,
        },
    )

    # 6. 정의된 워크플로우를 컴파일하여 실행 가능한 app으로 만듬.
    app = workflow.compile()
    return app
