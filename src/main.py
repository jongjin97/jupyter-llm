import os
import nbformat
from dotenv import load_dotenv

from src.tools.jupyter_executor import JupyterExecutor
from src.agent.graph import create_agent_workflow
from src.agent.state import AgentState

def main():
    """
    AI 에이전트의 전체 실행을 관장하는 메인 함수.
    """

    # 1. .env 파일에서 환경 변수 (API 키 등)을 로드합니다.
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("🛑 오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("프로젝트 루트에 .env 파일을 만들고 API 키를 추가해주세요.")
        return

    # 2. 리소스(커널)을 안전하게 관리하기 위해 try... finally 구문을 사용
    executor = None
    notebook_filename = "session_notebook.ipynb"
    try:
        # 3. 사용자로부터 수행할 작업을 입력받습니다.
        task = input("안녕하세요! 어떤 작업을 도와드릴까요? -> ")

        # 4. JupyterExecutor를 생성합니다.
        executor = JupyterExecutor(create_notebook_on_start=notebook_filename)

        # 5. 방금 생성된 초기 노트북 파일을 읽어 LangGraph의 상태 객체로 만듭니다.
        with open(notebook_filename, 'r', encoding='utf-8') as f:
            initial_notebook = nbformat.read(f, as_version=4)

        initial_state: AgentState = {
            "task": task,
            "notebook": initial_notebook,
            "notebook_path": notebook_filename,
            "kernel_executor": executor,
            "plan": [],
            "executed_code" : "",
            "observation": "",
        }

        # LangGraph 워크플로우를 생성(컴파일)합니다.
        app = create_agent_workflow()

        print("\n--- 🚀 AI 에이전트 작업을 시작합니다 ---")
        for s in app.stream(initial_state, {"recursion_limit": 15}):
            step_name = list(s.keys())[0]
            state_update = list(s.values())[0]

            print(f"\n✅ 다음 단계: [ {step_name} ]")
            print("-" * 25)

            if "plan" in state_update and state_update["plan"]:
                print(f"🤔 계획:\n{state_update['plan'][-1]}")

            if "observation" in state_update:
                print(f"👀 관찰 결과:\n{state_update['observation']}")

        print("\n--- 🎉 모든 작업이 완료되었습니다 ---")

    except Exception as e:
        print(f"\n🛑 에이전트 실행 중 오류가 발생했습니다: {e}")

    finally:
        # 9. 작업이 성공하든, 실패하든, 항상 커널을 안전하게 종료합니다.
        if executor:
            print("\n--- 셧다운 ---")
            executor.shutdown()


if __name__ == "__main__":
    main()