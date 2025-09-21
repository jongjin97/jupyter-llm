import os
import sys

import nbformat
from dotenv import load_dotenv
import traceback
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

    # 2. 파일이 존재하는지 확인하고, 없으면 새로 생성합니다.
    notebook_filename = "persistent_notebook.ipynb"
    try:
        with open(notebook_filename, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
            print(f"📖 기존 노트북 '{notebook_filename}'을 불러왔습니다.")
    except FileNotFoundError:
        notebook = nbformat.v4.new_notebook()
        with open(notebook_filename, 'w', encoding='utf-8') as f:
            nbformat.write(notebook, f)
        print(f"📄 새 노트북 '{notebook_filename}'을 생성했습니다.")

    executor = None
    try:
        executor = JupyterExecutor()
        app = create_agent_workflow()
        # 사용자로부터 수행할 작업을 입력받습니다.

        current_state: AgentState = {
            "task": "",
            "notebook": notebook,
            "notebook_path": notebook_filename,
            "kernel_executor": executor,
            "plan": [],
            "executed_code" : "",
            "stderr": "",
            "stdout": "",
        }

        print("\n🤖 AI 에이전트와의 대화를 시작합니다. 종료하려면 'exit' 또는 'quit'를 입력하세요.")
        while True:
            # 4. 사용자로부터 새로운 작업을 입력받습니다.
            task = input("\n▶ 당신의 명령: ")
            if task.lower() in ["exit", "quit"]:
                print("👋 세션을 종료합니다.")
                break
            # 5. 상태 업데이트: 현재 상태에 새로운 task를 추가합니다.
            current_state["task"] = task
            print("\n--- 🚀 AI 에이전트 작업 시작 ---")
            # 6. 현재 상태를 가지고 에이전트를 실행합니다.
            for s in app.stream(current_state, {"recursion_limit": 15}):
                step_name = list(s.keys())[0]
                state_update = list(s.values())[0]
                print(f"\n✅ 다음 단계: [ {step_name} ]", flush=True)
                print("-" * 25, flush=True)
                if "plan" in state_update and state_update["plan"]:
                    print(f"🤔 계획:\n{state_update['plan'][-1]}", flush=True)
                if "stdout" in state_update and state_update["stdout"]:
                    print(f"👀 STDOUT:\n{state_update['stdout']}", flush=True)
                if "stderr" in state_update and state_update["stderr"]:
                    print(f"🔥 STDERR:\n{state_update['stderr']}", flush=True)

                # 7. 상태 업데이트: 작업이 끝난 후의 최종 상태를 저장합니다.
                #    이렇게 하면 다음 루프에서 이전 작업 내용을 기억할 수 있습니다.
                # print(s[step_name])
                current_state['step_name'] = s[step_name]
                print("\n--- 🎉 작업 완료 ---")

    except Exception as e:
        print(f"\n🛑 에이전트 실행 중 오류가 발생했습니다: {e}")
        traceback.print_exc(file=sys.stdout)
    finally:
        if executor:
            executor.shutdown()


if __name__ == "__main__":
    main()