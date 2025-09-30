import os
import sys
import traceback
import nbformat
from dotenv import load_dotenv
import uuid

from src.tools.jupyter_executor import JupyterExecutor
from src.agent.graph import create_agent_workflow
from src.agent.state import AgentState


def main():
    """
    'Human-in-the-loop' AI 에이전트의 전체 실행을 관장하는 메인 함수.
    LangGraph의 내장 상태 관리(체크포인트)를 사용하여 안정성을 높인 최종 버전.
    """
    # --- 1. 초기 설정 ---
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("🛑 오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        return

    notebook_filename = "titanic_notebook_v7.ipynb"
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
        # --- 2. 리소스 및 에이전트 준비 ---
        executor = JupyterExecutor()
        app = create_agent_workflow(executor=executor)

        # ✨ 수정된 부분: 대화의 전체 생명주기를 관리할 고유 ID와 config 생성
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        # 최초 상태를 LangGraph의 체크포인트 시스템에 업데이트합니다.
        initial_state = {
            "notebook": notebook,
            "notebook_path": notebook_filename,
            # "kernel_executor": executor,
            "history": []
        }
        app.update_state(config, initial_state)

        print("\n🤖 AI 에이전트와의 대화를 시작합니다. 종료하려면 'exit' 또는 'quit'를 입력하세요.")

        # --- 3. 대화형 세션 루프 ---
        while True:
            task = input("\n▶ 당신의 명령: ")
            if task.lower() in ["exit", "quit"]:
                print("👋 세션을 종료합니다.")
                break

            print("\n--- 🤔 옵션 제안 중... ---")

            # ✨ 1단계: 'suggester' 노드까지 실행하여 옵션 제안받기
            events = app.stream({"task": task, "history": [], "suggested_options": []}, config, stream_mode="values")
            suggester_output_state = None
            for event in events:
                print(f"event: {event}")
                if event.get("suggested_options"):
                    suggester_output_state = event
                    break  # 옵션이 생성되었으면 루프 중단

            # ✨ 2단계: 사용자에게 옵션 보여주고 선택받기
            if not suggester_output_state:
                print("😅 제안할 옵션을 생성하지 못했습니다. 다른 명령을 시도해주세요.")
                continue

            suggested_options = suggester_output_state.get("suggested_options", [])
            print("\n🤔 다음 중 어떤 작업을 수행할까요?")
            for i, opt in enumerate(suggested_options):
                print(f"  [{i + 1}] {opt}")
            print("  [0] 직접 입력 (또는 취소)")

            choice = input("▶ 선택: ")
            selected_task = ""
            if choice == "0":
                selected_task = input("▶ 직접 입력: ")
            else:
                try:
                    selected_task = suggested_options[int(choice) - 1]
                except (ValueError, IndexError):
                    print("잘못된 선택입니다.")

            if not selected_task:
                print("작업이 취소되었습니다.")
                continue

            print(f"\n--- 🚀 '{selected_task}' 작업 시작 ---")

            # ✨ 3단계: 사용자의 선택으로 그래프 실행 '재개'하기
            # 동일한 config를 사용하면 LangGraph가 알아서 중단된 지점부터 실행을 이어갑니다.
            for event in app.stream({"task": selected_task}, config, stream_mode="values"):
                step_name = list(event.keys())[0]
                print(f"\n✅ 다음 단계: [ {step_name} ]", flush=True)

            print("\n--- 🎉 작업 완료 ---")

    except Exception as e:
        print(f"\n🛑 에이전트 실행 중 오류가 발생했습니다: {e}")
        traceback.print_exc(file=sys.stdout)
    finally:
        if executor:
            print("\n--- 셧다운 ---")
            executor.shutdown()


if __name__ == "__main__":
    main()