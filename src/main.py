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
    최종 안정화 버전의 'Human-in-the-loop' AI 에이전트 실행 함수.
    'stream_mode="values"'에 맞춰 이벤트 처리 로직을 수정한 버전.
    """
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("🛑 OPENAI_API_KEY가 설정되지 않았습니다.")
        return

    notebook_filename = "persistent_agent_notebook.ipynb"
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
        app = create_agent_workflow(executor=executor)

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        # ✨ 수정된 부분: history는 세션 내내 유지됩니다.
        initial_state = {"notebook": notebook, "notebook_path": notebook_filename, "history": []}
        app.update_state(config, initial_state)

        print("\n🤖 AI 에이전트와의 대화를 시작합니다. 종료하려면 'exit' 또는 'quit'를 입력하세요.")

        while True:
            task = input("\n▶ 당신의 명령: ")
            if task.lower() in ["exit", "quit"]:
                print("👋 세션을 종료합니다.")
                break

            # ✨ 수정된 부분: history를 더 이상 초기화하지 않고 task만 전달합니다.
            events = app.stream({"task": task}, config, stream_mode="values")

            is_complex_task = False
            suggested_options = []

            print("\n--- 🚀 AI 에이전트 작업 시작 ---")
            for event in events:
                # ✨ --- 여기가 핵심 수정 부분 ---
                # 'event'는 상태 딕셔너리 그 자체입니다. 노드 이름을 추측할 필요 없이 직접 키를 확인합니다.

                # 'plan'이 새로 생겼다면 generator가 실행된 것입니다.
                if event.get("plan"):
                    print("\n✅ 다음 단계: [ generator ]", flush=True)
                    print("-" * 25, flush=True)
                    print(f"🤔 계획:\n{event['plan'][-1]}", flush=True)
                # 'executed_code'가 생겼다면 executor가 실행된 것입니다.
                elif event.get("executed_code"):
                    print("\n✅ 다음 단계: [ executor ]", flush=True)
                    print("-" * 25, flush=True)
                    if event.get("stdout"):
                        print(f"👀 STDOUT:\n{event['stdout']}", flush=True)
                    if event.get("stderr"):
                        print(f"🔥 STDERR:\n{event['stderr']}", flush=True)

                # 'suggester' 노드에서 중단되었는지 확인
                if event.get("suggested_options"):
                    is_complex_task = True
                    suggested_options = event["suggested_options"]
                    break

            # --- ✨ 여기까지 ---

            if not is_complex_task:
                print("\n--- 🎉 작업 완료 ---")
                continue

            if not suggested_options:
                print("😅 제안할 옵션이 없습니다. 다른 명령을 시도해주세요.")
                continue

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

            execution_events = app.stream({"task": selected_task}, config, stream_mode="values")
            for event in execution_events:
                # ✨ 수정된 부분: 재개 시에도 동일한 출력 로직 적용
                if event.get("plan"):
                    print("\n✅ 다음 단계: [ generator ]", flush=True)
                    print("-" * 25, flush=True)
                    print(f"🤔 계획:\n{event['plan'][-1]}", flush=True)
                elif event.get("executed_code"):
                    print("\n✅ 다음 단계: [ executor ]", flush=True)
                    print("-" * 25, flush=True)
                    if event.get("stdout"):
                        print(f"👀 STDOUT:\n{event['stdout']}", flush=True)
                    if event.get("stderr"):
                        print(f"🔥 STDERR:\n{event['stderr']}", flush=True)

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