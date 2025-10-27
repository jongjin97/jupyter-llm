import os
import sys
import traceback
import nbformat
from dotenv import load_dotenv
import uuid

# --- CLI UI 라이브러리 임포트 ---
import pyfiglet
from rich.console import Console
from rich.panel import Panel
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.exceptions import InvalidArgument
# --- -------------------------- ---

from src.tools.jupyter_executor import JupyterExecutor
from src.agent.graph import create_agent_workflow
from src.agent.state import AgentState


# --- 헬퍼 함수 (show_option_menu) ---
def show_option_menu(options: list, console: Console) -> str | None:
    if not options:
        console.print("😅 제안할 수 있는 이전 옵션이 없습니다.", style="yellow")
        return None
    console.print("\n🤔 다음 중 어떤 작업을 수행할까요?")
    choices = [Choice(opt, name=opt) for opt in options]
    choices.append(Choice("direct_input", name="[ 직접 입력 ]"))
    choices.append(Choice(None, name="[ 취소 ]"))
    try:
        selected_task = inquirer.select(
            message="작업을 선택하세요:", choices=choices,
            default=choices[0].value if options else None,
            vi_mode=True, qmark="❓", amark="✔",
        ).execute()
    except (KeyboardInterrupt, InvalidArgument):
        console.print("\n작업이 취소되었습니다.", style="yellow")
        return None
    if selected_task == "direct_input":
        selected_task = console.input("▶ [bold cyan]직접 입력[/bold cyan]: ")
    if not selected_task:
        console.print("작업이 취소되었습니다.", style="yellow")
        return None
    return selected_task


# --- 헬퍼 함수 (run_execution_graph) ---
def run_execution_graph(app, config, task_to_run, session_history, previous_event, printed_plan=""):
    console = Console()
    console.print(f"\n--- 🚀 '{task_to_run}' 작업 시작 ---", style="bold yellow")

    execution_events = app.stream({"task": task_to_run, "history": session_history}, config, stream_mode="values")

    plan_just_printed = False

    for event in execution_events:
        if event.get("plan") and not plan_just_printed and printed_plan != event['plan'][-1]:
            printed_plan = event.get("plan")[-1]
            console.print(f"\n✅ 다음 단계: [ [bold magenta]generator[/bold magenta] ]")
            console.print(Panel(event['plan'][-1], title="🤔 계획", border_style="magenta", title_align="left"))
            plan_just_printed = True

        elif event.get("executed_code") and event.get("executed_code") != previous_event.get("executed_code"):
            console.print(f"\n✅ 다음 단계: [ [bold magenta]executor[/bold magenta] ]")
            if event.get("stdout"):
                console.print(Panel(event['stdout'], title="👀 STDOUT", border_style="green", title_align="left"))
            if event.get("stderr"):
                console.print(Panel(event['stderr'], title="🔥 STDERR", border_style="red", title_align="left"))

            if event.get("history"):
                session_history.append(event["history"][-1])
            plan_just_printed = False

        previous_event = event.copy()  # ✨ copy()로 수정

    console.print("\n--- 🎉 작업 완료 ---", style="bold green")

    return printed_plan


# --- 메인 함수 ---
def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("🛑 OPENAI_API_KEY가 설정되지 않았습니다.")
        return

    console = Console()
    notebook_filename = "persistent_agent_notebook.ipynb"
    try:
        with open(notebook_filename, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
            console.print(f"📖 기존 노트북 '{notebook_filename}'을 불러왔습니다.", style="green")
    except FileNotFoundError:
        notebook = nbformat.v4.new_notebook()
        with open(notebook_filename, 'w', encoding='utf-8') as f:
            nbformat.write(notebook, f)
        console.print(f"📄 새 노트북 '{notebook_filename}'을 생성했습니다.", style="yellow")

    executor = None
    try:
        executor = JupyterExecutor()
        app = create_agent_workflow(executor=executor)

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        initial_state = {"notebook": notebook, "notebook_path": notebook_filename, "history": []}
        app.update_state(config, initial_state)

        logo_text = pyfiglet.figlet_format("AI Code Agent", font="slant")
        console.print(Panel(logo_text, title="🚀 Interactive AI Code Agent for Jupyter 🚀", border_style="bold blue"))
        console.print("\n🤖 AI 에이전트와의 대화를 시작합니다.", style="bold")

        session_history = []
        last_suggested_options = []
        printed_plan = ""
        previous_event = {}
        while True:
            console.print("\n" + "=" * 50, style="bold dim")
            try:
                main_choice = inquirer.select(
                    message="무엇을 하시겠습니까?",
                    choices=[
                        Choice("new", name="[1] 새 작업 지시하기"),
                        Choice("previous", name="[2] 이전 제안 목록에서 작업 실행", enabled=last_suggested_options),
                        Choice("exit", name="[3] 종료"),
                    ],
                    default="new",
                    qmark="▶",
                    amark="✔",
                ).execute()
            except (KeyboardInterrupt, InvalidArgument):
                main_choice = "exit"

            if main_choice == "exit":
                console.print("👋 세션을 종료합니다.", style="bold yellow")
                break

            selected_task_for_execution = None

            if main_choice == "new":
                task = console.input("\n▶ [bold cyan]당신의 명령[/bold cyan]: ")
                if not task:
                    console.print("작업이 취소되었습니다.", style="yellow")
                    continue

                # ✨ 수정: 새 작업 시, 'session_history'를 전달하고 'suggested_options'만 초기화합니다.
                input_data = {"task": task, "history": session_history, "suggested_options": []}
                events = app.stream(input_data, config, stream_mode="values")

                is_complex_task = False
                plan_just_printed = False

                console.print("\n--- 🚀 AI 에이전트 작업 시작 ---", style="bold yellow")
                for event in events:
                    if event.get("plan") and not plan_just_printed and printed_plan != event['plan'][-1]:
                        printed_plan = event['plan'][-1]
                        console.print(f"\n✅ 다음 단계: [ [bold magenta]generator[/bold magenta] ]")
                        console.print(
                            Panel(event['plan'][-1], title="🤔 계획", border_style="magenta", title_align="left"))
                        plan_just_printed = True

                    elif event.get("executed_code") and event.get("executed_code") != previous_event.get(
                            "executed_code"):
                        console.print(f"\n✅ 다음 단계: [ [bold magenta]executor[/bold magenta] ]")
                        if event.get("stdout"):
                            console.print(
                                Panel(event['stdout'], title="👀 STDOUT", border_style="green", title_align="left"))
                        if event.get("stderr"):
                            console.print(
                                Panel(event['stderr'], title="🔥 STDERR", border_style="red", title_align="left"))

                        # ✨ 수정: (단순 작업도) history를 바로 누적
                        if event.get("history"):
                            session_history.append(event["history"][-1])
                        plan_just_printed = False

                    if event.get("suggested_options"):
                        is_complex_task = True
                        last_suggested_options = event.get("suggested_options", [])
                        break

                    previous_event = event.copy()  # ✨ copy()로 수정

                if not is_complex_task:
                    console.print("\n--- 🎉 작업 완료 ---", style="bold green")
                    continue
                else:
                    selected_task_for_execution = show_option_menu(last_suggested_options, console)

            elif main_choice == "previous":
                selected_task_for_execution = show_option_menu(last_suggested_options, console)

            if not selected_task_for_execution:
                console.print("작업이 취소되었습니다.", style="yellow")
                continue

            printed_plan = run_execution_graph(app, config, selected_task_for_execution, session_history, previous_event, printed_plan)

    except Exception as e:
        console.print(f"\n🛑 에이전트 실행 중 심각한 오류가 발생했습니다.", style="bold red")
        console.print_exception(show_locals=False)
    finally:
        if executor:
            console.print("\n--- 셧다운 ---", style="dim")
            executor.shutdown()


if __name__ == "__main__":
    main()