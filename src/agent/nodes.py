import nbformat
from nbformat.v4 import new_code_cell, new_output
from .state import AgentState
from pydantic import BaseModel, Field
# from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import List, Literal
from src.tools.jupyter_executor import JupyterExecutor


class SuggestedOptions(BaseModel):
    """A list of suggested next steps for the user to choose from."""
    options: List[str] = Field(description="A concise list of 3-5 logical next steps.")

class CodePlan(BaseModel):
    code: str = Field(description="Jupyter 커널에서 실행할 단일 Python 코드 블록.")
    reasoning: str = Field(description="이 코드가 주어진 작업을 어떻게 수행하는지에 대한 간략한 설명.")

class Route(BaseModel):
    """The routing decision for the user's task."""
    destination: Literal["simple_task", "complex_task"] = Field(description="The destination to route to based on task complexity.")
    task_type: Literal["file_system", "data_analysis", "visualization", "ml_engineering", "general"] = Field(description="The specific expertise required for the task.")

def router_node(state: AgentState) -> dict:
    """
    [역할: 총괄 매니저]
    사용자의 작업을 분석하여 '단순/복잡' 여부와 필요한 '전문가 유형'을 분류합니다.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert at classifying user requests for a Python coding agent. "
         "First, determine if the task is 'simple_task' (can be done in one obvious step) or 'complex_task' (is vague and needs user feedback). "
         "Second, classify the task into one of the following expertise types: "
         "- 'file_system': For tasks involving file or directory listing, reading, writing (os, glob, pathlib). "
         "- 'data_analysis': For tasks involving data manipulation, cleaning, and analysis (pandas, numpy). "
         "- 'visualization': For tasks involving plotting and creating charts (matplotlib, seaborn). "
         "- 'ml_engineering': For tasks involving machine learning model training and evaluation (scikit-learn). "
         "- 'general': For any other general Python coding task. "
         "Respond with both the destination and the task_type."),
        ("human", "User's task: {task}")
    ])
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    structured_llm = llm.with_structured_output(Route)

    route = structured_llm.invoke(prompt.format(task=state["task"]))

    # 다음 경로를 반환합니다. LangGraph는 이 값을 사용하여 분기합니다.
    return {"destination": route.destination, "task_type": route.task_type}

def option_suggester_node(state: AgentState) -> dict:
    """
    현재 상태(노트북 내용, 과거 기록 포함)를 종합적으로 분석하여
    사용자에게 다음에 수행할 작업 선택지를 제안합니다.
    """
    # ✨ --- 여기가 핵심 수정 부분 ---
    # planner처럼, 제안을 위해서도 충분한 맥락 정보를 수집합니다.

    # 최근 셀 내용 추출
    notebook_data = state.get("notebook")
    if isinstance(notebook_data, dict):
        notebook = nbformat.from_dict(notebook_data)
    else:
        notebook = notebook_data # 이미 객체인 경우 그대로 사용
    recent_cells_source = []
    if notebook and notebook.cells:
        num_recent_cells = 5
        for cell in notebook.cells[-num_recent_cells:]:
            if cell.cell_type == 'code':
                recent_cells_source.append(f"# Previous Code Cell:\n{cell.source}")
    formatted_recent_cells = "\n---\n".join(recent_cells_source)

    # 과거 작업 내역 요약
    formatted_history = "\n---\n".join(state.get("history", []))

    # 프롬프트에 수집한 모든 맥락 정보를 포함시킵니다.
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful data analysis assistant. Your job is to look at the current state of the analysis "
         "and suggest a list of logical next steps for the user to choose from. "
         "Review the user's request, the recent notebook cells, and the history to make relevant suggestions. "
         "Provide a concise list of 3-5 actionable options."),
        ("human",
         "--- User's Overall Task ---\n"
         "{task}\n\n"
         "--- Recent Notebook Cells ---\n"
         "{recent_cells}\n\n"
         "--- Analysis History ---\n"
         "{history}\n\n"
         "Based on all the information above, what are the best next steps for the user to choose from? Respond with a list of options.")
    ])

    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    structured_llm = llm.with_structured_output(SuggestedOptions)

    response = structured_llm.invoke(prompt.format(
        task=state['task'],
        recent_cells=formatted_recent_cells,
        history=formatted_history
    ))

    return {"suggested_options": response.options}


def code_generator_node(state: AgentState) -> dict:
    """
    사용자가 선택한 명확하고 구체적인 단일 작업을 Python 코드로 변환합니다.
    """
    # 1. 상태에서 필요한 모든 맥락 정보를 가져옵니다.
    #    이제 'task'는 "결측치 확인"과 같이 매우 구체적인 명령입니다.
    task = state["task"]
    notebook_data = state.get("notebook")

    stdout = state.get("stdout", "")
    stderr = state.get("stderr", "")

    # 체크포인터가 객체를 dict로 변환했을 수 있으므로, 다시 NotebookNode 객체로 복원합니다.
    if isinstance(notebook_data, dict):
        notebook = nbformat.from_dict(notebook_data)
    else:
        notebook = notebook_data # 이미 객체인 경우 그대로 사용

    # 최근 셀 내용 추출
    recent_cells_source = []
    if notebook and notebook.cells:
        num_recent_cells = 5
        for cell in notebook.cells[-num_recent_cells:]:
            if cell.cell_type == 'code':
                recent_cells_source.append(f"# Previous Code Cell:\n{cell.source}")
    formatted_recent_cells = "\n---\n".join(recent_cells_source)

    # 과거 작업 내역 요약
    formatted_history = "\n---\n".join(state.get("history", []))

    # 전문가 모드 결정
    task_type = state.get("task_type", "general")

    # 각 전문가 모드에 맞는 시스템 프롬프트를 정의합니다.
    expert_prompts = {
        "file_system": "You are a Python expert specializing in file system operations. Use `os`, `glob`, and `pathlib` to handle file and directory tasks efficiently and safely.",
        "data_analysis": "You are a senior data analyst. Your expertise is in using `pandas` and `numpy` for data manipulation, cleaning, aggregation, and analysis. Always aim for idiomatic pandas code.",
        "visualization": "You are a data visualization specialist. Use `matplotlib` and `seaborn` to create clear and insightful charts. **CRITICAL: You MUST execute `%matplotlib inline` before any plotting commands.**",
        "ml_engineering": "You are a machine learning engineer. Your specialty is using `scikit-learn` to build preprocessing pipelines, train models, and evaluate their performance. Use standard variable names like `X_train`, `y_train`.",
        "general": "You are a general-purpose, highly skilled Python code generation tool. Write clean, efficient, and correct Python code to accomplish the given task."
    }

    # 선택된 전문가 모드에 맞는 시스템 프롬프트를 가져옵니다.
    system_prompt = expert_prompts.get(task_type, expert_prompts["general"])
    # 역할이 단순화된 만큼, 프롬프트도 훨씬 더 명확하고 간결해집니다.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             f"{system_prompt}"
             "\n\n--- YOUR WORKFLOW & RULES ---\n"
             "1. **Analyze & Plan:** Review all context and the `Task To Execute Now`."
             "2. **Code Generation:** Write the Python code to accomplish the task."
             "3. **Self-Testing (CRITICAL):** After writing the main logic (like a function or a complex transformation), you MUST add a few lines of simple test code (`assert` or `print` checks) to verify that your code works as expected. This helps catch errors early."
             "   - *Example:* If you create a function `def add(a, b): ...`, you should add `assert add(3, 5) == 8` afterwards."
             "4. **Error Handling:** If the previous step had an error (`STDERR` is not empty), your only goal is to fix that error."
             "\n\n--- OTHER RULES ---\n"
             " - If a library is needed, `!pip install` it."
             " - If you need to plot, execute `%matplotlib inline` first."),
            ("human",
             "--- Context: Recent Notebook Cells ---\n"
             "{recent_cells}\n\n"
             "--- Context: History of Past Actions ---\n"
             "{history}\n\n"
             # ✨ 수정된 부분: 직전 실행 결과를 전달하는 섹션 추가
             "--- Context: Result of Last Execution ---\n"
             "STDOUT:\n{stdout}\n\n"
             "STDERR:\n{stderr}\n\n"
             "--- **Task To Execute Now** ---\n"
             "**{task}**\n\n"
             "Please write the single block of Python code to perform your task based on your workflow.")
        ]
    )

    # 3. LLM을 호출하여 코드를 생성합니다.
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    structured_llm = llm.with_structured_output(CodePlan)

    response = structured_llm.invoke(prompt.format(
        task=task,
        recent_cells=formatted_recent_cells,
        history=formatted_history,
        stdout=stdout,
        stderr=stderr
    ))

    # 4. 생성된 코드를 'plan'으로 반환하여 executor에게 전달합니다.
    return {"plan": [response.code]}


def code_executor_node(state: AgentState, executor: JupyterExecutor):
    """
    코드를 실행하고, 실행 내역을 노트북에 기록한 뒤 저장합니다.
    """
    code_to_run = state['plan'][-1]

    if code_to_run == "FINISH":
        return {"executed_code": "FINISH", "stdout": "Task completed."}

    # 1. 상태에서 노트북 객체와 경로를 가져옵니다.
    notebook_data = state['notebook']

    if isinstance(notebook_data, dict):
        notebook = nbformat.from_dict(notebook_data)
    else:
        notebook = notebook_data

    notebook_path = state['notebook_path']

    # 2. 노트북에 새로운 코드 셀을 추가합니다. (기록)
    cell = new_code_cell(code_to_run)
    notebook.cells.append(cell)

    # 3. 코드를 실행합니다.
    # executor = state['kernel_executor']
    result = executor.execute(code_to_run)

    # stdout 결과가 잇다면, name='stdout'인 stream 객체를 만들어 추가
    if result['stdout']:
        stdout_output = new_output(
            output_type="stream",
            name="stdout",
            text=result['stdout']
        )
        cell.outputs.append(stdout_output)

    # 리치 출력(이미지 등)을 추가
    if result['outputs']:

        for output_content in result['outputs']:
            # nbformat이 요구하는 'data', 'metadata' 형식을 그대로 전달
            cell.outputs.append(new_output(
                output_type=output_content.get('header', {}).get('msg_type', 'display_data'),
                data=output_content.get('data', {}),
                metadata=output_content.get('metadata', {})
            ))

    # stderr 결과가 있다면, name='stderr'인 stream 객체를 만들어 추가
    if result['stderr']:
        stderr_output = new_output(
            output_type="stream",
            name="stderr",
            text=result['stderr']
        )
        cell.outputs.append(stderr_output)

    # 추가 실행 결과를 'Output' 객체로 만듭니다.
    # 가장 일반적인 'stream' 타입의 출력으로 만듭니다.
    # output = new_output(
    #     output_type="stream",
    #     name="stdout", # 표준 출력
    #     text=observation, # executor가 반환한 결과 문자열
    # )
    # cell.outputs.append(output)
    # 4. 변경된 노트북 객체를 파일에 다시 씁니다. (저장)
    try:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(notebook, f)
    except Exception as e:
        result["stderr"] += f"\n\n경고: 노트북 파일 저장 실패 - {e}"

    history = state.get("history", [])
    summary = f"Executed Code:\n```python\n{code_to_run}\n```\n\nSTDOUT:\n{result['stdout']}\n\nSTDERR:\n{result['stderr']}"
    history.append(summary)

    # 5, 상태를 업데이트하여 반환합니다.
    return {
        "executed_code": code_to_run,
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "notebook": notebook,
        "history": history
    }