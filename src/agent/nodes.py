import nbformat
from nbformat.v4 import new_code_cell, new_output
from .state import AgentState
from pydantic import BaseModel, Field
# from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class CodePlan(BaseModel):
    """실행할 코드 계획"""
    code: str = Field(
        description="Jupyter 커널에서 실행할 파이썬 코드. 라이브러리 설치가 필요하면 `!pip install ...`을 포함해야 함. 모든 작업이 끝났으면 'FINISH'를 반환."
    )
    reasoning: str = Field(
        description="이 코드를 실행해야 하는 이유에 대한 간략한 설명."
    )

def planner_node(state: AgentState) -> dict:
    """
    LLM을 사용하여 사용자의 요청과 이전 실행 결과를 바탕으로 다음 단계를 계획합니다.
    (AI 에이전트의 '두뇌' 역할)
    """
    # 상태에서 노트북 객체를 가져와 최근 셀들의 내용을 추출합니다.
    notebook = state.get("notebook")
    recent_cells_source = []
    if notebook and notebook.cells:
        # 마지막 3개의 셀 내용만 가져옵니다.
        num_recent_cells = 3
        for cell in notebook.cells[-num_recent_cells:]:
            if cell.cell_type == 'code':
                # 보기 좋게 형식화하여 리스트에 추가
                recent_cells_source.append(f"# Previous Code Cell:\n{cell.source}")

    # 리스트를 하나의 문자열로 합칩니다.
    formatted_recent_cells = "\n---\n".join(recent_cells_source)
    # --- ✨ 여기까지 ---


    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are an expert Python programmer and a helpful AI assistant operating within a Jupyter Notebook environment. "
             "Your goal is to complete the user's task by planning and executing Python code. "
             "Review the history of previously executed code and its observations to decide the next step. "
             "If you encounter a ModuleNotFoundError, you must generate a `!pip install` command to install the missing library. "
             "When the user's entire task is complete, return 'FINISH' in the code field to end the session."
             "\n\n--- IMPORTANT RULES ---\n"
             "1. If the user asks for a plot, visualization, or graph using matplotlib, you MUST execute `%matplotlib inline` as the very first step before any other code."
             "2. After setting the backend, you can proceed with importing libraries and generating the plot code."),
            ("human",
             "User's Task: {task}\n\n"
             "--- RECENT NOTEBOOK CELLS ---\n"
             "{recent_cells}\n\n" 
             "Previously Executed Code:\n```python\n{executed_code}\n```\n\n"
             "Output from last execution (STDOUT):\n{stdout}\n\n"
             "Error from last execution (STDERR):\n{stderr}\n\n"
             "Based on the above, what is the next single step of code to execute? If there was an error in STDERR, try to fix it."),
        ]
    )
    # 사용할 언어 모델 정의
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

    # LLM이 CodePlan 구조에 맞춰 출력하도록 설정
    structured_llm = llm.with_structured_output(CodePlan)

    # 프롬프트를 채워서 LLM 호출
    response = structured_llm.invoke(prompt.format(
        task=state["task"],
        recent_cells=formatted_recent_cells,
        executed_code=state["executed_code"],
        stdout=state["stdout"],
        stderr=state["stderr"],
    ))

    # 결정된 계획을 상태에 업데이트하여 반환
    return {"plan": [response.code]}


def code_executor_node(state: AgentState):
    """
    코드를 실행하고, 실행 내역을 노트북에 기록한 뒤 저장합니다.
    """
    code_to_run = state['plan'][-1]

    if code_to_run == "FINISH":
        return {"stdout": "작업 완료."}

    # 1. 상태에서 노트북 객체와 경로를 가져옵니다.
    notebook = state['notebook']
    notebook_path = state['notebook_path']

    # 2. 노트북에 새로운 코드 셀을 추가합니다. (기록)
    cell = new_code_cell(code_to_run)
    notebook.cells.append(cell)

    # 3. 코드를 실행합니다.
    executor = state['kernel_executor']
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
        # print("이미지 있음")
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

    # 5, 상태를 업데이트하여 반환합니다.
    return {
        "executed_code": code_to_run,
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "notebook": notebook
    }