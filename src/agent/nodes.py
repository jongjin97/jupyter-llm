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
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are an expert Python programmer and a helpful AI assistant operating within a Jupyter Notebook environment. "
             "Your goal is to complete the user's task by planning and executing Python code. "
             "Review the history of previously executed code and its observations to decide the next step. "
             "If you encounter a ModuleNotFoundError, you must generate a `!pip install` command to install the missing library. "
             "When the user's entire task is complete, return 'FINISH' in the code field to end the session."),
            ("human",
             "User's Task: {task}\n\n"
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
    response = structured_llm.invoke(prompt.format(**state))

    # 결정된 계획을 상태에 업데이트하여 반환
    return {"plan": [response.code]}


def code_executor_node(state: AgentState):
    """
    코드를 실행하고, 실행 내역을 노트북에 기록한 뒤 저장합니다.
    """
    code_to_run = state['plan'][-1]

    if code_to_run == "FINISH":
        return {"observation": "작업 완료."}

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