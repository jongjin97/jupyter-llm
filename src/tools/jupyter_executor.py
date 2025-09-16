from jupyter_client.manager import KernelManager

class JupyterExecutor:
    """
    jupyter_client를 래핑하여 Jupyter 커널을 제어하는 클래스.
    커널 시작, 코드 실행, 결과 수집, 커널 종료 기능을 캡슐화
    """
    def __init__(self, timeout: int = 10, create_notebook_on_start: str = None):
        """
        클래스 인스턴스를 초기화 하고 Jupyter 커널을 시작
        """
        try:
            # 1. 커널 매니저를 통해 백그라운드에서 커널 프로세스를 시작합니다.
            self.km = KernelManager()
            self.km.start_kernel()
            print("🚀 Jupyter Kernel process started.")

            # 2. 커널과 통신할 클라이언트를 생성하고 채널을 엽니다.
            self.kc = self.km.client()

            self.kc.start_channels()

            # 3. 커널이 응답할 준비가 될 때까지 기다립니다.
            self.kc.wait_for_ready(timeout=timeout)
            print("✅ Jupyter Kernel is ready and connected.")

            # 초기 노트북 생성 메서드 호출
            if create_notebook_on_start:
                print(f"📄 Creating initial notebook: {create_notebook_on_start}...")
                self._create_initial_notebook(create_notebook_on_start)

        except Exception:
            print("🔥 Failed to start or connect to the kernel.")
            # traceback.print_exc() # 디버깅 시 상세 에러를 보려면 주석 해제
            self.shutdown() # 실패 시 자원 정리
            raise

    def _create_initial_notebook(self, filename: str):
        """
        nbformat을 사용하여 .ipynb 파일을 생성하는 메서드
        """
        creation_code = f"""
        import sys
        !{{sys.executable}} -m pip install -q nbformat
        import nbformat
        from nbformat.v4 import new_notebook, new_markdown_cell

        nb = new_notebook()
        welcome_message = '''
        # 세션 노트북: {filename}

        이 노트북은 JupyterExecutor가 시작될 때 자동으로 생성되었습니다.
        작업 내용을 여기에 기록하세요.
        '''
        nb.cells.append(new_markdown_cell(welcome_message))

        with open('{filename}', 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        """
        # execute 메서드를 재사용하여 코드를 실행
        result = self.execute(creation_code)
        print(result)

    def execute(self, code: str, timeout: int = 30) -> str:
        """
        주어진 코드를 커널에서 실행하고, 그 결과를 정리된 문자열로 반환합니다.

        Args:
            code (str): 실행할 Python 코드.
            timeout (int): 각 메시지를 기다릴 최대 시간 (초).

        Returns:
            str: 표준 출력(stdout)과 표준 에러(stderr)를 포함한 실행 결과.
        """
        if not self.is_alive():
            return "Kernel is not running."

        # 실행 요청 보내기
        self.kc.execute(code)

        stdout = ""
        stderr = ""

        # 실행이 완료될 때까지 커널로부터 메시지를 받아 처리
        while True:
            try:
                # IOPub 채널에서 메시지를 가져옵니다.
                msg = self.kc.get_iopub_msg(timeout=timeout)

                msg_type = msg['header']['msg_type']
                content = msg['content']

                # print(f"DEBUG: Received message type: {msg_type}") # 디버깅용

                # 커널 상태가 'idle(대기) 상태가 되면 실행이 끝났다는 의미
                if msg_type == 'status' and content['execution_state'] == 'idle':
                    break

                # 'stream' 메시지는 print()문의 결과
                if msg_type == 'stream':
                    if content['name'] == 'stdout':
                        stdout += content['text']
                    else:
                        stderr += content['text']

                # '팻말'에 해당하는 'execute_result' 메시지를 확인합니다.
                if msg_type == 'execute_result':
                    # 결과 데이터 중 일반 텍스트(text/plain) 표현을 가져옵니다.
                    if 'data' in content and 'text/plain' in content['data']:
                        stdout += content['data']['text/plain'] + '\n'

                # 에러 메시지를 처리
                if msg_type == 'error':
                    stderr += f"{content['ename']}: {content['evalue']}\n"
                    # traceback 정보가 있다면 추가
                    if 'traceback' in content:
                        stderr += "\n".join(content['traceback']) + "\n"

            except Exception:
                break

        # 결과를 하나의 문자열로 정리하여 반환
        observation = f"--- STDOUT ---\n{stdout}\n"
        if stderr:
            observation += f"--- STDERR --- (Error Occurred)\n{stderr}\n"

        return observation.strip()

    def shutdown(self):
        """
        커널 클라이언트 채널을 닫고 커널 프로세스를 안전하게 종료합니다.
        """
        if hasattr(self, 'kc') and self.kc and self.kc.channels_running:
            self.kc.stop_channels()
            print("🔌 Kernel client channels stopped.")

        if hasattr(self, 'km') and self.km and self.km.is_alive():
            self.km.shutdown_kernel(now=True)
            print("💥 Kernel process shut down.")

    def is_alive(self) -> bool:
        """
        커널 프로세스가 현재 실행 중인지 확인합니다.
        """
        return self.km.is_alive()


# --- 직접 실행하여 테스트하는 경우 ---
if __name__ == '__main__':
    print("JupyterExecutor를 테스트합니다.")
    executor = None
    try:
        executor = JupyterExecutor()

        # 테스트 1: 간단한 print문 실행
        print("\n[Test 1: Simple Print]")
        result1 = executor.execute("print('Hello from the kernel!')")
        print(result1)

        # 테스트 2: 라이브러리 설치 및 사용
        print("\n[Test 2: Install and use pandas]")
        result2_install = executor.execute("!pip install -q pandas")
        print(result2_install)

        result2_code = executor.execute(
            "import pandas as pd\n"
            "df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})\n"
            "print(df.to_string())"
        )
        print(result2_code)

        # 테스트 3: 의도적인 에러 발생
        print("\n[Test 3: Intentional Error]")
        result3 = executor.execute("print(x)")  # 정의되지 않은 변수 사용
        print(result3)

    finally:
        if executor:
            executor.shutdown()