import jupyter_client
from jupyter_client.manager import KernelManager
import sys


# --- 함수: 코드 실행 및 결과 출력 ---
def run_code_on_kernel(kc, code):
    """주어진 코드를 커널에서 실행하고, 결과를 스트리밍으로 출력하는 함수"""
    print(f"\n--- [실행 요청] ---\n{code.strip()}\n-------------------")

    msg_id = kc.execute(code)

    while True:
        try:
            msg = kc.get_iopub_msg(timeout=5)
            msg_type = msg['header']['msg_type']
            content = msg['content']

            print(msg)
            if msg_type == 'status' and content['execution_state'] == 'idle':
                break
            if msg_type == 'stream':
                # 에러(stderr)와 일반 출력(stdout)을 구분하여 표시
                if content['name'] == 'stderr':
                    print(f"🔥 [에러] {content['text']}", end="", file=sys.stderr)
                else:
                    print(f"✅ [출력] {content['text']}", end="")

        except Exception:
            break
    print("\n--- [실행 완료] ---")


# --- 메인 로직 ---

# 1. 커널 시작
km = KernelManager()
km.start_kernel()
kc = km.client()
kc.start_channels()

try:
    kc.wait_for_ready(timeout=10)
    print("🚀 커널이 준비되었습니다.")
except RuntimeError:
    print("커널 연결 시간 초과. 종료합니다.")
    kc.stop_channels()
    km.shutdown_kernel()
    exit()

# 2. (핵심) Pandas 라이브러리 설치 코드 실행
#    -q 옵션으로 설치 로그를 간소화합니다.
install_code = "!pip install -q pandas"
run_code_on_kernel(kc, install_code)

# 3. Pandas를 사용하는 원래 코드 실행
main_code = """
import pandas as pd
import io

# 간단한 데이터프레임 생성
data = '''
name,age
Alice,30
Bob,25
Charlie,35
'''

df = pd.read_csv(io.StringIO(data))
mean_age = df['age'].mean()
print(f"데이터프레임이 성공적으로 로드되었습니다.")
print(f"평균 나이는 {mean_age}세 입니다.")
"""
run_code_on_kernel(kc, main_code)

# 4. 커널 종료
print("\n💤 모든 작업 완료. 커널을 종료합니다.")
kc.stop_channels()
km.shutdown_kernel(now=True)