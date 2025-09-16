import jupyter_client
from jupyter_client.manager import KernelManager
import logging

# 1. Jupyter 커널 관리자를 생성
# 이를 총해 백그라운드에서 Jupyter 커널 프로세스가 시작
km = KernelManager()
km.start_kernel()
print("Jupyter 커널이 시작.")
print("연결 정보 파일: ", km.connection_file)

# 2. 커널과 통신할 클라이언트를 생성
# 이 클라이언트를 통해 코드를 보내고 결과를 받음.
kc = km.client()
kc.start_channels() # 통신 채널 시작

# 클라이언트가 커널에 완전히 연결될 때까지 대기
# is_alive()는 커널 프로세스가 실행 중인지 확인.
try:
    kc.wait_for_ready(timeout=5)
    print("커널 클라이언트가 성공적으로 연결.")
except RuntimeError:
    print("커널 연결 시간 초과.")
    kc.stop_channels()
    km.shutdown_kernel()
    exit()

# 실행할 코드를 문자열로 정의
code_to_execute = """
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
print("데이터프레임 정보:")
df.info()

# 데이터프레임 출력
print("\\n데이터프레임 내용:")
print(df)

# 간단한 연산
mean_age = df['age'].mean()
print(f"\\n평균 나이: {mean_age}")
"""

# 4. 코드를 커널로 보내 실행을 요청
# execute() 메서드는 메시지 ID를 반환.
msg_id = kc.execute(code_to_execute)
print(f"\n코드를 커널로 전송했습니다. (메시지 ID: {msg_id})")
print("-" * 30)
print("커널로부터 응답 수신 중...")
print("-" * 30)

# 5. 커널로부터 실행 결과를 받아옴.
# 커널은 코드 실행 과정에서 여러 종류의 메시지를 보냅니다.
# (예: status, execute_input. stream, display_data, execute_reply 등)
# get_iopub_msg()를 반복적으로 호출하여 모든 메시지르 처리

while True:
    try:
        # IOPub 채널에서 메시지를 기다림. timeout을 설정하여 무한 대기를 방지
        msg = kc.get_iopub_msg(timeout=1)
        print(msg)
        # 메시지 내용 확인
        msg_type = msg['header']['msg_type']
        content = msg['content']

        # 'status' 메시지: 커널 상태가 'idle' (대기 중)이 도면 실행이 끝났다는 의미
        if msg_type == 'status' and content['execution_state'] == 'idle':
            print("\n[알림] 커널이 'idle' 상태가 되어 응답 수신을 종료.")
            break

        # 'stream' 메시지: print()문의 결과 (표준 출력)
        if msg_type == 'stream':
            print(f"[표준 출력] {content['text']}", end="")

        # 'execute_result' 또는 'display_data' 메시지: 객체 표현식, 이미지 등
        # 이 예제에서는 간단한 텍스트만 처리
        if msg_type in ('execute_result', 'display_data'):
            if 'data' in content and 'text/plain' in content['data']:
                print(f"[객체 출력] {content['data']['text/plain']}")

        if msg_type == 'error':
            print(f"[에러 발생] {content['traceback']}")

    except Exception:
        break

# 6. 모든 작업이 끝나면 커널과 통신 채널을 종료
print("-" * 30)
print("모든 작업 완료. 커널과 클라이언트를 종료합니다.")
kc.stop_channels()
km.shutdown_kernel(now=True) # now=True는 즉시 종료를 의미합니다.
print("종료 완료.")