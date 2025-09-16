import os
import time
# 테스트하려는 클래스를 임포트합니다.
# 파일 이름이 다르다면 'jupyter_executor' 부분을 수정해주세요.
from jupyter_executor import JupyterExecutor


def run_test(test_function):
    """테스트 함수를 실행하고 성공/실패를 출력하는 헬퍼 함수"""
    test_name = test_function.__name__
    print(f"\n--- 🧪 [TEST] Running: {test_name} ---")
    try:
        test_function()
        print(f"--- ✅ [PASS] {test_name} ---")
    except Exception as e:
        print(f"--- 🔥 [FAIL] {test_name} ---")
        print(f"Error: {e}")
        # 상세한 에러 스택을 보려면 아래 줄의 주석을 해제하세요.
        # import traceback
        # traceback.print_exc()


def test_simple_execution():
    """가장 기본적인 코드 실행과 STDOUT 캡처를 테스트합니다."""
    executor = None
    try:
        executor = JupyterExecutor()
        code = "message = 'Hello, World!'\nprint(message)"
        result = executor.execute(code)

        print(result)
        assert "Hello, World!" in result
        assert "STDERR" not in result
    finally:
        if executor:
            executor.shutdown()


def test_error_capture():
    """코드 실행 시 발생하는 에러와 STDERR 캡처를 테스트합니다."""
    executor = None
    try:
        executor = JupyterExecutor()
        code = "print(undefined_variable)"  # 정의되지 않은 변수 사용
        result = executor.execute(code)

        print(result)
        assert "STDERR --- (Error Occurred)" in result
        assert "NameError" in result
    finally:
        if executor:
            executor.shutdown()


def test_notebook_creation_on_start():
    """클래스 생성 시 .ipynb 파일이 자동으로 생성되는지 테스트합니다."""
    executor = None
    test_filename = "test_notebook.ipynb"

    # 테스트 전에 혹시 파일이 남아있다면 삭제하여 깨끗한 상태에서 시작합니다.
    if os.path.exists(test_filename):
        os.remove(test_filename)

    try:
        # create_notebook_on_start 인자를 전달하여 객체를 생성합니다.
        executor = JupyterExecutor(create_notebook_on_start=test_filename)

        # 생성자 실행이 끝날 시간을 잠시 줍니다.
        time.sleep(2)

        # 파일이 실제로 생성되었는지 확인합니다.
        print(f"Checking if '{test_filename}' exists...")
        print(f"파일 경로: {os.path.abspath(test_filename)}")
        assert os.path.exists(test_filename), f"File '{test_filename}' was not created!"
        print(f"File '{test_filename}' created successfully.")

    finally:
        # 테스트가 끝나면 생성된 파일과 커널을 정리합니다.
        if executor:
            executor.shutdown()
        # if os.path.exists(test_filename):
        #     os.remove(test_filename)
        #     print(f"Cleaned up '{test_filename}'.")


# --- 메인 실행 블록 ---
if __name__ == "__main__":
    print("=" * 40)
    print("🚀 Starting JupyterExecutor Class Tests 🚀")
    print("=" * 40)

    run_test(test_simple_execution)
    run_test(test_error_capture)
    run_test(test_notebook_creation_on_start)

    print("\n\n🎉 All tests completed.")