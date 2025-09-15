import time
import traceback
import datetime
from save_model import retrain_model

# 학습 주기 (1주일)
SLEEP_INTERVAL = 60 * 60 * 24 * 7

# API → 실제 데이터 전환 기준
USE_API_UNTIL = 5000  # 예: user_id 5000 이하일 때까진 user_api 포함

def main():
    while True:
        try:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{now} - 모델 재학습 시작")

            # 신규 유저 ID를 파라미터로 넘길 수도 있음 (없으면 None)
            retrain_model(new_user_id=None, use_api_until=USE_API_UNTIL)

            print(f"{now} - 모델 재학습 완료\n")

        except Exception as e:
            print(f"에러 발생: {e}")
            traceback.print_exc()

        time.sleep(SLEEP_INTERVAL)

if __name__ == "__main__":
    main()
