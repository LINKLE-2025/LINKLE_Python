import time
import traceback
import datetime
from save_model import retrain_mode

# 학습 주기 1주일
SLEEP_INTERVAL = 60 * 60 * 24 * 7

# 
def main():
    while True:
        try:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{now} - 모델 재학습 시작")

            retrain_model()  # save_model.py에서 이 함수가 모델 저장까지 담당

            print(f"{now} - 모델 재학습 완료\n")

        except Exception as e:
            print(f"에러 발생: {e}")
            traceback.print_exc()

        time.sleep(SLEEP_INTERVAL)

if __name__ == "__main__":
    main()