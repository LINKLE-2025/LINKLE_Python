import time
import traceback
import datetime
from save_model import retrain_model

# 재학습 주기 (현재 5분 / 운영은 1주일 권장)
SLEEP_INTERVAL = 60 * 5  

# API → 실제 데이터 전환 기준
USE_API_UNTIL = 5000

def log(msg: str):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # print(f"[{now}] {msg}")

def run_training_loop():
    while True:
        try:
            # log("모델 재학습 시작")

            # 모델 재학습 수행
            result = retrain_model(use_api_until=USE_API_UNTIL)
            # log("모델 재학습 완료")

            # 성능 로그 출력
            # log(f"Precision@5 (Train): {result['train_precision']:.4f}")
            # log(f"Precision@5 (Test) : {result['test_precision']:.4f}")
            # log(f"AUC (Test)         : {result['test_auc']:.4f}")

        except Exception as e:
            log(f"에러 발생: {e}")
            traceback.print_exc()

        log(f"{SLEEP_INTERVAL}초 후 재학습 대기 중...\n")
        time.sleep(SLEEP_INTERVAL)

if __name__ == "__main__":
    run_training_loop()