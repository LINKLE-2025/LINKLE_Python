from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from joblib import load
import numpy as np
import threading
import time


# 0) 환경 변수 및 Flask 초기 설정
env = os.getenv("FLASK_ENV", "development")
load_dotenv(".env")
load_dotenv(f".env.{env}.local", override=True)

# Flask 앱 설정
# origins: 허용할 호스트 지정
# supports_credentials: true 값을 주면, 쿠키를 함께 넘겨줄 수 있다.
# allow_headers: 웹 브라우저에서 서버로 보내는 요청에 포함될 수 있는 헤더 목록을 지정
#  -> Content-Type: 웹 통신에서 데이터의 형식을 저장하는 HTTP 헤더이다.
#  -> Authorization: 권한 부여 또는 인가를 의미
# methods: 어떤 방식으로 통신할 것인지에 대해 지정
app = Flask(__name__)
CORS(
    app,
    origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    supports_credentials=True,
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "OPTIONS", "DELETE"]
)

# 1) DB 연결
engine = create_engine(
    f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

# 2) 데이터 로드
USE_API_UNTIL = 5000  # user_id 기준, 이 이하일 때 user_api도 포함

def load_data():
    # 실제 고객 데이터
    user_true = pd.read_sql("SELECT * FROM user", con=engine)
    linker_true = pd.read_sql("SELECT * FROM linker", con=engine)
    participate_true = pd.read_sql("SELECT * FROM participate", con=engine)

    # API 더미 데이터 (없을 수도 있으므로 try/except)
    try:
        user_api = pd.read_sql("SELECT * FROM user_api", con=engine)
        linker_api = pd.read_sql("SELECT * FROM linker_api", con=engine)
        participate_api = pd.read_sql("SELECT * FROM participate_api", con=engine)
    except Exception:
        user_api = pd.DataFrame(columns=user_true.columns)
        linker_api = pd.DataFrame(columns=linker_true.columns)
        participate_api = pd.DataFrame(columns=participate_true.columns)

    # 전환 기준 적용
    if user_true["user_id"].max() > USE_API_UNTIL:
        return user_true, linker_true, participate_true
    else:
        return (
            pd.concat([user_true, user_api], ignore_index=True),
            pd.concat([linker_true, linker_api], ignore_index=True),
            pd.concat([participate_true, participate_api], ignore_index=True)
        )

user_table, linker_table, participate_table = load_data()


# 3) 저장된 모델 및 매핑 로드
model = load("model/lightfm_model.pkl")
mapping = load("model/lightfm_mapping.pkl")

# dict 기반 매핑 불러오기
user_id_map = mapping.get("user_id_map", {})
user_feature_map = mapping.get("user_feature_map", {})
item_id_map = mapping.get("item_id_map", {})
item_feature_map = mapping.get("item_feature_map", {})


# 4) 주소 기반 유틸
def get_top_address_detail(user_id):
    query = """
    SELECT sub.address_detail
    FROM (
        SELECT l.address_detail, COUNT(*) AS cnt, MAX(p.participated_date) AS latest_date
        FROM participate p
        INNER JOIN linker l ON p.linker_id = l.linker_id
        WHERE p.user_id = %(user_id)s
        GROUP BY l.address_detail
    ) sub
    ORDER BY sub.cnt DESC, sub.latest_date DESC
    LIMIT 1;
    """
    df = pd.read_sql(query, con=engine, params={"user_id": user_id})
    return df.iloc[0]["address_detail"] if not df.empty else None


# 5) 추천 함수
def recommend_linkers(user_id, address_detail=None, top_n=5, exclude_already=True):
    
    # 신규 유저 (cold-start) → 인기 기반 추천
    # 아직 참여 데이터가 없는 유저는 학습 기반 예측이 불가하므로,
    # 전체 참여 기록(participate_table)을 기준으로 인기 링커 Top-N을 추천
    if user_id not in user_id_map or participate_table[participate_table["user_id"] == user_id].empty:
        popular_linkers = (
            participate_table["linker_id"]
            .value_counts()
            .head(top_n)
            .index
            .tolist()
        )
        return popular_linkers

    # 내부 ID 변환
    user_internal_id = user_id_map[user_id]

    # 모든 아이템 후보 가져오기
    # LightFM은 문자열/숫자 ID를 직접 쓰지 않고, 내부 매핑된 정수 ID(item_id_map)를 사용해야 함
    all_item_ids = list(set(linker_table["linker_id"].tolist()))
    if not all_item_ids:
        return []
    
    item_internal_ids = [item_id_map[i] for i in all_item_ids if i in item_id_map]
    if not item_internal_ids:
        return []

    # LightFM 점수 예측
    scores = model.predict(user_internal_id, item_internal_ids)
    score_df = pd.DataFrame({
        "linker_id": [i for i in all_item_ids if i in item_id_map],
        "score": scores
    })

    # 주소 기반 필터링
    # 예시: 1차 서울특별시 종로구/ 2차 서울특별시
    if address_detail:
        local_ids = linker_table[linker_table["address_detail"] == address_detail]["linker_id"]
        if local_ids.empty and " " in address_detail:
            sido = address_detail.split()[0]
            local_ids = linker_table[linker_table["address_detail"].str.startswith(sido)]["linker_id"]
        score_df = score_df[score_df["linker_id"].isin(local_ids)]

    # 이미 참여한 링커 제외
    if exclude_already:
        already = participate_table[participate_table["user_id"] == user_id]["linker_id"].tolist()
        score_df = score_df[~score_df["linker_id"].isin(already)]

    if score_df.empty:
        return []

    # Top-N 추출
    top_ids = (
        score_df.sort_values(by="score", ascending=False)
        .drop_duplicates(subset="linker_id")
        .head(top_n)["linker_id"]
        .tolist()
    )
    return top_ids


# 6) 평가 함수
'''
Precision@K = (추천 Top-K ∩ 실제 참여 아이템 수) / K
즉, 추천 Top-K 중에서 실제로 유저가 참여한 아이템이 몇 개 포함되었는지를 비율로 계산
'''
def precision_at_k(recommended_ids, true_ids, k):
    if not recommended_ids:
        return 0.0
    return len(set(recommended_ids[:k]) & set(true_ids)) / k
    
# 현재 상위 5개로 설정
'''
즉, 현재 5개 중에 몇개를 참여했는지를 리스트로 변환하여 이를 평균으로 정확도를 구하는 방식임
'''
def evaluate_model(user_ids, k=5):
    scores = []
    for user_id in user_ids:
        # 3개 이상인 경우 학습 진행
        parts = participate_table[participate_table["user_id"] == user_id]
        if len(parts) < 2:
            continue
        # 유저의 참여 내역 중 40%를 랜덤으로 샘플링 -> 이는 Test용 데이터로 활용함
        test_parts = parts.sample(frac=0.4, random_state=42)
        # 유저가 가장 많이 참여한 지역명을 가져오고,
        # 해당 지역에 속한 링커만 정답 후보(true_ids)로 사용하여 Precision@K 평가
        address = get_top_address_detail(user_id)
        local_linkers = linker_table[linker_table["address_detail"] == address]["linker_id"]
        true_ids = test_parts[test_parts["linker_id"].isin(local_linkers)]["linker_id"].tolist()
        recommended_ids = recommend_linkers(user_id, address, k, exclude_already=False)
        if true_ids and recommended_ids:
            scores.append(precision_at_k(recommended_ids, true_ids, k))
    return np.mean(scores) if scores else 0.0

# 7) API 엔드포인트
@app.route("/recommendAI", methods=["GET"])
def recommend():
    try:
        # 현재 백엔드에서 user_id, address_detail 값을 활용하여 진행중에 있음
        user_id = int(request.args.get("user_id"))
        address = request.args.get("address_detail")
        if not address:
            address = get_top_address_detail(user_id)
        results = recommend_linkers(user_id, address)
        return jsonify({"linker_ids": results}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/evaluateAI", methods=["GET"])
def evaluate():
    try:
        sample_users = user_table["user_id"].sample(100, random_state=42)
        precision = evaluate_model(sample_users)
        return jsonify({"precision_at_5": round(precision, 4)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 8) 모델 자동 리로드 (백그라운드 스레드)
'''
train_loop에서 주기적으로 학습·저장된 모델을 API 서버에서 주기적으로 다시 로드하여 최신 상태 유지
-> (학습 프로세스와 API 서버가 동시에 접근해도 안전하도록 백그라운드 스레드에서 처리)
'''
MODEL_RELOAD_INTERVAL = 60 * 5  # 5분마다 리로드

def model_reload_loop():
    global model, user_id_map, user_feature_map, item_id_map, item_feature_map
    while True:
        try:
            print("🔁 모델 재로딩 중...")
            model = load("model/lightfm_model.pkl")
            mapping = load("model/lightfm_mapping.pkl")
            user_id_map = mapping.get("user_id_map", {})
            user_feature_map = mapping.get("user_feature_map", {})
            item_id_map = mapping.get("item_id_map", {})
            item_feature_map = mapping.get("item_feature_map", {})
            print("✅ 모델 재로딩 완료")
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
        time.sleep(MODEL_RELOAD_INTERVAL)


# 9) 메인 실행
if __name__ == "__main__":
    threading.Thread(target=model_reload_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=8082)
