from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from joblib import load
import numpy as np

# 환경 변수
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
CORS(app, origins=os.getenv("ALLOWED_ORIGINS", "*").split(","), supports_credentials=True,
     allow_headers=["Content-Type", "Authorization"], methods=["GET", "POST", "OPTIONS", "DELETE"])

# DB 연결
engine = create_engine(
    f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

# 데이터 불러오기
# 실제 고객 + 테스트용 API 데이터를 병합할지 여부는 기준값으로 제어
USE_API_UNTIL = 5000  # 예: user_id가 이 값 이하일 때 user_api도 포함

def load_data():
    # 실제 고객 데이터
    user_true = pd.read_sql("SELECT * FROM user", con=engine)
    linker_true = pd.read_sql("SELECT * FROM linker", con=engine)
    participate_true = pd.read_sql("SELECT * FROM participate", con=engine)

    # API용 더미 데이터도 불러오기
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

# 저장된 모델 및 매핑 로드
model = load("model/lightfm_model.pkl")
user_id_map, user_feature_map, item_id_map, item_feature_map = load("model/lightfm_mapping.pkl")

# 주소 기준
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

# 추천 함수
def recommend_linkers(user_id, address_detail=None, top_n=20, exclude_already=True):
    # 신규 유저 처리 (cold-start fallback)
    if user_id not in user_id_map or participate_table[participate_table["user_id"] == user_id].empty:
        # 가장 인기 많은 링커 Top-N 추천
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
    all_item_ids = linker_table["linker_id"].tolist()
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
    if address_detail:
        local_ids = linker_table[linker_table["address_detail"] == address_detail]["linker_id"]
        if local_ids.empty and " " in address_detail:
            # ex) "서울특별시 마포구" → "서울특별시" 단위로 fallback
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
        .head(top_n)["linker_id"]
        .tolist()
    )
    return top_ids

# 평가 함수
def precision_at_k(recommended_ids, true_ids, k):
    if not recommended_ids:
        return 0.0
    return len(set(recommended_ids[:k]) & set(true_ids)) / k

def evaluate_model(user_ids, k=5):
    scores = []
    for user_id in user_ids:
        parts = participate_table[participate_table["user_id"] == user_id]
        if len(parts) < 2:
            continue
        test_parts = parts.sample(frac=0.4, random_state=42)
        address = get_top_address_detail(user_id)
        local_linkers = linker_table[linker_table["address_detail"] == address]["linker_id"]
        true_ids = test_parts[test_parts["linker_id"].isin(local_linkers)]["linker_id"].tolist()
        recommended_ids = recommend_linkers(user_id, address, k, exclude_already=False)
        if true_ids and recommended_ids:
            scores.append(precision_at_k(recommended_ids, true_ids, k))
    return np.mean(scores) if scores else 0.0

# API 엔드포인트
# API - 추천
@app.route("/recommendAI", methods=["GET"])
def recommend():
    try:
        user_id = int(request.args.get("user_id"))
        address = request.args.get("address_detail") or get_top_address_detail(user_id)
        results = recommend_linkers(user_id, address)
        return jsonify({"linker_ids": results}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API - 평가
@app.route("/evaluateAI", methods=["GET"])
def evaluate():
    try:
        sample_users = user_table["user_id"].sample(100, random_state=42)
        precision = evaluate_model(sample_users)
        return jsonify({"precision_at_5": round(precision, 4)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 서버 실행
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8082)
