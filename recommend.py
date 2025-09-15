import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sqlalchemy import create_engine
from sklearn.decomposition import TruncatedSVD
from dotenv import load_dotenv

# 환경 변수 로딩
env = os.getenv("FLASK_ENV", "development")
load_dotenv(".env")
load_dotenv(f".env.{env}", override=True)
load_dotenv(f".env.{env}.local", override=True)

# Flask 앱 생성 및 CORS 설정
app = Flask(__name__)
CORS(app, origins=os.getenv("ALLOWED_ORIGINS", "*").split(","), supports_credentials=True,
     allow_headers=["Content-Type", "Authorization"], methods=["GET", "POST", "OPTIONS"])

# DB 연결
engine = create_engine(
    f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

# DB에서 데이터 로딩
user_table = pd.read_sql('SELECT * FROM USER', con=engine)
linker_table = pd.read_sql('SELECT * FROM LINKER', con=engine)
participate_table = pd.read_sql('SELECT * FROM PARTICIPATE', con=engine)

# 유저-링커 행렬 생성 및 SVD 학습
user_item_matrix = pd.crosstab(participate_table['user_id'], participate_table['linker_id'])
svd = TruncatedSVD(n_components=10, random_state=42)
user_factors = svd.fit_transform(user_item_matrix)
item_factors = svd.components_.T
predicted_df = pd.DataFrame(np.dot(user_factors, item_factors.T),
                            index=user_item_matrix.index,
                            columns=user_item_matrix.columns)

# 유사 사용자 기반 가중치 계산
def get_similar_user_preference(user_id):
    target = user_table[user_table['user_id'] == user_id]
    if target.empty:
        return {}

    age, gender = target.iloc[0][['age', 'gender']]
    sim_users = user_table[(user_table['age'] == age) & (user_table['gender'] == gender)]['user_id']
    sim_parts = participate_table[participate_table['user_id'].isin(sim_users)]
    return sim_parts['linker_id'].value_counts(normalize=True).to_dict()

# 주소 추출
def get_top_address_detail(user_id):
    query = """
    SELECT sub.address_detail
    FROM (
        SELECT l.address_detail, COUNT(*) AS cnt, MAX(p.participated_date) AS latest_date
        FROM PARTICIPATE p
        INNER JOIN LINKER l ON p.linker_id = l.linker_id
        WHERE p.user_id = %(user_id)s
        GROUP BY l.address_detail
    ) sub
    ORDER BY sub.cnt DESC, sub.latest_date DESC
    LIMIT 1;
    """
    df = pd.read_sql(query, con=engine, params={"user_id": user_id})
    return df.iloc[0]['address_detail'] if not df.empty else None

# 하이브리드 추천 함수
def recommend_linkers(user_id, address_detail, top_n=20):
    if user_id not in predicted_df.index:
        return []

    scores = predicted_df.loc[user_id].copy()

    # 이미 참여한 항목 제외
    already = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index.tolist()
    scores.drop(labels=already, inplace=True, errors='ignore')

    # 가중치 반영
    weights = get_similar_user_preference(user_id)
    for lid in scores.index:
        scores.loc[lid] += weights.get(lid, 0)

    # 지역 필터
    local_ids = linker_table[linker_table['address_detail'] == address_detail]['linker_id']
    scores = scores[scores.index.isin(local_ids)]

    if scores.empty:
        return []

    return scores.sort_values(ascending=False).head(top_n).index.tolist()

# Precision@K
def precision_at_k(recommended_ids, true_ids, k):
    if not recommended_ids:
        return 0.0
    return len(set(recommended_ids[:k]) & set(true_ids)) / k

# 평가 함수
def evaluate_model(user_ids, k=20):
    precision_scores = []

    for user_id in user_ids:
        all_parts = participate_table[participate_table['user_id'] == user_id]
        if len(all_parts) < 2:
            continue

        test_parts = all_parts.sample(frac=0.4, random_state=42)
        true_ids = test_parts['linker_id'].tolist()
        address = get_top_address_detail(user_id)
        recommended_ids = recommend_linkers(user_id, address, top_n=k)

        print(f"[user_id: {user_id}] 추천: {recommended_ids} / 정답: {true_ids}")  # ✅ 여기에 위치해야 함

        if true_ids and recommended_ids:
            precision_scores.append(precision_at_k(recommended_ids, true_ids, k))

    return np.mean(precision_scores) if precision_scores else 0.0


# 추천 API
@app.route('/recommend', methods=['GET'])
def recommend():
    try:
        user_id = int(request.args.get('user_id'))
        address = request.args.get('address_detail') or get_top_address_detail(user_id)
        results = recommend_linkers(user_id, address)
        return jsonify({"linker_ids": results}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 정확도 평가 API
@app.route('/evaluate', methods=['GET'])
def evaluate():
    try:
        sample_users = user_table['user_id'].sample(100, random_state=42)
        precision = evaluate_model(sample_users)
        

        return jsonify({"precision_at_5": round(precision, 4)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 서버 실행
if __name__ == '__main__':
    ssl_cert = os.getenv("SSL_CERT_PATH")
    ssl_key = os.getenv("SSL_KEY_PATH")
    app.run(host="0.0.0.0", port=443)  # SSL 없이 실행 시 ssl_context=(ssl_cert, ssl_key) 주석 해제
