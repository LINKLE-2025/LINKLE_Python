import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sqlalchemy import create_engine
from sklearn.decomposition import TruncatedSVD
from dotenv import load_dotenv

# .env 파일 로드
# DB 정보를 .env 파일로 관리하여 보안 강화
# 환경 감지
env = os.getenv("FLASK_ENV", "development")

# 기본 .env
load_dotenv(".env")

# 환경별 파일
load_dotenv(f".env.{env}", override=True)
load_dotenv(f".env.{env}.local", override=True)

# Flask 앱 생성
app = Flask(__name__)

# CORS 설정
# .env에서 ALLOWED_ORIGINS를 가져와 허용
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
CORS(app, origins=allowed_origins)

# 환경변수로부터 값 읽기
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')

# SQLAlchemy 엔진 생성 (pymysql 사용)
engine = create_engine(f'mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

# 학습 테이블 DB에서 불러오기 -> DataFrame 형태
user_table = pd.read_sql('SELECT * FROM USER', con=engine)
linker_table = pd.read_sql('SELECT * FROM LINKER', con=engine)
participate_table = pd.read_sql('SELECT * FROM PARTICIPATE', con=engine)

def get_top_address_detail(user_id: int, engine) -> str | None:
    query = """
    SELECT sub.address_detail
    FROM (
        SELECT 
            l.address_detail,
            COUNT(*) AS cnt,
            MAX(p.participated_date) AS latest_date
        FROM PARTICIPATE p
        INNER JOIN LINKER l
            ON p.linker_id = l.linker_id
        WHERE p.user_id = %(user_id)s
        GROUP BY l.address_detail
    ) sub
    ORDER BY sub.cnt DESC, sub.latest_date DESC
    LIMIT 1;
    """
    df = pd.read_sql(query, con=engine, params={"user_id": user_id})
    if df.empty:
        return None
    return df.iloc[0]['address_detail']


# 유저-링커 행렬 생성
# crosstab을 활용하여 배열에 대한 단순 교차표를 만든다.
user_item_matrix = pd.crosstab(participate_table['user_id'], participate_table['linker_id'])

# SVD(특이값 분해, Singular Value Decomposition) 모델 활용
# 행렬 분해 방법 중 하나로 매우 많은 feature를 가진 고차원 행렬을 저차원 행렬로 분리하는 기법이다.
svd = TruncatedSVD(n_components=10, random_state=42)
user_factors = svd.fit_transform(user_item_matrix)
item_factors = svd.components_.T

# 예측 점수 행렬
predicted_ratings = np.dot(user_factors, item_factors.T)
predicted_df = pd.DataFrame(predicted_ratings,
                            index=user_item_matrix.index,
                            columns=user_item_matrix.columns)

# 중요도 설정시 나이대와 성별을 가지고 가중치 부여를 위한 작업
# 동일한 나이대와 성별을 가진 사용자들이 선호한 링커 정보를 기반으로
# 현재 사용자와 유사한 선호 경향에 대한 가중치 반환
def get_similar_user_preference(user_id, user_table, participate_table):
    target = user_table[user_table['user_id'] == user_id]
    if target.empty:
        return {}
    age = target.iloc[0]['age']
    gender = target.iloc[0]['gender']
    # 같은 성별+나이대 유저들
    sim_users = user_table[(user_table['age'] == age) & (user_table['gender'] == gender)]['user_id']
    sim_parts = participate_table[participate_table['user_id'].isin(sim_users)]
    return sim_parts['linker_id'].value_counts(normalize=True).to_dict()

# 전체적인 작업 진행
# SVD기반 점수 + 사용자 선호도 + 지역 필터링
def recommend_linkers_hybrid_local(user_id, address_detail, top_n=5):
    if user_id not in predicted_df.index:
        return []

    # 1) SVD 점수
    scores = predicted_df.loc[user_id].copy()

    # 2) 이미 참여한 링커 제거
    already = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index.tolist()
    scores.drop(labels=already, inplace=True, errors='ignore')

    # 3) 성별/나이대 가중치 반영
    weights = get_similar_user_preference(user_id, user_table, participate_table)
    for lid in scores.index:
        scores.loc[lid] += weights.get(lid, 0)

    # 4) 지역 필터
    local_ids = linker_table[linker_table['address_detail'] == address_detail]['linker_id']
    scores = scores[scores.index.isin(local_ids)]
    if scores.empty:
        return []

    # 5) Top-N
    top_ids = scores.sort_values(ascending=False).head(top_n).index.tolist()

    # linker_id 리스트만 반환
    return top_ids

# 추천 API 엔드포인트
@app.route('/recommend', methods=['GET'])
def recommend():
    try:
        user_id = int(request.args.get('user_id'))
        top_n = 5

        address = request.args.get('address_detail')
        if not address:
            address = get_top_address_detail(user_id, engine)

        results = recommend_linkers_hybrid_local(user_id, address, top_n) or []
        return jsonify({"linker_ids": results}), 200  # ✅ 항상 동일 스키마
    except Exception as e:
        return jsonify({'error': str(e)}), 500




# HTTPS 설정 및 서버 실행
if __name__ == '__main__':
    ssl_cert = os.getenv("SSL_CERT_PATH")
    ssl_key = os.getenv("SSL_KEY_PATH")

    app.run(host='0.0.0.0', port=443, ssl_context=(ssl_cert, ssl_key))