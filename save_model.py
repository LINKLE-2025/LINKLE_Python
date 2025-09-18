import os
import pandas as pd
import joblib
from sqlalchemy import create_engine
from dotenv import load_dotenv
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, auc_score
from lightfm.cross_validation import random_train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer



# Hybrid 추천기
def hybrid_recommend(user_id, top_n=10, weights=None, context=None):
    '''
    Hybrid 추천 (LightFM + Global Popularity + Category + ItemCF + CBF)
    현재 LightFM만 사용했을 경우 데이터 부족, 콜드 스타트 등의 문제로 정확도 및 신뢰도가 떨어지는 문제 발생
    이를 해결하기 위해 LightFM과 더불어 다양한 기법을 사용하여 이를 해결
    '''
    import numpy as np

    # 테이블 값이 없을때 에러 호출
    if context is None:
        raise ValueError("context (user_table, linker_table, model, dataset 등) 필요")

    # Context 불러오기
    model = context["model"]
    dataset = context["dataset"]
    user_table = context["user_table"]
    linker_table = context["linker_table"]
    participate_table = context["participate_table"]
    user_features = context["user_features"]
    item_features = context["item_features"]

    # 가중치 기본값
    # 가중치 기본값 (LightFM 비중을 가장 크게 설정: 0.5)
    if weights is None:
        weights = {"lightfm": 0.5, "global": 0.2, "category": 0.1, "itemcf": 0.1, "cbf": 0.1}

    uid = str(user_id)
    if uid not in dataset.mapping()[0]:
        # Cold-start → 인기 기반 추천
        return linker_table.sort_values("popularity", ascending=False).head(top_n)["linker_id"].tolist()

    # 1) LightFM 점수
    '''
    다음 조건을 활용해 예측 점수 반환
    user_id: 내부 정수 ID (위에서 dataset.mapping()[0][uid]로 얻음)
    item_ids: 내부 정수 ID 리스트 (여기서는 전체 아이템 [0,1,2,...])
    user_features: 유저 feature matrix
    item_features: 아이템 feature matrix
    '''
    scores = model.predict(
        dataset.mapping()[0][uid],
        np.arange(len(dataset.mapping()[2])),
        user_features=user_features,
        item_features=item_features
    )
    lightfm_scores = pd.Series(scores, index=list(dataset.mapping()[2].keys()))

    # 2) Global Popularity
    '''
    Global Popularity (전역 인기 기반 추천)
    - 원리:
      → 전체 유저 참여 데이터를 집계해서 "많이 참여된 링커"를 높은 점수로 줌
      → 즉, 참여도가 높은 아이템일수록 추천 점수가 올라감
    - 방법:
      1. participate_table에서 링커별 참여 횟수(popularity) 계산
      2. Min-Max 정규화 (0~1 사이 값으로 스케일링)
         (가장 인기 없는 링커=0, 가장 인기 많은 링커=1)
    - 장점:
      → 신규 유저(Cold-start user)에게 즉시 활용 가능 (데이터 부족 문제 완화)
      → 간단하고 직관적 (유저 데이터 없이도 인기 많은 것부터 보여주면 무난함)
    - 단점:
      → 개인화(personalization)가 없음 → 모든 유저에게 같은 결과 제공
    '''
    global_scores = linker_table.set_index("linker_id")["popularity"]
    global_scores = (global_scores - global_scores.min()) / (global_scores.max() - global_scores.min() + 1e-9)

    
    # 3) Category Popularity
    '''
    Category Popularity (카테고리 기반 추천)
    - 원리:
      → 각 유저의 "선호 카테고리(pref_cat)"를 추정하고,
        같은 카테고리에 속한 아이템들에게 가중치를 부여
      → 예: 유저가 "운동"을 많이 참여 → 운동 카테고리 아이템에 높은 점수
    - 방법:
      1. user_table에서 유저별 선호 카테고리(pref_cat) 가져오기
         (participate_table에서 가장 많이 참여한 카테고리 기반)
      2. linker_table의 category_id와 비교해서 같은 카테고리면 1, 아니면 0 점수 부여
    - 장점:
      → 유저 취향을 반영할 수 있음 (Global Popularity 대비 개인화 강화)
      → Cold-start user에도 활용 가능 (가입 시 입력한 관심사/카테고리 기반)
    - 단점:
      → 한 명이 여러 관심사를 가질 수 있는데 단일 카테고리만 반영하면 제한적
      → 카테고리 granularity(세분화 정도)에 따라 추천 품질 좌우됨
    '''
    if "pref_cat" in user_table.columns:
        user_pref_cat = user_table[user_table["user_id"] == user_id]["pref_cat"].iloc[0]
    else:
        user_pref_cat = "unknown"

    if user_pref_cat != "unknown":
        cat_scores = (linker_table["category_id"] == str(user_pref_cat)).astype(int)
    else:
        cat_scores = pd.Series(0, index=linker_table.index)
    cat_scores = pd.Series(cat_scores.values, index=linker_table["linker_id"])

    
    # 4) Item-based CF
    '''
    Item-based Collaborative Filtering (아이템 기반 협업 필터링)
    - 원리:
      → 유저가 과거에 참여한 링커(아이템)와 다른 링커 간의 "유사도"를 계산해서 추천에 활용
      → "이 아이템을 본 사람은 이런 아이템도 봤다" 라는 패턴을 학습
    - 방법:
      1. 참여 테이블(participate_table)을 이용해 "유저 × 아이템" 상호작용 행렬 생성
      2. 아이템 간 코사인 유사도(cosine similarity) 계산
      3. 유저가 이미 참여한 아이템들과 유사한 아이템의 평균 점수를 추천 점수로 활용
    - 장점:
      → Cold-start(유저 데이터 부족) 상황에서도 '아이템끼리의 유사도'를 기반으로 보완 가능
      → 해석이 직관적 ("A에 참여한 사람은 B에도 참여한다")
    - 단점:
      → 새로운 아이템이 추가되면 데이터가 쌓일 때까지 추천 불가
    '''
    user_items = participate_table[participate_table["user_id"] == user_id]["linker_id"].tolist()
    itemcf_scores = pd.Series(0, index=linker_table["linker_id"])

    # cosine_similartiy를 이용하여 아이템-아이템간의 유사도 측정
    if len(user_items) > 0:
        # 1. 사용자-아이템 상호작용 행렬 생성
        # 행: user_id, 열: linker_id, 값: 참여 여부(1/0)
        interaction_matrix = pd.crosstab(
            participate_table["user_id"], 
            participate_table["linker_id"]
        )
        # 2. 아이템-아이템 간 코사인 유사도 계산
        # -> interaction_matrix.T : (아이템 × 유저) 구조로 전치
        # -> 아이템 간 벡터 유사도를 구함
        sim_matrix = cosine_similarity(
            interaction_matrix.T
        )
        # 3. DataFrame으로 변환 (행렬의 각 행·열은 링커 ID)
        sim_df = pd.DataFrame(
            sim_matrix, 
            index=interaction_matrix.columns, 
            columns=interaction_matrix.columns
        )
        # 4. 사용자가 참여한 아이템(user_items)과의 평균 유사도 점수 계산
        # -> 유저가 참여했던 아이템들과 비슷한 아이템일수록 점수가 높음
        sim_scores = sim_df[user_items].mean(axis=1)
        # 5. Min-Max 정규화 (0~1 범위로 스케일링)
        itemcf_scores = (sim_scores - sim_scores.min()) / (sim_scores.max() - sim_scores.min() + 1e-9)

    
    # 5) Content-based Filtering (TF-IDF)
    '''
    콘텐츠 기반 필터링
    -> 사용자가 소비한 아이템에 대해 아이템의 내용(Content)이 비슷하거나 특별한 관계가 있는 다른 아이템을 추천하는 방법을 의미한다.
    -> 현재 콜드 스타트 문제로 인해 링커의 메모의 내용을 백터로 변환하여 유사도 측정 후 추천 해주는 기법 활용
    '''
    cbf_scores = pd.Series(0, index=linker_table["linker_id"])
    if "memo" in linker_table.columns:
        # 백터화
        tfidf = TfidfVectorizer(max_features=500)
        tfidf_matrix = tfidf.fit_transform(linker_table["memo"].fillna(""))
        if len(user_items) > 0:
            user_vec = tfidf_matrix[linker_table["linker_id"].isin(user_items)].mean(axis=0)
            sim_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
            cbf_scores = pd.Series(sim_scores, index=linker_table["linker_id"])

    # Hybrid 가중 합산
    # 위에 구한 내용들을 통해 구해진 점수를 활용하여 가중치를 부여해줌
    final_scores = (
        weights["lightfm"] * lightfm_scores.add(0, fill_value=0) +
        weights["global"]  * global_scores.add(0, fill_value=0) +
        weights["category"]* cat_scores.add(0, fill_value=0) +
        weights["itemcf"]  * itemcf_scores.add(0, fill_value=0) +
        weights["cbf"]     * cbf_scores.add(0, fill_value=0)
    )
    # 추천 결과 필터링: 실제 운영 DB에 존재하는 linker_id만 사용
    valid_linker_ids = context.get("valid_linker_ids")  # ← 이 키로 운영 ID 세트를 전달
    if valid_linker_ids is not None:
        final_scores = final_scores[final_scores.index.isin(valid_linker_ids)]

    return final_scores.sort_values(ascending=False).head(top_n).index.tolist()



# 모델 재학습
def retrain_model(new_user_id=None, use_api_until: int = 10000, cold_start_mode: bool = False):
    """
    LightFM + Hybrid 보조 추천기 학습
    """
    # 환경 변수
    load_dotenv(".env")
    load_dotenv(f".env.{os.getenv('FLASK_ENV', 'development')}.local", override=True)
    engine = create_engine(
        f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )

    # 데이터 로딩
    user_true = pd.read_sql('SELECT * FROM user', con=engine)
    linker_true = pd.read_sql('SELECT * FROM linker', con=engine)
    participate_true = pd.read_sql('SELECT * FROM participate', con=engine)
    user_api = pd.read_sql('SELECT * FROM user_api', con=engine)
    linker_api = pd.read_sql('SELECT * FROM linker_api', con=engine)
    participate_api = pd.read_sql('SELECT * FROM participate_api', con=engine)

    valid_linker_ids = set(linker_true["linker_id"])


    # 인원수를 설정하여 유저 아이디가 use_api_until에 설정한 값이 넘거가게 되면 신규 유저의 정보만을 가지고 진행
    if user_true["user_id"].max() > use_api_until:
        user_table, linker_table, participate_table = user_true, linker_true, participate_true
    else:
        user_table = pd.concat([user_true, user_api], ignore_index=True)
        linker_table = pd.concat([linker_true, linker_api], ignore_index=True)
        participate_table = pd.concat([participate_true, participate_api], ignore_index=True)

    # 참여 적은 유저 제거 (<3)
    active_users = participate_table["user_id"].value_counts()
    valid_users = active_users[active_users >= 3].index
    participate_table = participate_table[participate_table["user_id"].isin(valid_users)]

    # cold_start_mode=False → 참여 없는 유저/링커 제외
    if not cold_start_mode:
        user_table = user_table[user_table["user_id"].isin(participate_table["user_id"].unique())]
        linker_table = linker_table[linker_table["linker_id"].isin(participate_table["linker_id"].unique())]

    # 유저 전처리
    '''
    1. 성별 결측치는 "unknown"으로 채우고 → 수치형으로 변환 (-1=unknown, 0=남성, 1=여성)
    2. 나이를 0~100 범위로 정리
    3. 성별과 나이를 함께 활용하여 KMeans로 4개 군집으로 분류 (cluster_feature 생성)
    '''
    user_table = user_table.copy()  # SettingWithCopyWarning 방지
    user_table.loc[:, "gender"] = user_table["gender"].fillna("unknown")
    user_table.loc[:, "age"] = pd.to_numeric(user_table["age"], errors="coerce").fillna(0).astype(int).clip(0, 100)
    user_table.loc[:, "gender_num"] = user_table["gender"].map({"남성": 0, "여성": 1}).fillna(-1)

    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(user_table[["gender_num", "age"]])

    # n_clusters 동적 조정 (샘플 수보다 클 수 없음)
    n_samples = len(user_table)
    n_clusters = min(4, n_samples) if n_samples > 0 else 1
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    user_table["cluster_feature"] = "cluster:" + kmeans.fit_predict(X_cluster).astype(str)

    user_table["age_band"] = pd.cut(
        user_table["age"],
        bins=[0, 19, 29, 39, 49, 59, 120],
        labels=["10s", "20s", "30s", "40s", "50s", "60+"],
        include_lowest=True
    )
    user_table["age_feature"] = "age:" + user_table["age_band"].astype(str)

    # 유저별 주요 카테고리
    # 각 유저별 참여 카테고리 중 최빈값(mode)을 찾아 선호 카테고리(pref_cat) 지정
    linker_cat_map = linker_true[["linker_id", "category_id"]]
    user_participation = participate_table.merge(linker_cat_map, on="linker_id", how="left")
    pref_cat = user_participation.groupby("user_id")["category_id"].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown"
    )
    user_table = user_table.merge(pref_cat.rename("pref_cat"), left_on="user_id", right_index=True, how="left")
    user_table["pref_cat_feature"] = "pref_cat:" + user_table["pref_cat"].fillna("unknown").astype(str)

    # 링커 전처리
    # 카테고리/ 링커 참여도(인기도)/ 최근 90일 참여도 feature 생성
    linker_table = linker_table.copy()  # SettingWithCopyWarning 방지
    linker_table.loc[:, "category_id"] = linker_table["category_id"].fillna("unknown").astype(str)
    linker_table.loc[:, "category_feature"] = "category:" + linker_table["category_id"]

    linker_popularity = participate_table.groupby("linker_id").size().reset_index(name="popularity")
    linker_table = pd.merge(linker_table, linker_popularity, on="linker_id", how="left").fillna({"popularity": 0})
    linker_table["popularity_feature"] = "popularity:" + pd.qcut(
        linker_table["popularity"], q=5, labels=False, duplicates="drop"
    ).astype(str)

    if "participated_date" in participate_table.columns:
        participate_table = participate_table.copy()
        participate_table.loc[:, "date"] = pd.to_datetime(participate_table["participated_date"], errors="coerce")
        recent_cut = pd.Timestamp.now() - pd.Timedelta(days=90)
        recent_pop = participate_table[participate_table["date"] >= recent_cut].groupby("linker_id").size()
        linker_table = linker_table.merge(recent_pop.rename("recent_pop"), on="linker_id", how="left")
    linker_table["recent_pop"] = linker_table["recent_pop"].fillna(0).astype(int)
    linker_table["recent_pop_feature"] = "recent_pop:" + pd.qcut(
        linker_table["recent_pop"], q=4, labels=False, duplicates="drop"
    ).astype(str)

    # LightFM Dataset
    dataset = Dataset()
    dataset.fit(
        users=user_table["user_id"].astype(str).tolist(),
        items=linker_table["linker_id"].astype(str).tolist(),
        user_features=(user_table["cluster_feature"].unique().tolist() +
                       user_table["gender"].apply(lambda g: f"gender:{g}").unique().tolist() +
                       user_table["age_feature"].unique().tolist() +
                       user_table["pref_cat_feature"].unique().tolist()),
        item_features=(linker_table["category_feature"].unique().tolist() +
                       linker_table["popularity_feature"].unique().tolist() +
                       linker_table["recent_pop_feature"].unique().tolist())
    )

    interactions, _ = dataset.build_interactions([
        (str(row["user_id"]), str(row["linker_id"]))
        for _, row in participate_table.iterrows()
    ])

    train, test = random_train_test_split(interactions, test_percentage=0.2, random_state=42)
    user_features = dataset.build_user_features([
        (str(row["user_id"]), [row["cluster_feature"], f"gender:{row['gender']}",
                               row["age_feature"], row["pref_cat_feature"]])
        for _, row in user_table.iterrows()
    ])
    item_features = dataset.build_item_features([
        (str(row["linker_id"]), [row["category_feature"], row["popularity_feature"], row["recent_pop_feature"]])
        for _, row in linker_table.iterrows()
    ])

    # LightFM 학습
    model = LightFM(loss="warp", no_components=128)
    model.fit(train, user_features=user_features, item_features=item_features, epochs=70, num_threads=1)

    # 평가
    '''
    test 데이터가 없으면 precision/auc 계산이 불가능 → 0.0으로 대체
    '''
    if test.getnnz() == 0:
        train_p, test_p, test_auc = 0.0, 0.0, 0.0
    else:
        train_p = precision_at_k(model, train, user_features=user_features, item_features=item_features, k=5).mean()
        test_p = precision_at_k(model, test, user_features=user_features, item_features=item_features, k=5).mean()
        test_auc = auc_score(model, test, user_features=user_features, item_features=item_features).mean()

    print(f"\n📊 Precision@5 (Train): {train_p:.4f}")
    print(f"📊 Precision@5 (Test) : {test_p:.4f}")
    print(f"📊 AUC (Test) : {test_auc:.4f}")

    # 저장
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/lightfm_model.pkl")
    joblib.dump({
        "user_id_map": dataset.mapping()[0],
        "user_feature_map": dataset.mapping()[1],
        "item_id_map": dataset.mapping()[2],
        "item_feature_map": dataset.mapping()[3],
    }, "model/lightfm_mapping.pkl")
    joblib.dump({
        "model": model,
        "dataset": dataset,
        "user_table": user_table,
        "linker_table": linker_table,
        "participate_table": participate_table,
        "user_features": user_features,
        "item_features": item_features,
        "valid_linker_ids": valid_linker_ids 
    }, "model/hybrid_context.pkl")
    print("\n LightFM 모델과 Hybrid context 저장 완료.")

    return {
        "train_precision": train_p,
        "test_precision": test_p,
        "test_auc": test_auc
    }
