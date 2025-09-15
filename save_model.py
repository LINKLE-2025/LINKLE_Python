def retrain_model(new_user_id=None, use_api_until: int = 5000):
    """
    LightFM 모델 재학습
    - 초기에는 user_api / linker_api / participate_api + 실제 user 테이블을 함께 학습
    - 일정 유저 수 이상 도달하면(user_id > use_api_until) 실제(user) 데이터만 사용
    """
    import os
    import pandas as pd
    from sqlalchemy import create_engine
    from dotenv import load_dotenv
    from lightfm import LightFM
    from lightfm.data import Dataset
    import joblib

    # 환경 변수 로딩
    load_dotenv(".env")
    load_dotenv(f".env.{os.getenv('FLASK_ENV', 'development')}.local", override=True)

    # DB 연결
    engine = create_engine(
        f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )

    # 실제 데이터
    user_true = pd.read_sql('SELECT * FROM user', con=engine)
    linker_true = pd.read_sql('SELECT * FROM linker', con=engine)
    participate_true = pd.read_sql('SELECT * FROM participate', con=engine)

    # API(더미) 데이터
    user_api = pd.read_sql('SELECT * FROM user_api', con=engine)
    linker_api = pd.read_sql('SELECT * FROM linker_api', con=engine)
    participate_api = pd.read_sql('SELECT * FROM participate_api', con=engine)

    # 유저 수 기준으로 실제만 쓸지, 섞어서 쓸지 결정
    if user_true["user_id"].max() > use_api_until:
        user_table = user_true
        linker_table = linker_true
        participate_table = participate_true
    else:
        user_table = pd.concat([user_true, user_api], ignore_index=True)
        linker_table = pd.concat([linker_true, linker_api], ignore_index=True)
        participate_table = pd.concat([participate_true, participate_api], ignore_index=True)

    # LightFM 데이터셋 구축
    dataset = Dataset()
    dataset.fit(users=user_table['user_id'], items=linker_table['linker_id'])
    (interactions, _) = dataset.build_interactions([
        (row['user_id'], row['linker_id']) for _, row in participate_table.iterrows()
    ])

    # 모델 학습
    model = LightFM(loss='warp')
    model.fit(interactions, epochs=10, num_threads=4)

    # 모델과 매핑 저장
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/lightfm_model.pkl')
    joblib.dump(dataset.mapping(), 'model/lightfm_mapping.pkl')

    print("모델과 매핑 정보 저장 완료.")
