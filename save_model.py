# train_loop.py 파일에서 일정 주기마다 학습 실행
def retrain_model():
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

    # 데이터 로드

    # 테스트용 더미 데이터 로드
    user_table_api = pd.read_sql('SELECT * FROM user_api', con=engine)
    linker_table_api = pd.read_sql('SELECT * FROM linker_api', con=engine)
    participate_table_api = pd.read_sql('SELECT * FROM participate_api', con=engine)

    # 실제 고객 데이터 로드
    user_table_true = pd.read_sql('SELECT * FROM user', con=engine)
    linker_table_true = pd.read_sql('SELECT * FROM linker', con=engine)
    participate_table_true = pd.read_sql('SELECT * FROM participate', con=engine)

    # 실제 참여 데이터 건수 기준으로 분기
    if len(participate_table_true) > len(participate_table_api):
        user_table = user_table_true
        linker_table = linker_table_true
        participate_table = participate_table_true
    else:
        user_table = pd.concat([user_table_true, user_table_api], ignore_index=True)
        linker_table = pd.concat([linker_table_true, linker_table_api], ignore_index=True)
        participate_table = pd.concat([participate_table_true, participate_table_api], ignore_index=True)

    
    # LightFM 데이터셋 구축
    dataset = Dataset()
    dataset.fit(users=user_table_api['user_id'], items=linker_table_api['linker_id'])

    (interactions, _) = dataset.build_interactions([
        (row['user_id'], row['linker_id']) for _, row in participate_table_api.iterrows()
    ])

    # 모델 학습
    model = LightFM(loss='warp')
    model.fit(interactions, epochs=10, num_threads=4)

    # 모델과 매핑 저장
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/lightfm_model.pkl')
    joblib.dump(dataset.mapping(), 'model/lightfm_mapping.pkl')

    print("모델과 매핑 정보 저장 완료.")