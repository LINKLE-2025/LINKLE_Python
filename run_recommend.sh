source /opt/anaconda3/etc/profile.d/conda.sh
conda activate lightfm-py310 
export FLASK_ENV=development 

# 모델 학습
python save_model.py
# 학습 끝난 API 실
python recommend_api.py
