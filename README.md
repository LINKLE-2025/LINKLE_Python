# LINKLE 프로젝트 통합 설명서  
React 기반 프론트엔드 + Python 추천 시스템

---

## 📑 목차
1. 프로젝트 개요
2. 프론트엔드 구성
3. AI 추천 시스템
   - LightFM 개요
   - 콜드 스타트 문제
   - Hybrid 추천기
4. 평가 지표
5. LightFM vs SVD
6. 모델 적용 방식
7. 전체 아키텍처
8. 실행 방법
9. 라이선스

---

## 🧩 프로젝트 개요

| 구분 | 설명 |
|------|------|
| 프로젝트명 | LINKLE |
| 주요 기능 | 실시간 지도, 채팅, 포스팅 기반 SNS + 추천 시스템 |
| 기술 스택 | **Full-stack**: React (Frontend) + Spring Boot (Backend) + Python (AI) |
| 목표 | 주변 사용자들과의 실시간 소통 + AI 기반 개인화 추천 제공 |
| 전략 | 모듈화 및 병렬 개발 + API 기반 연동 |
| 핵심 기술 | 지도 API, 채팅 시스템, 추천 알고리즘, 음성 TTS, 결제 연동 등 |

---

## 🌐 프론트엔드 구성 (React 기반)

### ✅ 핵심 기능

| 카테고리   | 세부 기능 |
|------------|----------|
| 지도/위치 | 카카오맵, 실시간 위치 기반 소통, 번개/클래스톡 연동 |
| 채팅 | 1:1 DM, 실시간 그룹 채팅, 알림 |
| 포스트 | 사진 + 태그 기반 개인 히스토리 |
| 검색 | 친구/장소/태그 검색 |
| 결제/구독 | Toss Payments API |
| AI/음성 | TTS 화면 낭독, AI 비서 기능 |
| 프로필 | 사용자 설정, 배경/닉네임 수정 |

### 🛠 프론트 기술 스택

| 구분 | 사용 기술 |
|------|-----------|
| 개발 언어 | TypeScript |
| 프레임워크 | React 19 (Vite 기반) |
| 스타일링 | Tailwind CSS / styled-components |
| 라우팅/상태관리 | React Router, React Context API |
| API 연동 | Axios, REST API, JWT 인증 |
| 지도 서비스 | Kakao Map API |
| 결제 | Toss Payments API |

---

## 🧠 AI 추천 시스템 구성 (Python 기반)

### 📌 목적
사용자의 관심사와 활동 이력을 기반으로 **개인 맞춤형 추천 콘텐츠 제공** (이벤트, 친구, 활동 등)

---

## 📌 Hybrid 추천 시스템 (LightFM + CF + CBF)
이 프로젝트는 **LightFM** 기반의 추천 시스템을 중심으로,  
데이터 부족 및 콜드 스타트 문제를 완화하기 위해 다양한 보조 기법을 결합한 **하이브리드 추천기**입니다.



---

## 1️⃣ LightFM 개요
**LightFM**은 협업 필터링(Collaborative Filtering)과 콘텐츠 기반 필터링(Content-based Filtering)을 결합한 하이브리드 추천 알고리즘입니다.

- **장점**
  - 사용자 feature(성별, 나이, 선호 카테고리 등)와 아이템 feature(카테고리, 태그, 텍스트 등)를 함께 학습
  - WARP, BPR 등 다양한 랭킹 기반 손실 함수를 지원 → Precision@K 최적화에 강점
  - 콜드 스타트 문제 완화 가능
- **단점**
  - 구현 복잡도가 높고 feature 엔지니어링 필요

---

## 2️⃣ 콜드 스타트 문제
추천 시스템에서 **데이터 부족으로 추천이 어려운 상황**을 의미합니다.

- **신규 유저 콜드 스타트**: 새로 가입한 유저는 아직 참여 데이터가 없음
- **신규 아이템 콜드 스타트**: 새로 추가된 링커(모임)는 참여 데이터가 없음
- **시스템 콜드 스타트**: 초기 배포 시 전체 데이터가 부족

➡️ 본 시스템에서는 **Global Popularity, Category 기반 추천, CBF(TF-IDF)** 등을 활용하여 보완합니다.

---

## 3️⃣ Hybrid 추천기 구성
`hybrid_recommend()` 함수는 LightFM과 여러 보조 기법의 점수를 가중합산하여 최종 추천을 제공합니다.

### 🔹 1) LightFM 점수
```python
scores = model.predict(
    dataset.mapping()[0][uid],                  # 유저 내부 ID
    np.arange(len(dataset.mapping()[2])),       # 전체 아이템 ID
    user_features=user_features,
    item_features=item_features
)
```
- `predict()`를 통해 **사용자-아이템 상호작용 점수**를 예측
- 결과: 유저가 모든 아이템에 대해 가질 선호도(score) 벡터

---

### 🔹 2) Global Popularity (전역 인기 기반)
```python
global_scores = linker_table.set_index("linker_id")["popularity"]
global_scores = (global_scores - global_scores.min()) / (global_scores.max() - global_scores.min() + 1e-9)
```
- 전체 참여 데이터를 기반으로 인기 있는 아이템에 높은 점수를 부여
- **장점**: 신규 유저 콜드 스타트 시 활용 가능
- **단점**: 개인화 부족 (모든 유저에게 동일 추천)

---

### 🔹 3) Category Popularity (카테고리 기반)
```python
user_pref_cat = user_table[user_table["user_id"] == user_id]["pref_cat"].iloc[0]
cat_scores = (linker_table["category_id"] == str(user_pref_cat)).astype(int)
```
- 유저가 가장 많이 참여한 **선호 카테고리(pref_cat)** 기반으로 추천
- **장점**: Cold-start 완화 (가입 시 관심사 활용 가능)
- **단점**: 하나의 카테고리만 반영하면 다중 취향을 놓칠 수 있음

---

### 🔹 4) Item-based Collaborative Filtering
```python
interaction_matrix = pd.crosstab(participate_table["user_id"], participate_table["linker_id"])
sim_matrix = cosine_similarity(interaction_matrix.T)
sim_df = pd.DataFrame(sim_matrix, index=interaction_matrix.columns, columns=interaction_matrix.columns)
sim_scores = sim_df[user_items].mean(axis=1)
```
- 아이템 간 코사인 유사도를 계산
- 유저가 참여한 아이템과 유사한 다른 아이템을 추천
- **장점**: 해석이 직관적 (“이 아이템을 본 사람은 저 아이템도 봤다”)
- **단점**: 신규 아이템은 데이터 부족 시 추천 불가

---

### 🔹 5) Content-based Filtering (TF-IDF)
```python
tfidf = TfidfVectorizer(max_features=500)
tfidf_matrix = tfidf.fit_transform(linker_table["memo"].fillna(""))
user_vec = tfidf_matrix[linker_table["linker_id"].isin(user_items)].mean(axis=0)
sim_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
cbf_scores = pd.Series(sim_scores, index=linker_table["linker_id"])
```
- 아이템의 텍스트(`memo`)를 벡터화하여 유사도 기반 추천
- 신규 아이템에도 활용 가능 → 콜드 스타트 완화
- **장점**: 아이템 메타데이터를 활용 가능
- **단점**: 텍스트 품질에 따라 성능이 달라짐

---

## 4️⃣ 최종 Hybrid 추천
최종 추천 점수는 다음과 같이 가중합산하여 계산합니다.

```python
final_scores = (
    weights["lightfm"] * lightfm_scores.add(0, fill_value=0) +
    weights["global"]  * global_scores.add(0, fill_value=0) +
    weights["category"]* cat_scores.add(0, fill_value=0) +
    weights["itemcf"]  * itemcf_scores.add(0, fill_value=0) +
    weights["cbf"]     * cbf_scores.add(0, fill_value=0)
)
```

---

## 5️⃣ 평가 지표
- **Precision@K**: 추천 상위 K개 아이템 중 실제 정답이 얼마나 포함되었는지 측정
- **AUC**: 모델이 정답을 비정답보다 더 높은 점수로 예측하는 비율
```
📊 Precision@5 (Train): 0.1427  
📊 Precision@5 (Test): 0.0137  
📊 AUC (Test): 0.6976  
```

```추천 API 예제
curl "http://localhost:8082/recommendAI?user_id=123&address_detail=서울특별시"

```

---

## 📊 요약
- **LightFM**만 사용하면 데이터 부족/콜드 스타트 문제 발생
- **Hybrid 방식**(Global Popularity + Category + ItemCF + CBF)을 추가하여 보완
- Cold-start 상황에서도 안정적인 추천 가능

---


## 🔍 SVD와 비교 상세

### 1. LightFM (Hybrid Recommendation)

- **기반**: 협업 필터링 + 콘텐츠 기반 필터링
- **사용 목적**: Cold Start 완화 및 맞춤형 추천
- **핵심 특징**:
  - 사용자/아이템 feature 사용
  - WARP 손실 함수로 Precision@K 최적화
- **활용 데이터**: 사용자 정보, 아이템 카테고리, 과거 상호작용 로그

#### 장단점 요약
| 장점 | 단점 |
|------|------|
| Cold Start 대응 | 구현 복잡 |
| 유연한 확장 | Feature 엔지니어링 필요 |

---

### 2. SVD / SVD++

- **기반**: 사용자-아이템 평점 행렬 기반 행렬 분해
- **특징**: 간단한 구조, Netflix Prize에서 성능 입증
- **단점**: Cold Start 대응 불가

---

### 3. LightFM vs SVD 비교

| 항목 | LightFM | SVD |
|------|---------|-----|
| 방식 | 하이브리드 | 협업 필터링 |
| Cold Start 대응 | ✅ 가능 | ❌ 불가 |
| 최적화 목적 | Precision@K, Ranking | RMSE, 예측 정확도 |
| 입력 데이터 | feature + 상호작용 | 상호작용만 |



## ⚙ 모델 적용 방식

| 항목 | 설명 |
|------|------|
| 모델 | `LightFM(loss='warp')` |
| 보조 로직 | Hybrid (LightFM + 인기 기반 + 카테고리 기반 + ItemCF + CBF) |
| 평가 지표 | Precision@K, AUC |
| 저장 위치 | `model/lightfm_model.pkl`, `model/hybrid_context.pkl` |
| API 라우트 | `/recommendAI`, `/evaluateAI` (Flask 기반) |

---

## 🔗 전체 아키텍처

```
[사용자]
   ↓
[React 프론트엔드]
   ↓ REST API
[Spring Boot 백엔드] ↔ JWT 인증, DB, KakaoMap API
   ↓
[Python Flask 서버]
   ↓
[추천 시스템 (LightFM + Hybrid)]
```

---

## 🚀 실행 방법

### 프론트엔드

```bash
git clone https://github.com/LINKLE-2025/LINKLE_Frontend.git
cd LINKLE_Frontend
npm install
npm run dev
```

### 백엔드 & AI 서버
- Spring Boot 서버 실행
- Python Flask 서버에서 추천 API 실행
  ```bash
  flask run --host=0.0.0.0 --port=5000
  ```

---

## 🪪 라이선스

MIT 라이선스를 따릅니다. 자세한 내용은 [LICENSE](./LICENSE)를 참고하세요.

---

> 📌 더 효율적인 문서화와 협업을 원한다면 [GPTOnline.ai](https://gptonline.ai/ko/)에서 AI 문서 자동화를 사용해보세요!
