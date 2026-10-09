# TNR(True News Recommendation) — 가짜뉴스 완화 뉴스 추천 모델 통합 비교 프레임워크

세 가지 fake-news-mitigation 뉴스 추천 모델을 **하나의 NNR-style 프레임워크**로 통합해,
**동일한 데이터·분할·추천 라이브러리·평가 지표**로 공정 비교한다.

- **REC4MIT** [WWW'22] — Veracity-aware and Event-driven Personalized News Recommendation
- **HDInt** [KDD'24] — Hierarchical and Disentangling Interest Learning Framework
- **PRISM** [SIGIR'25] — Bridging Interests and Truth (diffusion 기반)

> 세 모델 모두 **재구현(non-official)** 이며, 벤치마크 수치는 각 논문과 일치하지 않을 수 있다.
> 목표는 절대 수치 재현이 아니라 **동일 조건에서의 상대 비교**.

---

## 개요

세 모델은 공통적으로 "사용자의 다음 뉴스를 추천하되(관심사), 가짜 뉴스가 상위에 오르지
않게 억제(진실성)"하는 것을 목표로 하지만, 접근 방식이 다르다:

| 모델 | 핵심 메커니즘 | 가짜 억제 방식 |
|------|--------------|---------------|
| **REC4MIT** | 이벤트 탐지 + 이벤트 전이 메모리 + 이벤트/진위 분리(adversarial) | **하드 필터** (추론 시 가짜 후보 점수 −1e4) |
| **HDInt** | 극성→진위 직렬 분리기 + 계층 트랜스포머 + 후보-쿼리 어텐션 | 진위 분리 구조 |
| **PRISM** | IB 분리기(Phase1) + 조건부 diffusion(Phase2) + negative-prompt CFG | **소프트 가이드** (생성 유도) |

통합의 핵심은 **모델 알고리즘은 각 논문대로 보존**하되, **데이터·분할·추천 라이브러리·평가를
완전히 공유**하는 것이다. 이를 위해 각 모델은 공통 인터페이스
(`fit()` + `rank_all()`)만 구현하고, 내부 학습 루프는 원본 그대로 유지한다.

---

## 실행 요약

```bash
# 1. 전체 뉴스 임베딩 캐싱 (최초 1회, cls_ratio 무관)
python -m preprocess.prepare --step cache --dataset pol gossip

# 2. temporal P_c/P_u 분할 (cls_ratio 지정)
python -m preprocess.prepare --step shared --dataset pol gossip --cls_ratio 0.40

# 3. 모델별 임베딩 조립 (캐시 → P_u 라이브러리 + P_c 분류셋)
python -m preprocess.prepare --step rec4mit --dataset pol gossip
python -m preprocess.prepare --step hdint   --dataset pol gossip
python -m preprocess.prepare --step prism   --dataset pol gossip

# 4. 학습 + 평가 (3모델 순차 비교)
python main.py --model all --dataset pol
python main.py --model all --dataset gossip

# 5. (선택) 저장된 체크포인트로 학습 없이 평가만
python evaluate.py --model all --dataset pol
```

> cls_ratio를 바꿔 재실험할 때는 **2~4단계만** 반복하면 된다. 1단계(cache)는 cls_ratio와
> 무관하므로 1회만 돌리면 캐시에서 조립만 하므로 빠르다 (BERT/KeyBERT 재실행 없음).

---

## 프로젝트 구조

```
TNR/
├── main.py                     # 통합 진입점 — 3모델 학습+평가, 비교표 생성
├── evaluate.py                 # full-library 랭킹 평가 + 체크포인트 단독 평가 CLI
├── loader.py                   # 공통 데이터 로더 (load_shared, SeqDataset)
├── config_pol.json             # pol 하이퍼파라미터 (3모델)
├── config_gossip.json          # gossip 하이퍼파라미터
│
├── models/                     # 모델 어댑터 (알고리즘은 각 논문 원본 보존)
│   ├── base.py                 # BaseRecModel — fit/rank_all/save/load 인터페이스
│   ├── rec4mit.py              # REC4MIT 어댑터 (이벤트 메모리 + 하드 필터)
│   ├── hdint.py                # HDInt 어댑터 (분리기 + 계층 트랜스포머)
│   ├── prism.py                # PRISM 어댑터 (2단계 학습)
│   └── _prism_core.py          # PRISM 원본 모델 (diffusion + IB, 그대로 복사)
│
├── preprocess/                 # 전처리 파이프라인
│   ├── prepare.py              # 진입점 (cache / shared / 모델별 조립)
│   ├── common.py               # 공통 산출물 + temporal P_c/P_u 분할
│   ├── embed.py                # 전체 뉴스 임베딩 캐싱 + P_u/P_c 조립
│   ├── polarity.py             # 극성 라벨 (도메인 → 좌/중/우, HDInt용)
│   └── enrich.py               # 원본CSV(data/원본) + FakeNewsNet → enriched CSV (1회용; FakeNewsNet 삭제돼 재실행 불가)
│
├── data/                       # 입력 CSV
│   ├── {pol,gossip}_enriched.csv   # news_id,title,description,label,user_ids,user_times,news_url,text
│   └── 원본/                        # 원본 CSV
│
├── processed/{pol,gossip}/     # 전처리 산출물
│   ├── _cache/                 # 전체 뉴스 임베딩 (cls_ratio 무관)
│   │   ├── content.npy  [T,1536]   ST title⊕desc  (rec4mit+hdint)
│   │   ├── keyword.npy  [T,2304]   KeyBERT 3키워드 (hdint)
│   │   ├── cls.npy      [T,768]    BERT CLS       (prism)
│   │   ├── news_ids.json           행 → news_id
│   │   └── keyword_strings.json    news_id → 키워드 문자열 (검증용)
│   ├── shared/                 # 공통 산출물 (P_u 한정)
│   │   ├── news_index.json  veracity.json  polarity.json
│   │   ├── splits.json  user_index.json
│   │   ├── pc_news.json        # P_c 뉴스 id+라벨 (PRISM 분류기용)
│   │   └── meta.json           # num_news(=|P_u|), pc_size, cls_ratio ...
│   ├── rec4mit/emb.npz         # [M+1,1536]
│   ├── hdint/emb.pkl           # {content[M+1,1536], keyword[M+1,2304]}
│   └── prism/{emb.pkl, cls_data.pkl}   # P_u 라이브러리 + P_c 분류셋
│
├── logs/{model}/{dataset}/     # 학습/평가 로그 (타임스탬프별)
└── model_dir/{model}/{dataset}/   # 체크포인트 (.pth)
```

---

## 통합 설계 — 공통 프로토콜

세 모델의 학습 절차는 근본적으로 다르므로(이벤트 메모리 / 2단 분리기 / diffusion),
encode/predict를 억지로 통일하지 않는다. 대신 **공정 비교에 필요한 요소만 공유**한다.

| 공유 요소 | 내용 |
|-----------|------|
| 뉴스 인덱싱 | P_u 뉴스 1..M (0=패딩), 세 모델 동일 |
| 유저 시퀀스·인스턴스 | `[user_idx, [ctx...], target]`, 슬라이딩 윈도우, CTX_CAP=10 |
| 분할 | 인스턴스 단위 80/10/10 |
| 추천 라이브러리 | **P_u 전체** (세 모델 동일 후보에서 full-ranking) |
| 진위 라벨 | 0=real, 1=fake |
| 평가 | 진짜 타깃 인스턴스만, P_u 전수 랭킹, 공통 지표 |

**모델 인터페이스** (`models/base.py`):

```python
class BaseRecModel:
    def fit(self, train, val, log)                     # 원본 학습 루프 그대로
    def rank_all(self, ctx_ids, ctx_mask, user) -> [B, M+1]   # P_u 전체 점수
    def save(self, path) / load(self, path)            # 체크포인트
```

각 모델은 자기 max_len으로 context를 절삭한다: **REC4MIT 4 / HDInt 4 / PRISM 10**.

---

## 배경 — 왜 P_c/P_u 분할을 도입했나

PRISM 논문은 원래 뉴스를 **P_c(분류기) / P_u(추천)** 로 나누고, 추천은 **P_u 안에서만**
수행하도록 설계돼 있다. 반면 REC4MIT·HDInt는 이런 분할 개념이 없고 단일 뉴스 라이브러리를 쓴다.

통합 초기에는 **세 모델의 평가 지표를 통일**하기 위해, PRISM에서도 분할을 두지 않고
**P_c = P_u = 전체 뉴스 데이터셋**으로 가정했다 (분류기 학습·추천 랭킹 모두 전체 뉴스 사용).

그 결과 **PRISM이 gossip에서 동작하지 않았다** (HR@10 ≈ 0.0007, 랜덤 수준). 
PRISM은 "diffusion으로 임베딩을 생성해 전체 후보와 코사인 검색"하는 구조라 **후보(라이브러리)가
커지면 붕괴**하는 반면, REC4MIT/HDInt는 후보를 직접 스코어링 + 하드필터라 큰 라이브러리를 견딘다.
→ **P_c/P_u 분할의 필요성**을 시사.

따라서 PRISM을 논문대로 P_c/P_u로 분할해 P_u에서 추천·학습하도록 되돌린다. 다만 세 모델을
비교하려면 **뉴스(후보) 규모가 같아야** 하므로, REC4MIT·HDInt도 동일한 **P_u를 추천 후보로**
사용한다. 즉 세 모델이 같은 P_u 라이브러리에서 랭킹 → 공정 비교가 성립한다.
`cls_ratio`로 P_u 규모를 조절해 적정 지점을 탐색한다.

---

## 데이터 파이프라인 — temporal P_c/P_u 분할

PRISM 논문(§3)의 **temporal 분할**을 세 모델 공통 전처리로 채택한다.

```
전체 뉴스를 '첫 등장 시각(min user_times)' 순으로 정렬
 │
 ├─ 앞쪽 cls_ratio ──→ P_c (Classification pool)  ─ PRISM 진위 분류기 학습 전용
 └─ 뒤쪽 나머지     ──→ P_u (Recommendation pool) ─ 세 모델 공통 추천 라이브러리·시퀀스·타깃
                                                      (유저 없는 뉴스 → P_c)
        P_c ∩ P_u = ∅   (data leakage 방지, 논문 §3)
```

- **P_c** = PRISM Phase-1 분류기(e_real/e_fake 학습)에만 사용. REC4MIT/HDInt는 P_c를 쓰지 않음.
- **P_u** = 세 모델이 공유하는 추천 후보. news_index·veracity·polarity·시퀀스·인스턴스가 모두 P_u 한정.
- **cls_ratio** = P_c/P_u 분할 비율. **논문에 수치 미명시 → 튜닝 파라미터** (기본 0.40).

### 데이터셋 특성 — 진위의 시간 분포 (temporal 분할 적합성)

temporal 분할은 "진짜/가짜가 시간축에 고루 섞여 있다"를 암묵 전제한다. **두 데이터셋의 성격이 다르다:**

| 시간 구간 | GossipCop fake 비율 | PolitiFact fake 비율 |
|-----------|--------------------|---------------------|
| 0–20% (최초) | 41.8% | **4.2%** |
| 20–40% | 19.2% | 21.0% |
| 40–60% | 17.1% | 68.3% |
| 60–80% | 19.2% | **94.1%** |
| 80–100% (최후) | 28.4% | 77.5% |
| **전체** | real 74.9% / fake 25.1% | real 46.9% / fake 53.1% |

- **GossipCop**: 시간대별 fake 비율이 17~42%로 **비교적 고름** → temporal 분할해도 P_c/P_u 양쪽에 진짜·가짜가 섞임. **temporal 분할 적합.** (P_u는 전체가 real-heavy(~75%)라 real 비율 높음)
- **PolitiFact**: **초기=진짜, 후기=가짜로 극단 분리** (0–20% fake 4% → 60–80% fake 94%). 원본에서 진짜/가짜 등장 시각이 체계적으로 다름(수집·팩트체크 타임라인 추정).
  → temporal 분할 시 **진짜는 P_c로, 가짜는 P_u로** 갈림. cls_ratio=0.40에서 **P_c ≈ 87% 진짜 / P_u ≈ 80% 가짜**.

**함의 (PolitiFact):**
- P_c(분류기)가 진짜 편중 → PRISM 진위 분류기가 가짜 예제를 거의 못 봐 **e_fake 앵커 학습 부실** 가능 (정확도만 보면 "다 진짜" 찍어도 높게 나오는 함정).
- P_u(추천 풀)가 가짜 편중 → **RT/1-R의 real base rate가 ~0.20** (랜덤도 RT 0.20). 지표는 이 base 대비로 해석해야 함.
- 즉 **PolitiFact에는 temporal 분할이 잘 맞지 않음** — 필요 시 stratified(계층) 분할 등 대안 검토 여지 (단 PRISM §3 temporal 설계에서 벗어남).

### 2단계 임베딩 (캐시 → 조립)

뉴스별 임베딩은 cls_ratio와 무관(기사 텍스트만 의존)하므로 분리한다:

```
① cache  (1회, 느림)   전체 뉴스 임베딩 → processed/{ds}/_cache/
② shared (cls_ratio)   temporal 분할 → P_c/P_u 결정
③ 조립   (빠름)         캐시에서 news_id 조회 → P_u 라이브러리 + P_c 분류셋
```

cls_ratio를 바꿔 탐색할 때 ①을 재실행하지 않아 루프가 빠르다.

### 인코더 규약 (논문 §4.1.1)

| 모델 | 인코더 | 입력 | 차원 |
|------|--------|------|------|
| REC4MIT / HDInt | SentenceTransformer("bert-base-uncased") mean-pool, normalize | title(32) ⊕ desc(128) | 1536 |
| HDInt (추가) | + KeyBERT 3키워드 × 768 | title+desc | 2304 |
| PRISM | transformers BertModel **CLS** (frozen) | title+본문 (max 512) | 768 |

---

## 하이퍼파라미터 — 논문값 대조표

데이터셋별 `config_{dataset}.json`에서 관리(`main.py`가 로드). **논문값을 따를 것을 원칙.**
논문에 없거나 다르게 정한 값은 사유를 남긴다. (구현값은 config + 어댑터 기본값 기준, 2026-10 확인.)

### 공통 하이퍼파라미터

| 하이퍼파라미터 | 모델 | 논문값 | 구현값 | 사유 / 비고 |
|---------------|------|--------|--------|------------|
| **sequence length** (→ context) | REC4MIT | t=5 — **타깃이 길이 안에 포함** → ctx 4 | ctx 4 (`MAX_LEN`) | 일치 (§5.1) |
| | HDInt | t=5 — 타깃이 길이 안에 포함 → ctx 4 | ctx 4 | 일치 (§5.1) |
| | PRISM | 10 — **타깃은 길이 밖(별도)**, 10은 히스토리 S_u → ctx 10 | ctx 10 | 일치 (§3·§5.1). REC4MIT/HDInt는 타깃이 숫자 **안에**, PRISM은 타깃이 숫자 **밖에** — **관례 차이**, 둘 다 논문대로. `CTX_CAP=10` 저장 후 각자 절삭 |
| **학습 negative 수** | REC4MIT | 4 (2진짜+2가짜) | 4 (2+2) | 일치 (§5.1) |
| | HDInt | 4 (2진짜+2가짜) | 4 (2+2) | 일치 (§5.2.3) |
| | PRISM | 없음 (full library) | 없음 | 일치 (§5.1, negative sampling 안 함) |
| **learning rate** | REC4MIT | 0.001 | 1e-3 | 일치 (§4.4.2) |
| | HDInt | 미명시 | 3e-4 | 논문 미명시 → 튜닝값 |
| | PRISM | 미명시 | 1e-3 (lr, lr_cls) | 논문 미명시 → 원 코드 |
| **batch size** | REC4MIT | 64 | 64 | 일치 (§4.4.2) |
| | HDInt | 미명시 | 64 | 논문 미명시 |
| | PRISM | 미명시 | 256 | 원 코드 |
| **epoch** | REC4MIT | 미명시 | 15 | 논문 미명시 → HDInt(15)와 맞춤 |
| | HDInt | 15 | 15 | 일치 (§5.2.3) |
| | PRISM | 미명시 | 150 (pol) / 70 (gossip) | 원 코드 150; **gossip은 P2 loss 조기 평탄화 + 시간 고려해 70(=P1 30+P2 40)으로 축소** |

### 임베딩 차원

| 하이퍼파라미터 | 모델 | 논문값 | 구현값 | 사유 / 비고 |
|---------------|------|--------|--------|------------|
| **news embedding dim** | REC4MIT | 내부차원 미명시 | 입력 1536 / v_dim 256 · e_dim 128 | 입력 = ST(title768⊕desc768); 내부차원 원 코드 |
| | HDInt | **128** (news ID, §5.2.3) | **embed_dim 128** | (이전 256 → 논문값 128로 수정) |
| | PRISM | 768 (BERT 입력) | input_dim 768 | 일치; 내부 hidden/bottleneck 128 (원 코드) |
| **user ID embedding dim** | REC4MIT | 미명시 | 128 | 원 코드 |
| | HDInt | 32 (§5.2.3) | 32 (`use_user_id=True`) | 일치 |
| | PRISM | 없음 | 없음 | PRISM은 유저 ID 임베딩 없이 시퀀스로만 모델링 |

### 모델 고유 하이퍼파라미터

| 하이퍼파라미터 | 모델 | 논문값 | 구현값 | 사유 / 비고 |
|---------------|------|--------|--------|------------|
| **latent event 수 k** | REC4MIT | pol 10 / gos 20 (§4.2.1) | pol 10 / gos 20 | 일치 |
| **λ** (next-news loss) | HDInt | pol 1.3 / gos 10 (§5.5) | 1.3 / 10 | 일치 |
| **γ** (polarity loss) | HDInt | 3 (§5.5) | 3 | 일치 |
| **num_heads** | HDInt | 4 (§5.5) | 4 | 일치 (1-layer transformer) |
| **w** (CFG guidance) | PRISM | 영어 ≈ 21 (§5.6) | 21 | 일치 (pol도 영어라 21; 이전 5는 중국어 값이라 수정) |
| **timesteps** | PRISM | 미명시 | 500 | 원 코드 |
| **τ (contrastive, λ_c)** | PRISM | 항 정의·값 미명시 (§4.1) | 0.4 (config `tau`) | 논문 수치 없음 → 원 코드 |
| **τ (retrieval, InfoNCE)** | PRISM | 없음 | 0.07 (`_prism_core`) | **이방성 BERT CLS에서 순수 내적이 허브 붕괴를 일으켜, 이를 막기 위해 추가** (cosine 정규화 + InfoNCE 온도, 논문 외) |
| **φ** (OT weight λ_OT) | PRISM | 항 정의·값 미명시 | 0.8 | 원 코드 |
| **λ_r** (recon weight) | PRISM | 항 정의·값 미명시 | 1.0 | 원 코드 |
| **diff_cof** (λ_rec) | PRISM | 항 정의·값 미명시 (§4.2) | 0.1 | 원 코드 |
| **p** (CFG dropout) | PRISM | 미명시 | 0.1 | 원 코드 |
| **class_epoch** | PRISM | 미명시 | 30 | 원 코드 (Phase 1 종료 지점) |
| **beta_start / end** | PRISM | 미명시 | 1e-4 / 0.02 (linear) | 원 코드 noise schedule |

> 요약: lr·batch·negative·k·λ·γ·heads·w·**embed_dim** 등 **논문 명시값은 전부 일치**.
> 나머지(epoch·timesteps·φ·τ·diff_cof 등)는 **논문 미명시 → 원 코드값**.

---

## 평가 지표 (PRISM 기준)

진짜 타깃 인스턴스만 대상으로, **P_u 전체 라이브러리 랭킹**. K ∈ {5, 10, 20}.

| 지표 | 설명 | 방향 |
|------|------|------|
| **HR@K** | Hit Rate — 상위 K에 정답 포함 비율 | ↑ |
| **NDCG@K** | 순위 가중 정확도 | ↑ |
| **RT@K** | Real News Ratio — 상위 K 중 진짜 비율 | ↑ |
| **FNSR@K** | Weighted Fake News Suppression (w(i)=K−i+1, 상위일수록 패널티↑) | ↑ |
| **F1@K** | 2·HR·FNSR / (HR+FNSR) — 정확도·억제 통합 | ↑ |
| **1-R** | Real News Priority — top-1이 진짜인 비율 | ↑ |

**모델 선택 기준** (val 기준, 각 원본 방식 그대로):
- REC4MIT: val **loss 최소**
- HDInt: val **HR@10 최대**
- PRISM: **composite = HR@5 × RealRatio@5 최대** (진위 붕괴 체크포인트 자동 탈락)

---

## 실험 결과

### 중간 점검 — P_u/P_c 분리 **이전** (전체 뉴스 라이브러리에서 full-ranking)

> 전체 뉴스를 단일 라이브러리로 두고(= P_c/P_u 미분할) 학습·평가한 결과. (PRISM w=21).
> **PRISM이 gossip에서 붕괴(HR@10 0.0010 ≈ 랜덤)** 한 것이 이후 P_u/P_c 분할 도입의 계기.
> (REC4MIT `k`, HDInt `λ/γ` 등 일부 config는 이 시점 기준이며 이후 보정됨 → 분할 후 결과로 갱신 예정.)

**PolitiFact** (전체 라이브러리 = 597)

| Model | HR@10 | NDCG@10 | RT@10 | FNSR@10 | F1@10 | 1-R |
|-------|-------|---------|-------|---------|-------|-----|
| **REC4MIT** | **0.5348** | 0.3086 | 1.0000 | 1.0000 | 0.6969 | 1.0000 |
| HDInt | 0.4361 | 0.2231 | 1.0000 | 1.0000 | 0.6074 | 1.0000 |
| PRISM | 0.2210 | 0.1228 | 0.5898 | 0.6509 | 0.3300 | 0.8476 |

**GossipCop** (전체 라이브러리 = 17,527)

| Model | HR@10 | NDCG@10 | RT@10 | FNSR@10 | F1@10 | 1-R |
|-------|-------|---------|-------|---------|-------|-----|
| **REC4MIT** | **0.2770** | 0.1309 | 1.0000 | 1.0000 | 0.4338 | 1.0000 |
| HDInt | 0.1077 | 0.0564 | 0.9959 | 0.9957 | 0.1943 | 0.9963 |
| PRISM | 0.0010 | 0.0006 | 0.7641 | 0.7669 | 0.0020 | 0.7973 |

> REC4MIT/HDInt의 RT/FNSR/1-R ≈ 1.0은 **추론 시 가짜 하드필터** 때문(성능 아닌 구조적 보장).
> PRISM은 소프트 가이드라 RT가 낮게 나옴. 지표는 real base rate(gossip ≈ 75% real) 대비로 해석.

### P_u/P_c 분할 **이후** (작성 예정)

분할 + 보정 config로 full training 후 갱신. 재현: `python main.py --model all --dataset {pol,gossip}`.

**PolitiFact** (cls_ratio TBD) · **GossipCop** (cls_ratio = 0.80, P_u = 3,506)

| Model | HR@10 | NDCG@10 | RT@10 | FNSR@10 | F1@10 | 1-R |
|-------|-------|---------|-------|---------|-------|-----|
| REC4MIT | — | — | — | — | — | — |
| HDInt | — | — | — | — | — | — |
| PRISM | — | — | — | — | — | — |

### (선택) cls_ratio 스윕 — 라이브러리 규모별 추세

P_u 규모에 따른 세 모델 견고성 비교 (PRISM은 규모↑ 시 급락, REC4MIT/HDInt는 완만 예상).

| dataset | cls_ratio | P_u | REC4MIT HR@10 | HDInt HR@10 | PRISM HR@10 |
|---------|-----------|-----|---------------|-------------|-------------|
| gossip | 0.40 | 10,517 | — | — | — |
| gossip | 0.60 | ~7,000 | — | — | — |
| gossip | 0.80 | 3,506 | — | — | — |

---

## 로그 및 체크포인트

- 학습/평가 로그: `logs/{model}/{dataset}/{model}_{timestamp}.log` (ppp 양식)
- 비교표: `logs/comparison/{dataset}/comparison_{timestamp}.log` (`--model all`)
- 체크포인트: `model_dir/{model}/{dataset}/{model}_{timestamp}.pth`
  - PRISM은 classifier+diffusion 두 모듈을 한 파일에 저장
  - 저장되는 건 val 기준 **최적 체크포인트** (마지막 epoch 아님)

동일 `timestamp`를 공유하므로 로그와 체크포인트가 1:1 매칭된다.

---

## 데이터셋 통계 (참고)

| | 전체 뉴스 | 유저 | 라벨(real/fake) |
|---|----------|------|----------------|
| pol | 597 | — | — |
| gossip | 17,527 | 50,822 | — |

> P_u/P_c 규모는 cls_ratio에 따라 달라진다. 예: gossip cls_ratio=0.40 → P_c 7,010 / P_u 10,517.
> `python -m preprocess.prepare --step shared`가 분할 통계를 출력한다.

---

## 참고

- REC4MIT: [WWW'22] Veracity-aware and Event-driven Personalized News Recommendation for Fake News Mitigation
- HDInt: [KDD'24] A Hierarchical and Disentangling Interest Learning Framework for Unbiased and True News Recommendation
- PRISM: [SIGIR'25] Bridging Interests and Truth, Towards Mitigating Fake News with Personalized and Truthful Recommendations
