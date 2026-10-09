"""
공통 데이터 파이프라인 (모델 비의존) — PRISM temporal P_c/P_u 분할 적용.

data/{gossip,pol}_enriched.csv 1벌에서 세 모델이 공유할 산출물을 만든다.

핵심 설계 (PRISM 논문 §3 temporal 분할을 세 모델 공통 적용):
  * 뉴스를 첫 등장 시각 기준으로 분할:
      - P_c (앞 CLS_RATIO) = 분류기 풀. PRISM 진위 분류기 학습 전용.
      - P_u (뒤 나머지)    = 추천 풀. 세 모델의 랭킹 라이브러리·시퀀스·타깃.
      - P_c ∩ P_u = ∅ (data leakage 방지, 논문 §3).
      - 유저 없는 뉴스(추천 대상 불가) → P_c.
  * news_index·ve라시티·극성·시퀀스·인스턴스는 전부 **P_u 한정**.
    → REC4MIT/HDInt도 P_u 라이브러리에서만 랭킹 (풀 크기 공통화 = 공정 비교).
  * P_c 뉴스 id+라벨은 pc_news.json으로 따로 저장 (PRISM 분류기가 사용; 나머지 모델은 무시).
  * CLS_RATIO는 논문 미명시 → 튜닝 파라미터 (스탠드얼론 기본 0.40).

산출물 (processed/{ds}/shared/):
  - news_index.json : {news_id: idx}   (P_u, idx 1..M, 0=패딩)
  - veracity.json   : [M+1] 0(real)/1(fake), idx0=패딩(0)
  - polarity.json   : [M+1] 0/1/2, idx0=패딩(1 중립)
  - splits.json     : {train/val/test: [[user_idx, [ctx_idx...], target_idx], ...]}
  - user_index.json : {user_id: idx}   (idx 1..U, 0=패딩)
  - pc_news.json    : {news_id: label}  (P_c, PRISM 분류기용)
  - meta.json       : {dataset, num_news(=|P_u|), num_users, pc_size, cls_ratio, ...}

라벨 규약: 0=real, 1=fake.
"""

import ast
import json
import os
import random
from collections import defaultdict

import pandas as pd

from .polarity import extract_domain, lookup_polarity, NEUTRAL

CTX_CAP = 10          # context 최대 길이 (PRISM=10; REC4MIT/HDInt는 4로 절삭)
SPLIT = (0.8, 0.1, 0.1)
SEED = 42
CLS_RATIO = 0.40      # P_c(앞)/P_u(뒤) temporal 분할 비율. 논문 미명시 → 튜닝 파라미터.

DATASETS = {
    "gossip": {"raw_prefix": "gossipcop"},
    "pol":    {"raw_prefix": "politifact"},
}


def load_news(root: str, dataset: str) -> pd.DataFrame:
    """enriched CSV 우선 로드 (news_url/text 포함). 없으면 원본 CSV fallback."""
    enriched = os.path.join(root, "data", f"{dataset}_enriched.csv")
    base = os.path.join(root, "data", "원본", f"{dataset}.csv")
    path = enriched if os.path.exists(enriched) else base
    df = pd.read_csv(path)
    df["news_id"] = df["news_id"].astype(str)
    df = df.drop_duplicates(subset="news_id", keep="first").reset_index(drop=True)
    return df


def _news_first_time(df: pd.DataFrame) -> dict:
    """뉴스별 첫 등장 시각 = min(user_times). 유저/시각 없으면 None."""
    first = {}
    for _, row in df.iterrows():
        try:
            times = ast.literal_eval(row["user_times"])
        except (ValueError, SyntaxError, KeyError):
            times = []
        first[row["news_id"]] = min(times) if times else None
    return first


def temporal_split(df: pd.DataFrame, cls_ratio: float = CLS_RATIO):
    """
    시간순 분할 (논문 §3): 첫 등장 앞 cls_ratio = P_c(분류기), 뒤 = P_u(추천).
    유저 없는 뉴스(추천 대상 불가) → P_c. P_c ∩ P_u = ∅.
    반환: pc_ids(set), pu_ids(list, 시간순 정렬).
    """
    first = _news_first_time(df)
    timed = sorted([n for n, t in first.items() if t is not None],
                   key=lambda n: first[n])
    no_time = [n for n, t in first.items() if t is None]
    cut = int(len(timed) * cls_ratio)
    pc_ids = set(timed[:cut]) | set(no_time)
    pu_ids = timed[cut:]                       # 시간순 유지
    return pc_ids, pu_ids


def build_index(df: pd.DataFrame, pu_ids: list):
    """P_u 뉴스만 idx 1..M (0=패딩) 부여. veracity 리스트 동시 생성."""
    lab = df.set_index("news_id")["label"].to_dict()
    news2idx = {nid: i + 1 for i, nid in enumerate(pu_ids)}
    veracity = [0] * (len(pu_ids) + 1)         # idx0 = 패딩
    for nid, idx in news2idx.items():
        veracity[idx] = int(lab[nid])
    return news2idx, veracity


def build_user_sequences(df: pd.DataFrame, news2idx: dict):
    """
    user_ids/user_times(뉴스별 유저+시각)를 뒤집어 유저별 시간순 뉴스 idx 시퀀스로 변환.
    P_u 뉴스(news2idx에 있는 것)만 유지, 중복은 첫 등장만. 길이>=2 유저만.
    반환: {user_id: [news_idx ...]}, user2idx
    """
    user_events = defaultdict(list)  # uid -> [(news_idx, time)]
    for _, row in df.iterrows():
        nid = row["news_id"]
        if nid not in news2idx:                # P_c 뉴스는 시퀀스에서 제외
            continue
        nidx = news2idx[nid]
        try:
            users = ast.literal_eval(row["user_ids"])
            times = ast.literal_eval(row["user_times"])
        except (ValueError, SyntaxError):
            continue
        for u, t in zip(users, times):
            user_events[str(u)].append((nidx, t))

    user_seq = {}
    for uid, evs in user_events.items():
        seen, ordered = set(), []
        for nidx, _ in sorted(evs, key=lambda x: x[1]):
            if nidx not in seen:
                seen.add(nidx)
                ordered.append(nidx)
        if len(ordered) >= 2:  # 최소 이력1 + 타깃1
            user_seq[uid] = ordered

    user2idx = {uid: i + 1 for i, uid in enumerate(sorted(user_seq))}  # 0=패딩
    return user_seq, user2idx


def build_instances(user_seq: dict, user2idx: dict):
    """
    슬라이딩 윈도우 인스턴스: 각 유저의 t=1..n-1 에 대해
      (user_idx, context=seq[max(0,t-CTX_CAP):t], target=seq[t])
    """
    instances = []
    for uid, seq in user_seq.items():
        uidx = user2idx[uid]
        for t in range(1, len(seq)):
            ctx = seq[max(0, t - CTX_CAP):t]
            instances.append([uidx, ctx, seq[t]])
    return instances


def split_instances(instances: list, seed: int = SEED):
    """인스턴스 단위 랜덤 80/10/10 분할."""
    rng = random.Random(seed)
    idx = list(range(len(instances)))
    rng.shuffle(idx)
    n = len(idx)
    n_test = int(n * SPLIT[2])
    n_val = int(n * SPLIT[1])
    test_i = idx[:n_test]
    val_i = idx[n_test:n_test + n_val]
    train_i = idx[n_test + n_val:]
    pick = lambda ids: [instances[i] for i in ids]
    return pick(train_i), pick(val_i), pick(test_i)


def build_shared(root: str, dataset: str, out_dir: str, cls_ratio: float = CLS_RATIO):
    """공통 산출물 생성 + 저장 (temporal P_c/P_u 분할). 통계 dict 반환."""
    os.makedirs(out_dir, exist_ok=True)
    df = load_news(root, dataset)
    lab = df.set_index("news_id")["label"].to_dict()

    # 1) temporal 분할 → P_c(분류기) / P_u(추천)
    pc_ids, pu_ids = temporal_split(df, cls_ratio)

    # 2) P_u 한정 인덱스 + veracity
    news2idx, veracity = build_index(df, pu_ids)

    # 3) 극성 (P_u 한정; enriched CSV의 news_url → 도메인 → 0/1/2). URL 없으면 중립(1).
    polarity = [1] * (len(pu_ids) + 1)  # idx0 = 중립 패딩
    pol_stats = {0: 0, 1: 0, 2: 0, "no_url": 0}
    has_url = "news_url" in df.columns
    url_map = df.set_index("news_id")["news_url"].to_dict() if has_url else {}
    for nid, idx in news2idx.items():
        url = str(url_map.get(nid, "")).strip() if pd.notna(url_map.get(nid)) else ""
        if not url:
            pol_stats["no_url"] += 1
            polarity[idx] = NEUTRAL
            continue
        pol = lookup_polarity(extract_domain(url))
        polarity[idx] = pol
        pol_stats[pol] += 1

    # 4) 유저 시퀀스(P_u) → 인스턴스 → 분할
    user_seq, user2idx = build_user_sequences(df, news2idx)
    instances = build_instances(user_seq, user2idx)
    train, val, test = split_instances(instances)

    # 5) P_c 뉴스 id+라벨 (PRISM 분류기 전용)
    pc_news = {nid: int(lab[nid]) for nid in sorted(pc_ids)}

    meta = {
        "dataset": dataset,
        "num_news": len(pu_ids),          # 추천 라이브러리 = P_u (코드가 참조하는 키)
        "pu_size": len(pu_ids),           # = num_news, 명시용 (추천 풀 P_u)
        "num_users": len(user2idx),
        "ctx_cap": CTX_CAP,
        "cls_ratio": cls_ratio,
        "pc_size": len(pc_ids),           # 분류기 풀 P_c
        "total_news": len(df),            # = pu_size + pc_size
        "num_instances": len(instances),
        "veracity_dist": {"real": veracity.count(0) - 1, "fake": veracity.count(1)},
        "polarity_stats": pol_stats,
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
    }

    json.dump(news2idx, open(os.path.join(out_dir, "news_index.json"), "w"))
    json.dump(veracity, open(os.path.join(out_dir, "veracity.json"), "w"))
    json.dump(polarity, open(os.path.join(out_dir, "polarity.json"), "w"))
    json.dump({"train": train, "val": val, "test": test},
              open(os.path.join(out_dir, "splits.json"), "w"))
    json.dump(user2idx, open(os.path.join(out_dir, "user_index.json"), "w"))
    json.dump(pc_news, open(os.path.join(out_dir, "pc_news.json"), "w"))
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=2)
    return meta
