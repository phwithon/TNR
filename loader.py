"""
공통 데이터 로더 — processed/{ds}/shared/ 산출물을 읽어 세 모델에 공급.

핵심: 세 모델이 동일한 인스턴스/분할/라벨/후보 라이브러리를 공유한다.
각 인스턴스 = [user_idx, [ctx_idx...], target_idx] (ctx는 CTX_CAP까지).
모델별 max_len으로 좌측 패딩 절삭해서 (ctx[L], mask[L], target, user) 텐서 생성.
"""

import json
import os

import torch
from torch.utils.data import Dataset


def load_shared(proc_dir: str, dataset: str) -> dict:
    d = os.path.join(proc_dir, dataset, "shared")
    load = lambda n: json.load(open(os.path.join(d, n), encoding="utf-8"))
    return {
        "news_index": load("news_index.json"),
        "veracity": load("veracity.json"),     # [N+1] 0=real,1=fake
        "polarity": load("polarity.json"),     # [N+1] 0/1/2
        "splits": load("splits.json"),         # {train,val,test: [[u,[ctx],tgt]]}
        "user_index": load("user_index.json"),
        "meta": load("meta.json"),
    }


def pad_ctx(ctx: list, max_len: int):
    """좌측 0-패딩으로 길이 max_len 정렬. (ids[max_len], mask[max_len] bool) 반환."""
    ids = ctx[-max_len:]
    pad = max_len - len(ids)
    return ([0] * pad + ids, [False] * pad + [True] * len(ids))


class SeqDataset(Dataset):
    """인스턴스 리스트 → (ctx_ids[L], ctx_mask[L], target, user) 텐서."""

    def __init__(self, instances: list, max_len: int):
        self.inst = instances
        self.max_len = max_len

    def __len__(self):
        return len(self.inst)

    def __getitem__(self, i):
        u, ctx, tgt = self.inst[i]
        ids, mask = pad_ctx(ctx, self.max_len)
        return (torch.tensor(ids, dtype=torch.long),
                torch.tensor(mask, dtype=torch.bool),
                torch.tensor(tgt, dtype=torch.long),
                torch.tensor(u, dtype=torch.long))


def veracity_tensor(shared) -> torch.Tensor:
    return torch.tensor(shared["veracity"], dtype=torch.long)


def polarity_tensor(shared) -> torch.Tensor:
    return torch.tensor(shared["polarity"], dtype=torch.long)
