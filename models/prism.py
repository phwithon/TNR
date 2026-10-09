"""
PRISM [SIGIR'25] 통합 어댑터.

모델 정의는 원본 guided_diffusion.py를 그대로 패키지에 복사(_prism_core.py)해 사용.
여기서는 공통 데이터(시퀀스/분할/진위 라벨 + prism 임베딩 [N+1,768] + cls_data)에 맞춘
2-phase 학습(분류기→diffusion)과 sample() 기반 rank_all을 구현한다.

PRISM 규약 주의:
  - 시퀀스는 **우측 패딩** (extract_axis_1이 len-1 위치=마지막 실토큰을 뽑음)
  - attn mask: True=유효 (원본 Attention.masked_fill(~mask) 규약)
  - max_len = 10
"""

import copy
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .base import BaseRecModel
from ._prism_core import Diffusion, ModelWithEmbeddingIB

MAX_LEN = 10


def _right_pad(ctx, max_len):
    ids = ctx[-max_len:]
    ids = ids + [0] * (max_len - len(ids))
    mask = [1 if x != 0 else 0 for x in ids]
    return ids, mask


class _RecDS(Dataset):
    """(tokens[max_len], target[1], mask[max_len] bool, embeddings[max_len,768])."""

    def __init__(self, instances, emb_table, max_len):
        self.inst = instances
        self.emb = emb_table           # [N+1,768] CPU tensor
        self.max_len = max_len

    def __len__(self):
        return len(self.inst)

    def __getitem__(self, i):
        _, ctx, tgt = self.inst[i]
        ids, mask = _right_pad(list(ctx), self.max_len)
        ids_t = torch.tensor(ids, dtype=torch.long)
        return (ids_t, torch.tensor([tgt], dtype=torch.long),
                torch.tensor(mask, dtype=torch.bool), self.emb[ids_t])


class _ClsDS(Dataset):
    def __init__(self, cls_data):
        self.d = cls_data

    def __len__(self):
        return len(self.d)

    def __getitem__(self, i):
        return self.d[i]["embedding"], self.d[i]["label"]


class PRISMModel(BaseRecModel):
    name = "prism"

    def __init__(self, shared, emb_dir, args, device):
        super().__init__(shared, emb_dir, args, device)
        with open(os.path.join(emb_dir, "emb.pkl"), "rb") as f:
            self.emb_table = pickle.load(f)["embedding"].float()   # [N+1,768]
        with open(os.path.join(emb_dir, "cls_data.pkl"), "rb") as f:
            self.cls_data = pickle.load(f)
        self.veracity = torch.tensor(shared["veracity"], dtype=torch.long)

        self.diffusion = Diffusion(
            timesteps=args.get("timesteps", 500),
            beta_start=args.get("beta_start", 1e-4), beta_end=args.get("beta_end", 0.02),
            hyper_w=args.get("w", 21.0), fusion_mode=args.get("fusion_mode", "seqattn"),
            max_len=MAX_LEN, p=args.get("p", 0.1)).to(device)
        self.classifier = ModelWithEmbeddingIB(
            args.get("input_dim", 768), args.get("hidden_dim", 128),
            args.get("bottleneck_dim", 128), 2).to(device)
        # news_embeddings 테이블 주입 (freeze)
        self.news_emb = nn.Embedding.from_pretrained(self.emb_table, freeze=True).to(device)
        self.diffusion.model.news_embeddings = self.news_emb
        self.diffusion.model.device = device

    def _set_news_emb(self):
        self.diffusion.model.news_embeddings = self.news_emb

    def fit(self, train, val, log):
        dev = self.device; a = self.args
        # ── Phase 1: 진위 분리기(classifier) ──
        cls_dl = DataLoader(_ClsDS(self.cls_data), batch_size=a.get("batch_size", 256),
                            shuffle=True)
        opt_c = torch.optim.Adam(self.classifier.parameters(), lr=a.get("lr_cls", 1e-3))
        phi, tau, lam_r = a.get("phi", 0.8), a.get("tau", 0.4), a.get("lambda_r", 1.0)
        ce = max(1, a.get("class_epoch", 30))
        for ep in range(ce):
            self.classifier.train(); tot = 0.0; acc = 0.0; n = 0
            for x, y in tqdm(cls_dl, desc=f"[prism] P1 {ep+1}/{ce}", leave=False):
                x, y = x.to(dev), y.long().to(dev)
                xr, zl, zr, zi, ot, xo, closs, logits = self.classifier(x, y)
                loss = F.cross_entropy(logits, y) + phi * ot + tau * closs + lam_r * F.mse_loss(xr, xo)
                opt_c.zero_grad(); loss.backward(); opt_c.step()
                tot += loss.item(); acc += (logits.argmax(1) == y).float().mean().item(); n += 1
            log(f"[prism] P1 cls {ep+1}/{ce}  loss {tot/n:.3f}  acc {acc/n:.3f}")
        for p in self.classifier.parameters():
            p.requires_grad = False
        self.classifier.eval()

        # ── Phase 2: diffusion ──
        self._set_news_emb()
        tr_dl = DataLoader(_RecDS(train, self.emb_table, MAX_LEN),
                           batch_size=a.get("batch_size", 256), shuffle=True)
        opt_d = torch.optim.Adam(self.diffusion.parameters(), lr=a.get("lr", 1e-3))
        diff_cof = a.get("diff_cof", 0.1)
        epochs = a.get("epochs", 150) - ce if a.get("epochs", 150) > ce else a.get("diff_epochs", 30)
        e_real = self.classifier.label_embedding(torch.tensor([0]).to(dev))
        val_every = a.get("val_every", 10)   # 500-step 역확산 검증이 비싸 주기적으로만
        best = -1.0; best_sd = None
        for ep in range(epochs):
            self.diffusion.train(); tot = 0.0; n = 0
            for tokens, target, mask, emb in tqdm(tr_dl, desc=f"[prism] P2 {ep+1}/{epochs}", leave=False):
                tokens, target = tokens.to(dev), target.to(dev)
                mask, emb = mask.to(dev), emb.to(dev)
                loss_d, logits = self.diffusion.p_losses(
                    list(tokens), mask.sum(dim=1, keepdim=True), target,
                    (emb, mask), authenticity=e_real)
                loss = loss_d + diff_cof * F.cross_entropy(logits, target.squeeze(-1))
                opt_d.zero_grad(); loss.backward(); opt_d.step()
                tot += loss.item(); n += 1
            if (ep + 1) % val_every == 0 or ep == epochs - 1:
                comp = self._val_hr(val)
                log(f"[prism] P2 diff {ep+1}/{epochs}  loss {tot/n:.3f}  val comp {comp:.4f}")
                if comp > best:
                    best = comp; best_sd = copy.deepcopy(self.diffusion.state_dict())
            else:
                log(f"[prism] P2 diff {ep+1}/{epochs}  loss {tot/n:.3f}")
        if best_sd is not None:
            self.diffusion.load_state_dict(best_sd)

    # PRISM은 분류기(Phase1)+diffusion(Phase2) 두 모듈 → 한 파일에 함께 저장.
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"classifier": self.classifier.state_dict(),
                    "diffusion": self.diffusion.state_dict()}, path)

    def load(self, path: str) -> None:
        sd = torch.load(path, map_location=self.device, weights_only=True)
        self.classifier.load_state_dict(sd["classifier"])
        self.diffusion.load_state_dict(sd["diffusion"])
        self.classifier.eval(); self.diffusion.eval()
        self._set_news_emb()   # news_emb 테이블 재주입 (freeze)

    @torch.no_grad()
    def _val_hr(self, val, sample=300):
        """원본 train.py 모델 선택 기준: composite = HR@5 x RealRatio@5.
        (진위 붕괴 체크포인트는 RealRatio@5=0 → composite=0으로 자동 탈락.)"""
        dev = self.device
        self._set_news_emb(); self.diffusion.eval()
        v = val[:sample]
        dl = DataLoader(_RecDS(v, self.emb_table, MAX_LEN), batch_size=128)
        e_real = self.classifier.label_embedding(torch.tensor([0]).to(dev))
        e_fake = self.classifier.label_embedding(torch.tensor([1]).to(dev))
        ver = self.veracity.to(dev)
        hit5 = 0.0; real5 = 0.0; tot = 0
        for tokens, target, mask, emb in dl:
            tokens, target, mask, emb = tokens.to(dev), target.to(dev), mask.to(dev), emb.to(dev)
            _, scores = self.diffusion.sample(list(tokens), mask.sum(dim=1, keepdim=True),
                                              target, (emb, mask), e_real, e_fake)
            top5 = scores.topk(5, dim=1).indices                  # [B,5]
            hit5 += (top5 == target).any(dim=1).sum().item()
            real5 += (ver[top5] == 0).float().mean(dim=1).sum().item()
            tot += target.size(0)
        hr5 = hit5 / max(tot, 1); rr5 = real5 / max(tot, 1)
        return hr5 * rr5

    @torch.no_grad()
    def rank_all(self, ctx_ids, ctx_mask, user):
        """ctx_ids/ctx_mask는 공통 로더(좌측패딩, True=유효). PRISM용으로 우측패딩 재구성."""
        dev = self.device
        self._set_news_emb(); self.diffusion.eval()
        B = ctx_ids.size(0)
        toks = []; masks = []
        for i in range(B):
            valid = ctx_ids[i][ctx_mask[i]].tolist()       # 실제 뉴스 idx (순서 유지)
            ids, mk = _right_pad(valid, MAX_LEN)
            toks.append(ids); masks.append([bool(x) for x in mk])
        tokens = torch.tensor(toks, dtype=torch.long, device=dev)
        mask = torch.tensor(masks, dtype=torch.bool, device=dev)
        emb = self.emb_table.to(dev)[tokens]
        e_real = self.classifier.label_embedding(torch.tensor([0]).to(dev))
        e_fake = self.classifier.label_embedding(torch.tensor([1]).to(dev))
        _, scores = self.diffusion.sample(list(tokens), mask.sum(dim=1, keepdim=True),
                                          tokens[:, :1], (emb, mask), e_real, e_fake)
        return scores   # [B, N+1], idx0 = -inf

    @torch.no_grad()
    def fake_prob(self):
        dev = self.device
        z_r = self.classifier.encoder_r(self.emb_table.to(dev))
        return F.softmax(self.classifier.classifier(z_r), dim=-1)[:, 1]
