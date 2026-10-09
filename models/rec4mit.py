"""
REC4MIT [WWW'22] 이식 + 통합 어댑터.

원본(REC4MIT/rec4mit.py, Model/layer1-3.py)의 핵심 nn.Module을 그대로 옮기고,
공통 로더(공유 시퀀스/라벨 + rec4mit 임베딩 테이블 [N+1,1536])에 맞춘
fit()/rank_all()을 제공한다.

원본과의 차이(통합 때문):
  - EmbeddingLayer init을 외부 [N+1,1536] 테이블로 대체 (원본 init_emb 우회)
  - 데이터는 공통 분할/라벨 사용 (원본 10-fold → 통합 단일 분할)
  - 후보: 1정답 + 진짜2 + 가짜2 (원본과 동일 샘플링)
  - rank_all: 전체 라이브러리 N개 후보 점수 (통합 평가용)
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base import BaseRecModel
from loader import SeqDataset

MAX_LEN = 4   # 원본 ctx_len


# ───────────────────── 원본 모듈 이식 (layer1-3) ─────────────────────

class EmbeddingLayer(nn.Module):
    def __init__(self, emb, id_dim=128, out_dim=256):
        super().__init__()
        num_news, meta_dim = emb.shape
        self.id_emb = nn.Embedding(num_news, id_dim, padding_idx=0)
        self.meta_emb = nn.Embedding(num_news, meta_dim, padding_idx=0)
        with torch.no_grad():
            self.meta_emb.weight.copy_(emb)
        self.fc_v = nn.Linear(id_dim + meta_dim, out_dim)

    def forward(self, idx):
        return self.fc_v(torch.cat([self.id_emb(idx), self.meta_emb(idx)], -1))


class Encoder(nn.Module):
    def __init__(self, in_dim=256, z1_dim=256, z2_dim=256, h_dim=256):
        super().__init__()
        self.dense1 = nn.Linear(in_dim, z1_dim)
        self.dense2 = nn.Linear(in_dim + z1_dim, z2_dim)
        self.dense3 = nn.Linear(in_dim + z2_dim, h_dim)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, in_):
        z1 = self.act(self.dense1(in_))
        z2 = self.act(self.dense2(torch.cat([in_, z1], -1)))
        h = self.act(self.dense3(torch.cat([in_, z2], -1)))
        return h


class EventDecoder(nn.Module):
    def __init__(self, in_dim=256, z1_dim=256, z2_dim=256, h_dim=256):
        super().__init__()
        self.dense1 = nn.Linear(in_dim, z1_dim)
        self.dense2 = nn.Linear(in_dim + z1_dim, z2_dim)
        self.dense3 = nn.Linear(in_dim + z2_dim, h_dim)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, in_):
        z1 = self.act(self.dense1(in_))
        z2 = self.act(self.dense2(torch.cat([in_, z1], -1)))
        h = self.act(self.dense3(torch.cat([in_, z2], -1)))
        return h


class VeracityDecoder(nn.Module):
    def __init__(self, in_dim=256, z1_dim=256, z2_dim=256, h_dim=256):
        super().__init__()
        self.dense1 = nn.Linear(in_dim, z1_dim)
        self.dense2 = nn.Linear(in_dim + z1_dim, z2_dim)
        self.dense3 = nn.Linear(in_dim + z2_dim, h_dim)
        self.denseL = nn.Linear(h_dim, 1)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, in_):
        z1 = self.act(self.dense1(in_))
        z2 = self.act(self.dense2(torch.cat([in_, z1], -1)))
        l_i = self.act(self.dense3(torch.cat([in_, z2], -1)))
        sig_input = self.denseL(l_i).squeeze(-1)
        return l_i, sig_input


class DisentangleLoss(nn.Module):
    def __init__(self, e_dim=128):
        super().__init__()
        self.dense_e = nn.Linear(e_dim, 1)

    def forward(self, v, e, l, v_sig_input, y, mask=None):
        loss_r = 0.5 * ((torch.cat([e, l], -1) - v) ** 2).mean(-1)
        loss_l = F.binary_cross_entropy_with_logits(v_sig_input, y, reduction="none")
        e_sig_input = self.dense_e(e).squeeze(-1)
        adv_err = F.binary_cross_entropy_with_logits(e_sig_input, y, reduction="none")
        loss_a = 1.0 / adv_err.clamp(min=1e-6)
        losses = loss_l + loss_r + loss_a
        if mask is not None:
            total = (losses * mask).sum() / mask.sum().clamp(min=1)
        else:
            total = losses.mean()
        return total, {"L_r": loss_r.mean().item(), "L_l": loss_l.mean().item(),
                       "L_a": loss_a.mean().item()}


class EventDetector(nn.Module):
    def __init__(self, e_dim=128, events=20):
        super().__init__()
        self.W1 = nn.Linear(e_dim, events, bias=False)

    def forward(self, e):
        beta = F.softmax(self.W1(e), dim=-1)
        e_split = beta.unsqueeze(-1) * e.unsqueeze(-2)
        return beta, e_split


class EventTransitionNet(nn.Module):
    def __init__(self, e_dim=128, events=20, ctx_len=4, pos_dim=32, attn_dim=64, user_dim=128):
        super().__init__()
        self.W2 = nn.Linear(e_dim + pos_dim, attn_dim)
        self.W3 = nn.Linear(attn_dim, 1, bias=False)
        self.W4 = nn.Linear(e_dim + user_dim, e_dim)
        self.pos_emb = nn.Embedding(ctx_len, pos_dim)

    def build_R(self, e, e_split, pad_mask=None):
        b, ctx, _ = e.shape
        p = self.pos_emb.weight
        f = torch.cat([e, p.unsqueeze(0).expand(b, -1, -1)], -1)
        gamma = self.W3(torch.tanh(self.W2(f))).squeeze(-1)
        if pad_mask is not None:
            gamma = gamma.masked_fill(~pad_mask, -1e9)
        gamma = F.softmax(gamma, dim=-1)
        R = torch.einsum("bl,blkd->bkd", gamma, e_split)
        return R, gamma

    def activate(self, R, e_t, u):
        delta = F.softmax(torch.bmm(e_t, R.transpose(1, 2)), -1)
        c = torch.einsum("bck,bkd->bcd", delta, R)
        u = u.unsqueeze(1).expand(-1, c.size(1), -1)
        c_u = torch.tanh(self.W4(torch.cat([c, u], -1)))
        return c_u, delta


class NextNewsPredictor(nn.Module):
    def forward(self, c_u, e_t):
        return (c_u * e_t).sum(-1)


class Rec4Mit(nn.Module):
    def __init__(self, init_emb, num_users, k=20, ctx_len=4,
                 v_dim=256, e_dim=128, user_dim=128):
        super().__init__()
        self.embedding = EmbeddingLayer(init_emb, out_dim=v_dim)
        self.encoder = Encoder(in_dim=v_dim, h_dim=v_dim)
        self.event_dec = EventDecoder(in_dim=v_dim, h_dim=e_dim)
        self.veracity_dec = VeracityDecoder(in_dim=v_dim, h_dim=e_dim)
        self.detector = EventDetector(e_dim, k)
        self.transition = EventTransitionNet(e_dim, k, ctx_len, user_dim=user_dim)
        self.predictor = NextNewsPredictor()
        # 통합은 user idx 1..U (0=패딩) 규약 → padding_idx=0.
        # (원본은 0-based u2i라 padding_idx=num_users였으나, 그대로 쓰면 마지막
        #  유저(idx=U)가 패딩으로 취급돼 0벡터·무학습되는 off-by-one 발생.)
        self.user_emb = nn.Embedding(num_users + 1, user_dim, padding_idx=0)
        self.dis_loss = DisentangleLoss(e_dim)

    def disentangle(self, ids):
        v = self.embedding(ids)
        h = self.encoder(v)
        e = self.event_dec(h)
        l, logit = self.veracity_dec(h)
        return v, e, l, logit

    def forward(self, seq_ids, cand_ids, u_idx, pad_mask):
        v_s, e_s, l_s, lg_s = self.disentangle(seq_ids)
        v_c, e_c, l_c, lg_c = self.disentangle(cand_ids)
        _, e_split = self.detector(e_s)
        R, _ = self.transition.build_R(e_s, e_split, pad_mask)
        c_u, _ = self.transition.activate(R, e_c, self.user_emb(u_idx))
        logits = self.predictor(c_u, e_c)
        return logits, (v_s, e_s, l_s, lg_s), (v_c, e_c, l_c, lg_c)


# ───────────────────── 통합 어댑터 ─────────────────────

class Rec4MitModel(BaseRecModel):
    name = "rec4mit"

    def __init__(self, shared, emb_dir, args, device):
        super().__init__(shared, emb_dir, args, device)
        import os
        emb = np.load(os.path.join(emb_dir, "emb.npz"))["emb"]   # [N+1,1536]
        init = torch.from_numpy(emb).float()
        # 원본 user padding_idx = num_users. 공통 user_idx는 1..U (0=없음)이므로
        # user 임베딩 크기를 num_users+1로 두고 그대로 사용.
        # k = 잠재 이벤트 수 (논문 §4.2.1: PolitiFact 10 / GossipCop 20). config에서 전달.
        self.model = Rec4Mit(init, num_users=self.num_users, k=args.get("k", 20),
                             ctx_len=MAX_LEN).to(device)
        self.veracity = torch.tensor(shared["veracity"], dtype=torch.long)
        lab = np.array(shared["veracity"])
        self.real_pool = np.where(lab[1:] == 0)[0] + 1   # idx (1..N)
        self.fake_pool = np.where(lab[1:] == 1)[0] + 1

    def _sample_cands(self, tgt: int):
        """1정답 + 진짜2 + 가짜2 (원본 FoldSet 방식)."""
        neg = np.concatenate([np.random.choice(self.real_pool, 2),
                              np.random.choice(self.fake_pool, 2)])
        return [tgt] + neg.tolist()

    def _loader(self, instances, shuffle):
        # 원본은 진짜 타깃 인스턴스만 학습. 통합에서도 동일 적용.
        inst = [x for x in instances if self.veracity[x[2]].item() == 0]
        ds = SeqDataset(inst, MAX_LEN)
        return DataLoader(ds, batch_size=self.args.get("batch_size", 64), shuffle=shuffle)

    def fit(self, train, val, log):
        dev = self.device
        dl = self._loader(train, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.args.get("lr", 1e-3))
        epochs = self.args.get("epochs", 15)
        ver = self.veracity.to(dev)
        best = float("inf"); best_sd = None
        for ep in range(epochs):
            self.model.train(); tot = 0.0
            for ids, mask, tgt, u in tqdm(dl, desc=f"[rec4mit] ep {ep+1}/{epochs}", leave=False):
                ids, mask, tgt, u = ids.to(dev), mask.to(dev), tgt.to(dev), u.to(dev)
                B = ids.size(0)
                cand = torch.stack([torch.tensor(self._sample_cands(t.item()), device=dev)
                                    for t in tgt])  # [B,5]
                logits, ctx, cd = self.model(ids, cand, u, mask)
                target = torch.zeros_like(logits); target[:, 0] = 1
                loss_p = F.binary_cross_entropy_with_logits(logits, target)
                y_s = ver[ids].float(); y_c = ver[cand].float()
                loss_d = self.model.dis_loss(*ctx, y_s, mask)[0] + self.model.dis_loss(*cd, y_c)[0]
                loss = loss_p + loss_d
                opt.zero_grad(); loss.backward()
                # 수정: gradient clipping 추가 (hdint와 동일). 적대항 1/adv_err(line 103)이
                # gossip 등 대규모 데이터에서 폭발 → 발산하는 것 방지. 논문 손실/구조/하이퍼파라미터
                # 불변(§4.4): gradient norm만 임계 초과 시 축소하는 수치 안정 장치.
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                tot += loss.item()
            vl = self._val_loss(val, ver)
            log(f"[rec4mit] epoch {ep+1}/{epochs}  train {tot/len(dl):.3f}  val {vl:.3f}")
            if vl < best:
                best = vl; best_sd = copy.deepcopy(self.model.state_dict())
        if best_sd is not None:
            self.model.load_state_dict(best_sd)

    @torch.no_grad()
    def _val_loss(self, val, ver):
        dev = self.device
        dl = self._loader(val, shuffle=False)
        self.model.eval(); tot, n = 0.0, 0
        for ids, mask, tgt, u in dl:
            ids, mask, tgt, u = ids.to(dev), mask.to(dev), tgt.to(dev), u.to(dev)
            cand = torch.stack([torch.tensor(self._sample_cands(t.item()), device=dev) for t in tgt])
            logits, ctx, cd = self.model(ids, cand, u, mask)
            target = torch.zeros_like(logits); target[:, 0] = 1
            loss = F.binary_cross_entropy_with_logits(logits, target) \
                + self.model.dis_loss(*ctx, ver[ids].float(), mask)[0] \
                + self.model.dis_loss(*cd, ver[cand].float())[0]
            tot += loss.item() * ids.size(0); n += ids.size(0)
        return tot / max(n, 1)

    @torch.no_grad()
    def rank_all(self, ctx_ids, ctx_mask, user):
        """전체 라이브러리 점수 [B, N+1]. 후보 e_c를 전부 미리 계산 후 배치별 활성화."""
        dev = self.device
        self.model.eval()
        ctx_ids, ctx_mask, user = ctx_ids.to(dev), ctx_mask.to(dev), user.to(dev)
        B = ctx_ids.size(0); N = self.num_news

        # 컨텍스트 이벤트 + 전이 메모리 R  (후보 무관, 1회)
        _, e_s, _, _ = self.model.disentangle(ctx_ids)
        _, e_split = self.model.detector(e_s)
        R, _ = self.model.transition.build_R(e_s, e_split, ctx_mask)   # [B,k,d]
        u = self.model.user_emb(user)                                   # [B,ud]

        # 모든 뉴스 e_c + 진위 로짓 (1..N) 사전 계산
        all_ids = torch.arange(1, N + 1, device=dev)
        e_all = []; fake_all = []
        for s in range(0, N, 4096):
            _, e_c, _, lg_c = self.model.disentangle(all_ids[s:s + 4096])
            e_all.append(e_c)
            fake_all.append((torch.sigmoid(lg_c) >= 0.5).float())      # 예측 가짜=1
        e_all = torch.cat(e_all, 0)                                     # [N,d]
        fake = torch.cat(fake_all, 0)                                   # [N]

        scores = torch.full((B, N + 1), float("-inf"), device=dev)
        for s in range(0, N, 4096):
            e_chunk = e_all[s:s + 4096].unsqueeze(0).expand(B, -1, -1)  # [B,c,d]
            c_u, _ = self.model.transition.activate(R, e_chunk, u)      # [B,c,d]
            sc = (c_u * e_chunk).sum(-1)                                # [B,c]
            scores[:, s + 1:s + 1 + sc.size(1)] = sc
        # 논문 §4.3 (Eq 8) / 원본 test.py: veracity 분류기로 예측-가짜 후보 억제
        scores[:, 1:] = scores[:, 1:] - fake.unsqueeze(0) * 1e4
        return scores
