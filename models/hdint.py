"""
HDInt [KDD'24] 이식 + 통합 어댑터 (frozen 경로).

원본(HDInt/models/component/hdint_model.py)의 분리기·계층관심사 모듈을 그대로 옮기되,
통합에서는 BERT finetune(2-stage)은 생략하고 **frozen 임베딩 경로**만 사용한다:
  content_emb [N+1,1536], keyword_emb [N+1,2304] (processed/{ds}/hdint/emb.pkl)

공통 데이터(시퀀스/분할/진위+극성 라벨)를 쓰고, 손실은 논문대로
  L = λ·BCE + γ·(극성 분리) + (진위 분리).
rank_all은 원본 encode_all_news/score_candidates를 재활용해 전체 라이브러리 점수를 낸다.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base import BaseRecModel
from loader import SeqDataset

MAX_LEN = 4


# ───────────────────── 원본 모듈 이식 ─────────────────────

class DisentangleEncoder(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim * 2, embed_dim)
        self.fc3 = nn.Linear(embed_dim * 2, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, c_j):
        z1 = self.act(self.drop(self.fc1(c_j)))
        z2 = self.act(self.drop(self.fc2(torch.cat([c_j, z1], dim=-1))))
        h = self.act(self.drop(self.fc3(torch.cat([c_j, z2], dim=-1))))
        return h


class DisentangleDecoder(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim * 2, embed_dim)
        self.fc3 = nn.Linear(embed_dim * 2, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, h):
        d1 = self.act(self.drop(self.fc1(h)))
        d2 = self.act(self.drop(self.fc2(torch.cat([d1, h], dim=-1))))
        f = self.act(self.drop(self.fc3(torch.cat([d2, h], dim=-1))))
        return f


class _Disentangler(nn.Module):
    """극성/진위 공용 분리기 (재구성+라벨+적대 loss). num_classes로 구분."""

    def __init__(self, embed_dim, num_classes, dropout=0.1):
        super().__init__()
        self.encoder = DisentangleEncoder(embed_dim, dropout)
        self.comp_decoder = DisentangleDecoder(embed_dim, dropout)   # 성분(극성 p / 진위 v)
        self.free_decoder = DisentangleDecoder(embed_dim, dropout)   # 제거본(f / e_clean)
        self.recon_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, labels=None, valid_mask=None):
        h = self.encoder(x)
        comp = self.comp_decoder(h)
        free = self.free_decoder(h)
        loss_dict = {}
        if labels is not None:
            if valid_mask is not None:
                xs, fs, cs, lbl = x[valid_mask], free[valid_mask], comp[valid_mask], labels[valid_mask]
            else:
                xs, fs, cs, lbl = x, free, comp, labels
            if lbl.numel() > 0:
                recon = self.recon_proj(torch.cat([fs, cs], dim=-1))
                loss_dict["recon"] = 0.5 * F.mse_loss(recon, xs)
                loss_dict["label"] = F.cross_entropy(self.classifier(cs), lbl)
                y_hat = F.softmax(self.classifier(fs), dim=-1)
                y_true = y_hat.gather(1, lbl.unsqueeze(1)).squeeze(1).clamp(0.05, 0.95)
                loss_dict["adversarial"] = (-1.0 / torch.log(y_true)).mean()
        return free, comp, loss_dict

    @staticmethod
    def total(loss_dict):
        return sum(loss_dict.values()) if loss_dict else torch.tensor(0.0)


class DisentanglingModule(nn.Module):
    def __init__(self, embed_dim, num_polarity_classes=3, dropout=0.1):
        super().__init__()
        self.polarity_dis = _Disentangler(embed_dim, num_polarity_classes, dropout)
        self.veracity_dis = _Disentangler(embed_dim, 2, dropout)

    def forward(self, e_j, polarity_labels=None, veracity_labels=None, valid_mask=None):
        f_j, _, pol_ld = self.polarity_dis(e_j, polarity_labels, valid_mask)
        e_clean, _, ver_ld = self.veracity_dis(f_j, veracity_labels, valid_mask)
        return e_clean, _Disentangler.total(pol_ld), _Disentangler.total(ver_ld)

    def predict_veracity(self, e_j):
        f_j, _, _ = self.polarity_dis(e_j)
        h = self.veracity_dis.encoder(f_j)
        v = self.veracity_dis.comp_decoder(h)
        return F.softmax(self.veracity_dis.classifier(v), dim=-1)[:, 1]


class InterestLearner(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers=1, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                           dim_feedforward=embed_dim * 4, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x, padding_mask=None):
        return self.transformer(x, src_key_padding_mask=padding_mask)


class InterestAggregation(nn.Module):
    def __init__(self, embed_dim, user_embed_dim=32, dropout=0.1, use_user_id=False):
        super().__init__()
        self.W1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W2 = nn.Linear(embed_dim, embed_dim, bias=False)
        fusion_in = embed_dim * 2 + (user_embed_dim if use_user_id else 0)
        self.fusion = nn.Linear(fusion_in, embed_dim)

    def forward(self, K_tilde, E_tilde, cand_keyword, cand_clean, mask=None, user_emb=None):
        qh = self.W1(cand_keyword).unsqueeze(1)
        ql = self.W2(cand_clean).unsqueeze(1)
        score_h = (K_tilde * qh).sum(dim=-1)
        score_l = (E_tilde * ql).sum(dim=-1)
        if mask is not None:
            score_h = score_h.masked_fill(mask, float("-inf"))
            score_l = score_l.masked_fill(mask, float("-inf"))
        alpha = torch.softmax(score_h, dim=1).unsqueeze(-1)
        beta = torch.softmax(score_l, dim=1).unsqueeze(-1)
        i_high = (alpha * K_tilde).sum(dim=1)
        i_low = (beta * E_tilde).sum(dim=1)
        parts = [i_high, i_low]
        if user_emb is not None:
            parts.append(user_emb)
        return self.fusion(torch.cat(parts, dim=-1))


class HDInt(nn.Module):
    """frozen 경로 전용 (content_in=1536, keyword_in=2304)."""

    def __init__(self, content_in_dim=1536, keyword_in_dim=2304, embed_dim=256,
                 num_heads=4, num_polarity_classes=3, num_transformer_layers=1,
                 dropout=0.1, num_users=0, user_embed_dim=32):
        super().__init__()
        self.content_proj = nn.Linear(content_in_dim, embed_dim)
        self.disentangler = DisentanglingModule(embed_dim, num_polarity_classes, dropout)
        self.keyword_proj = nn.Linear(keyword_in_dim, embed_dim)
        self.high_learner = InterestLearner(embed_dim, num_heads, num_transformer_layers, dropout)
        self.low_learner = InterestLearner(embed_dim, num_heads, num_transformer_layers, dropout)
        self.use_user_id = num_users > 0
        if self.use_user_id:
            self.user_embedding = nn.Embedding(num_users + 1, user_embed_dim, padding_idx=0)
        self.aggregation = InterestAggregation(embed_dim, user_embed_dim, dropout, self.use_user_id)

    def _proj_content(self, x):
        return torch.sigmoid(self.content_proj(x))

    def _score(self, u, cand_clean):
        return torch.sigmoid((u * cand_clean).sum(dim=-1))

    def forward(self, ctx_content, ctx_keyword, cand_content, cand_keyword,
                padding_mask=None, ctx_polarity=None, ctx_veracity=None,
                cand_polarity=None, cand_veracity=None, user_id=None):
        B, S = ctx_content.shape[0], ctx_content.shape[1]
        flat = self._proj_content(ctx_content.reshape(B * S, -1))
        fp = ctx_polarity.reshape(B * S) if ctx_polarity is not None else None
        fv = ctx_veracity.reshape(B * S) if ctx_veracity is not None else None
        fvalid = (~padding_mask).reshape(B * S) if padding_mask is not None else None
        ctx_clean_flat, pol_ctx, ver_ctx = self.disentangler(flat, fp, fv, fvalid)
        ctx_clean = ctx_clean_flat.reshape(B, S, -1)

        cand_proj = self._proj_content(cand_content)
        cand_clean, pol_cand, ver_cand = self.disentangler(cand_proj, cand_polarity, cand_veracity)

        ctx_kw = self.keyword_proj(ctx_keyword)
        cand_kw = self.keyword_proj(cand_keyword)
        K_tilde = self.high_learner(ctx_kw, padding_mask)
        E_tilde = self.low_learner(ctx_clean, padding_mask)
        user_emb = self.user_embedding(user_id) if (self.use_user_id and user_id is not None) else None
        u = self.aggregation(K_tilde, E_tilde, cand_kw, cand_clean, padding_mask, user_emb)
        return self._score(u, cand_clean), pol_ctx, pol_cand, ver_ctx, ver_cand

    @torch.no_grad()
    def encode_all_news(self, content_emb, keyword_emb, batch_size=2048):
        self.eval()
        dev = next(self.parameters()).device
        cleans, keywords = [], []
        for s in range(0, content_emb.size(0), batch_size):
            c = content_emb[s:s + batch_size].to(dev)
            k = keyword_emb[s:s + batch_size].to(dev)
            e_clean, _, _ = self.disentangler(self._proj_content(c))
            cleans.append(e_clean); keywords.append(self.keyword_proj(k))
        return {"clean": torch.cat(cleans, 0), "keyword": torch.cat(keywords, 0)}

    @torch.no_grad()
    def score_all(self, ctx_content, ctx_keyword, padding_mask, cache, user_id=None, chunk=4096):
        """단일 컨텍스트 → 전체 뉴스 점수 [N+1]. cache = encode_all_news 결과."""
        dev = next(self.parameters()).device
        N1 = cache["clean"].size(0)   # N+1
        ctx_clean, _, _ = self.disentangler(self._proj_content(ctx_content.to(dev)))
        ctx_kw = self.keyword_proj(ctx_keyword.to(dev))
        mask = padding_mask.to(dev)
        K1 = self.high_learner(ctx_kw.unsqueeze(0), mask.unsqueeze(0)).squeeze(0)
        E1 = self.low_learner(ctx_clean.unsqueeze(0), mask.unsqueeze(0)).squeeze(0)
        uemb = None
        if self.use_user_id and user_id is not None:
            uemb = self.user_embedding(torch.tensor([user_id], device=dev))
        scores = torch.full((N1,), float("-inf"), device=dev)
        for s in range(1, N1, chunk):
            e = min(s + chunk, N1); C = e - s
            cc = cache["clean"][s:e]; ck = cache["keyword"][s:e]
            Kt = K1.unsqueeze(0).expand(C, -1, -1)
            Et = E1.unsqueeze(0).expand(C, -1, -1)
            mk = mask.unsqueeze(0).expand(C, -1)
            ue = uemb.expand(C, -1) if uemb is not None else None
            u = self.aggregation(Kt, Et, ck, cc, mk, ue)
            scores[s:e] = self._score(u, cc)
        return scores


# ───────────────────── 통합 어댑터 ─────────────────────

class HDIntModel(BaseRecModel):
    name = "hdint"

    def __init__(self, shared, emb_dir, args, device):
        super().__init__(shared, emb_dir, args, device)
        with open(os.path.join(emb_dir, "emb.pkl"), "rb") as f:
            d = pickle.load(f)
        self.content_emb = d["content"].float()      # [N+1,1536]
        self.keyword_emb = d["keyword"].float()      # [N+1,2304]
        self.veracity = torch.tensor(shared["veracity"], dtype=torch.long)
        self.polarity = torch.tensor(shared["polarity"], dtype=torch.long)
        # 논문 5.2.3: negative는 2 real + 2 fake 균형 샘플링 → 진위별 풀 분리
        lab = np.array(shared["veracity"])
        rp = np.where(lab[1:] == 0)[0] + 1   # idx 1..N
        fp = np.where(lab[1:] == 1)[0] + 1
        full = np.arange(1, self.num_news + 1)
        self._real_pool = torch.from_numpy(rp if len(rp) else full).long()
        self._fake_pool = torch.from_numpy(fp if len(fp) else full).long()
        self.model = HDInt(
            content_in_dim=self.content_emb.size(1), keyword_in_dim=self.keyword_emb.size(1),
            embed_dim=args.get("embed_dim", 256), num_heads=args.get("num_heads", 4),
            num_transformer_layers=args.get("num_transformer_layers", 1),
            dropout=args.get("dropout", 0.1), num_users=self.num_users,
            user_embed_dim=args.get("user_embed_dim", 32),
        ).to(device)
        self._cache = None

    @staticmethod
    def _bce(pos, neg):
        return -torch.log(pos + 1e-8).mean() - torch.log(1 - neg + 1e-8).mean()

    def _loader(self, instances, shuffle):
        # 논문 5.2.1: "retain only those instances with a ground-truth true target
        # news piece" — 진짜 타깃(veracity==0) 인스턴스만 학습·검증.
        inst = [x for x in instances if self.veracity[x[2]].item() == 0]
        return DataLoader(SeqDataset(inst, MAX_LEN),
                          batch_size=self.args.get("batch_size", 64), shuffle=shuffle)

    def _sample_negs(self, tgt, K, dev):
        """논문 5.2.3: 2 real + 2 fake 균형 negative [B,K]. 타깃 충돌은 재샘플."""
        B = tgt.size(0); half = K // 2
        rp, fp = self._real_pool.to(dev), self._fake_pool.to(dev)

        def draw(pool, c):
            idx = pool[torch.randint(len(pool), (B, c), device=dev)]
            coll = idx == tgt.unsqueeze(1)          # 타깃(진짜)과 중복 방지
            for _ in range(3):
                if not coll.any():
                    break
                idx[coll] = pool[torch.randint(len(pool), (int(coll.sum().item()),), device=dev)]
                coll = idx == tgt.unsqueeze(1)
            return idx

        return torch.cat([draw(rp, half), draw(fp, K - half)], dim=1)

    def fit(self, train, val, log):
        dev = self.device
        ce, ke = self.content_emb.to(dev), self.keyword_emb.to(dev)
        ver, pol = self.veracity.to(dev), self.polarity.to(dev)
        lam = self.args.get("lambda", 10.0); gamma = self.args.get("gamma", 3.0)
        K = self.args.get("num_negatives", 4)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.args.get("lr", 3e-4))
        dl = self._loader(train, True); epochs = self.args.get("epochs", 15)
        N = self.num_news; best = -1.0; best_sd = None
        val_every = self.args.get("val_every", 3)
        for ep in range(epochs):
            self.model.train(); tot = 0.0
            for ids, mask, tgt, u in tqdm(dl, desc=f"[hdint] ep {ep+1}/{epochs}", leave=False):
                ids, mask, tgt, u = ids.to(dev), mask.to(dev), tgt.to(dev), u.to(dev)
                B = ids.size(0)
                ctx_c, ctx_k = ce[ids], ke[ids]
                cp, cv = pol[ids], ver[ids]
                pad = ~mask   # HDInt 규약: True=패딩 (SeqDataset mask는 True=유효)
                pos, pol_c, pol_cp, ver_c, ver_cp = self.model(
                    ctx_c, ctx_k, ce[tgt], ke[tgt], pad, cp, cv, pol[tgt], ver[tgt], u)
                neg_ids = self._sample_negs(tgt, K, dev)   # 2 real + 2 fake (논문 5.2.3)
                neg_scores = []; pol_n = ver_n = torch.tensor(0.0, device=dev)
                for k in range(K):
                    nid = neg_ids[:, k]
                    ns, _, pn, _, vn = self.model(
                        ctx_c, ctx_k, ce[nid], ke[nid], pad, cp, cv, pol[nid], ver[nid], u)
                    neg_scores.append(ns); pol_n = pol_n + pn; ver_n = ver_n + vn
                neg = torch.stack(neg_scores, 1)
                pol_t = pol_c + pol_cp + pol_n / K
                ver_t = ver_c + ver_cp + ver_n / K
                loss = lam * self._bce(pos, neg) + gamma * pol_t + ver_t
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); opt.step()
                tot += loss.item()
            if (ep + 1) % val_every == 0 or ep == epochs - 1:
                hr = self._val_hr(val)
                log(f"[hdint] epoch {ep+1}/{epochs}  loss {tot/len(dl):.3f}  val HR@10 {hr:.4f}")
                if hr > best:
                    best = hr; best_sd = copy.deepcopy(self.model.state_dict())
            else:
                log(f"[hdint] epoch {ep+1}/{epochs}  loss {tot/len(dl):.3f}")
        if best_sd is not None:
            self.model.load_state_dict(best_sd)
        self._cache = None

    @torch.no_grad()
    def _val_hr(self, val, k=10, sample=500):
        dev = self.device
        ce, ke = self.content_emb.to(dev), self.keyword_emb.to(dev)
        cache = self.model.encode_all_news(ce, ke)
        # 논문 정책: 진짜 타깃 인스턴스만으로 모델 선택 (평가기와 동일 기준)
        real_val = [x for x in val if self.veracity[x[2]].item() == 0]
        v = real_val[:sample]; hit = 0
        for u, ctx, tgt in v:
            from loader import pad_ctx
            ids, mask = pad_ctx(ctx, MAX_LEN)
            sc = self.model.score_all(ce[torch.tensor(ids)], ke[torch.tensor(ids)],
                                      ~torch.tensor(mask), cache, u)
            topk = sc.topk(k).indices.tolist()
            hit += int(tgt in topk)
        return hit / max(len(v), 1)

    @torch.no_grad()
    def rank_all(self, ctx_ids, ctx_mask, user):
        dev = self.device
        ce, ke = self.content_emb.to(dev), self.keyword_emb.to(dev)
        if self._cache is None:
            self._cache = self.model.encode_all_news(ce, ke)
        B = ctx_ids.size(0)
        out = torch.full((B, self.num_news + 1), float("-inf"), device=dev)
        for i in range(B):
            ids = ctx_ids[i]
            out[i] = self.model.score_all(ce[ids], ke[ids], ~ctx_mask[i], self._cache,
                                          int(user[i]))
        return out

    @torch.no_grad()
    def fake_prob(self):
        dev = self.device
        return self.model.disentangler.predict_veracity(
            self.model._proj_content(self.content_emb.to(dev)))
