"""
통합 평가기 — 전체 라이브러리 랭킹 + 공통 지표.

세 모델 모두 rank_all(ctx_ids, ctx_mask, user) → [B, N+1] 로 전체 뉴스를 스코어링한다.
동일 테스트 인스턴스·동일 후보 라이브러리·동일 지표로 공정 비교.

지표 (PRISM [SIGIR'25] §5.3 기준):
  추천 정확도 : HR@k, NDCG@k   (타깃 = 유저의 실제 다음 뉴스)
  가짜 억제   : RT@k   (상위 k 중 진짜 비율, Real News Ratio)
               FNSR@k (가중 가짜뉴스 억제율; w(i)=k-i+1, 상위일수록 큰 패널티)
               1-R    (Real News Priority; top-1이 진짜인 비율)
  통합        : F1@k  = 2·HR@k·FNSR@k / (HR@k + FNSR@k)

정책:
  - 타깃이 진짜(veracity==0)인 인스턴스만 평가 (REC4MIT 논문: 진짜만 추천 대상).
  - idx0(패딩)은 후보에서 제외 (rank_all이 -inf 처리).
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from loader import SeqDataset


def _collate(batch):
    ids, mask, tgt, u = zip(*batch)
    return torch.stack(ids), torch.stack(mask), torch.stack(tgt), torch.stack(u)


@torch.no_grad()
def evaluate(model, instances, shared, max_len, device,
             ks=(5, 10, 20), batch_size=128, max_eval=None, log=print):
    veracity = torch.tensor(shared["veracity"], dtype=torch.long)  # [N+1] 0=real,1=fake
    # 진짜 타깃 인스턴스만
    inst = [x for x in instances if veracity[x[2]].item() == 0]
    if max_eval and len(inst) > max_eval:
        rng = np.random.default_rng(42)
        inst = [inst[i] for i in rng.choice(len(inst), max_eval, replace=False)]
    log(f"  [eval] {len(inst)} real-target instances (ks={ks})")

    dl = DataLoader(SeqDataset(inst, max_len), batch_size=batch_size, collate_fn=_collate)
    maxk = max(ks)
    N = shared["meta"]["num_news"]
    ver = veracity.to(device)

    agg = {f"{m}@{k}": 0.0 for k in ks for m in ("HR", "NDCG", "RT", "FNSR")}
    real_first = 0
    n = 0

    for ids, mask, tgt, u in dl:
        ids, mask, u = ids.to(device), mask.to(device), u.to(device)
        tgt = tgt.to(device)
        scores = model.rank_all(ids, mask, u)          # [B, N+1]
        B = scores.size(0)
        top = scores.topk(maxk, dim=1).indices          # [B, maxk]  뉴스 idx

        top_ver = ver[top]                               # [B, maxk] 0/1
        # top-1 real
        real_first += (top_ver[:, 0] == 0).sum().item()

        # 타깃 순위 (top 내 위치, 없으면 큰 값)
        match = (top == tgt.unsqueeze(1))                # [B, maxk]
        has = match.any(dim=1)
        pos = torch.where(has, match.float().argmax(dim=1), torch.full_like(tgt, maxk))

        for k in ks:
            ink = has & (pos < k)                        # 타깃이 top-k 안
            rank = pos.float() + 1.0
            agg[f"HR@{k}"] += ink.sum().item()
            agg[f"NDCG@{k}"] += (ink.float() / torch.log2(rank + 1.0)).sum().item()
            # 가짜 억제
            tk = top_ver[:, :k]                          # [B,k]
            agg[f"RT@{k}"] += ((tk == 0).float().mean(dim=1)).sum().item()
            w = torch.arange(k, 0, -1, device=device).float()   # k, k-1, ..., 1
            fnsr = 1.0 - (w * (tk == 1).float()).sum(dim=1) / w.sum()
            agg[f"FNSR@{k}"] += fnsr.sum().item()
        n += B

    metrics = {m: v / max(n, 1) for m, v in agg.items()}
    # F1@k = 2·HR·FNSR/(HR+FNSR)  (PRISM 통합 지표, 집계값 기준)
    for k in ks:
        hr, fn = metrics[f"HR@{k}"], metrics[f"FNSR@{k}"]
        metrics[f"F1@{k}"] = (2 * hr * fn / (hr + fn)) if (hr + fn) > 0 else 0.0
    metrics["1-R"] = real_first / max(n, 1)
    metrics["n_eval"] = n
    return metrics


def format_metrics(metrics, ks=(5, 10, 20)) -> str:
    lines = ["=" * 60, f"  Evaluation ({metrics['n_eval']} instances) - PRISM metrics", "-" * 60]
    hdr = "  {:<8}".format("Metric") + "".join("{:>12}".format(f"@{k}") for k in ks)
    lines.append(hdr)
    for m in ("HR", "NDCG", "RT", "FNSR", "F1"):
        row = "  {:<8}".format(m) + "".join("{:>12.4f}".format(metrics[f"{m}@{k}"]) for k in ks)
        lines.append(row)
    lines.append("-" * 60)
    lines.append("  1-R (Real News Priority, top-1 real ratio) = {:.4f}".format(metrics["1-R"]))
    lines.append("=" * 60)
    return "\n".join(lines)


def _latest_ckpt(model_dir, model, dataset):
    """model_dir/{model}/{dataset}/*.pth 중 가장 최근(mtime) 파일 경로. 없으면 None."""
    import glob
    d = os.path.join(model_dir, model, dataset)
    files = glob.glob(os.path.join(d, "*.pth"))
    return max(files, key=os.path.getmtime) if files else None


def _eval_ckpt(model_name, dataset, device, shared, max_eval, ckpt=None):
    """저장된 체크포인트를 불러와 평가만 수행 → metrics 반환 (학습 X)."""
    from main import REGISTRY, load_config, PROC, MODELDIR
    ckpt = ckpt or _latest_ckpt(MODELDIR, model_name, dataset)
    if not ckpt or not os.path.exists(ckpt):
        raise SystemExit(f"체크포인트를 찾을 수 없음: "
                         f"{ckpt or f'model_dir/{model_name}/{dataset}/*.pth 없음'}")
    cls, max_len = REGISTRY[model_name]
    args = load_config(dataset)[model_name]
    emb_dir = os.path.join(PROC, dataset, model_name)
    print(f"[{model_name}] loading {ckpt}")
    model = cls(shared, emb_dir, args, device)
    model.load(ckpt)   # 학습 생략, 저장된 최적 가중치 복원
    return evaluate(model, shared["splits"]["test"], shared, max_len, device,
                    max_eval=max_eval)


def _write_comparison(results, dataset, models):
    """main.py와 동일 양식의 비교표 생성 → logs/comparison/{dataset}/ 에 저장."""
    import time
    from main import LOGDIR
    cols = ("HR", "NDCG", "RT", "FNSR", "F1")
    lines = [f"\n########### COMPARISON ({dataset}) - PRISM metrics @10 ###########"]
    lines.append("  {:<10}".format("Model") + "".join("{:>10}".format(f"{m}@10") for m in cols)
                 + "{:>10}".format("1-R"))
    for m in models:
        r = results[m]
        lines.append("  {:<10}".format(m)
                     + "".join("{:>10.4f}".format(r[f"{mm}@10"]) for mm in cols)
                     + "{:>10.4f}".format(r["1-R"]))
    text = "\n".join(lines)
    print(text)
    ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    d = os.path.join(LOGDIR, "comparison", dataset)
    os.makedirs(d, exist_ok=True)
    cpath = os.path.join(d, f"comparison_{ts}.log")
    with open(cpath, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\ncomparison saved -> {cpath}")


def main():
    """학습 없이 저장된 체크포인트를 불러와 평가만 수행.

    사용:
      python evaluate.py --model rec4mit --dataset pol                  # 최신 체크포인트 자동 선택
      python evaluate.py --model prism   --dataset pol --ckpt <경로>    # 특정 체크포인트 지정
      python evaluate.py --model all     --dataset pol                  # 3모델 최신 체크포인트 → 비교표만 재생성
    """
    import argparse
    from main import PROC
    from loader import load_shared

    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["rec4mit", "hdint", "prism", "all"])
    p.add_argument("--dataset", required=True, choices=["gossip", "pol"])
    p.add_argument("--ckpt", default=None, help="체크포인트 경로 (미지정 시 최신 .pth 자동; --model all에선 무시)")
    p.add_argument("--max_eval", type=int, default=3000)
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shared = load_shared(PROC, a.dataset)
    print(f"Device: {device}  |  dataset {a.dataset}")

    if a.model == "all":
        models = ["rec4mit", "hdint", "prism"]
        results = {m: _eval_ckpt(m, a.dataset, device, shared, a.max_eval) for m in models}
        for m in models:
            print(f"\n### {m} ###"); print(format_metrics(results[m]))
        _write_comparison(results, a.dataset, models)
    else:
        metrics = _eval_ckpt(a.model, a.dataset, device, shared, a.max_eval, a.ckpt)
        print(format_metrics(metrics))


if __name__ == "__main__":
    main()
