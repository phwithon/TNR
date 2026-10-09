"""
통합 실행 진입점 — 세 모델을 동일 데이터/분할/평가로 비교.

사용:
  python main.py --model rec4mit --dataset pol
  python main.py --model hdint   --dataset gossip
  python main.py --model prism   --dataset pol
  python main.py --model all     --dataset pol        # 세 모델 순차 비교

모델 내부(이벤트메모리/2단분리기/diffusion)는 각 재구현 그대로,
데이터·분할·후보 라이브러리·평가 지표는 완전 공유.

로그: logs/{model}/{dataset}/{model}_{timestamp}.log 로 모델별·데이터별 분리 저장
      (ppp 양식: '%(asctime)s  %(message)s'). 콘솔에도 동시 출력.
      --model all 이면 비교표는 logs/comparison/{dataset}/comparison_{timestamp}.log 에도 저장.
"""

import argparse
import json
import logging
import os
import sys
import time

import torch

from loader import load_shared
from evaluate import evaluate, format_metrics
from models.rec4mit import Rec4MitModel, MAX_LEN as L_REC
from models.hdint import HDIntModel, MAX_LEN as L_HD
from models.prism import PRISMModel, MAX_LEN as L_PR

ROOT = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(ROOT, "processed")
LOGDIR = os.path.join(ROOT, "logs")
MODELDIR = os.path.join(ROOT, "model_dir")

REGISTRY = {
    "rec4mit": (Rec4MitModel, L_REC),
    "hdint":   (HDIntModel, L_HD),
    "prism":   (PRISMModel, L_PR),
}


def load_config(dataset):
    path = os.path.join(ROOT, f"config_{dataset}.json")
    with open(path) as f:
        return json.load(f)


def make_logger(model, dataset, ts):
    """모델별·데이터별 로그 파일 + 콘솔 핸들러. (file, log_fn) 반환.
    파일: '%(asctime)s  %(message)s' (ppp 양식), 콘솔: 메시지만."""
    d = os.path.join(LOGDIR, model, dataset)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{model}_{ts}.log")
    logger = logging.getLogger(f"{dataset}.{model}.{ts}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fh = logging.FileHandler(path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)

    def log(msg=""):
        for line in str(msg).split("\n"):   # 멀티라인도 줄마다 타임스탬프
            logger.info(line)

    return path, log


def run_one(model_name, dataset, device, max_eval, ts, epochs_override=None):
    shared = load_shared(PROC, dataset)
    cls, max_len = REGISTRY[model_name]
    args = load_config(dataset)[model_name]
    if epochs_override:
        args["epochs"] = epochs_override
    emb_dir = os.path.join(PROC, dataset, model_name)
    path, log = make_logger(model_name, dataset, ts)
    log(f"########## {model_name.upper()} on {dataset} ##########")
    log(f"device: {device}  max_eval: {max_eval}")
    log(f"config: {args}")
    model = cls(shared, emb_dir, args, device)
    model.fit(shared["splits"]["train"], shared["splits"]["val"], log)
    ckpt = os.path.join(MODELDIR, model_name, dataset, f"{model_name}_{ts}.pth")
    model.save(ckpt)
    log(f"model saved -> {ckpt}")
    metrics = evaluate(model, shared["splits"]["test"], shared, max_len, device,
                       max_eval=max_eval, log=log)
    log(format_metrics(metrics))
    log(f"log saved -> {path}")
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["rec4mit", "hdint", "prism", "all"])
    p.add_argument("--dataset", required=True, choices=["gossip", "pol"])
    p.add_argument("--max_eval", type=int, default=3000,
                   help="평가 인스턴스 상한 (전체 라이브러리 랭킹은 비싸서 샘플링)")
    p.add_argument("--epochs", type=int, default=None, help="epochs 오버라이드")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    print(f"Device: {device}  |  run {ts}")

    models = ["rec4mit", "hdint", "prism"] if args.model == "all" else [args.model]
    results = {}
    for m in models:
        results[m] = run_one(m, args.dataset, device, args.max_eval, ts, args.epochs)

    if len(models) > 1:
        cols = ("HR", "NDCG", "RT", "FNSR", "F1")
        lines = [f"\n########### COMPARISON ({args.dataset}) - PRISM metrics @10 ###########"]
        lines.append("  {:<10}".format("Model") + "".join("{:>10}".format(f"{m}@10") for m in cols)
                     + "{:>10}".format("1-R"))
        for m in models:
            r = results[m]
            lines.append("  {:<10}".format(m)
                         + "".join("{:>10.4f}".format(r[f"{mm}@10"]) for mm in cols)
                         + "{:>10.4f}".format(r["1-R"]))
        text = "\n".join(lines)
        print(text)
        d = os.path.join(LOGDIR, "comparison", args.dataset)
        os.makedirs(d, exist_ok=True)
        cpath = os.path.join(d, f"comparison_{ts}.log")
        with open(cpath, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"\ncomparison saved -> {cpath}")


if __name__ == "__main__":
    main()
