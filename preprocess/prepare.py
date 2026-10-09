"""
전처리 진입점 — 공통 산출물 + 모델별 임베딩 생성.

사용:
  python -m preprocess.prepare --step enrich  --dataset pol gossip              # raw → enriched CSV (FakeNewsNet 필요)
  python -m preprocess.prepare --step cache   --dataset pol gossip              # 전체 뉴스 임베딩 1회 캐싱 (cls_ratio 무관)
  python -m preprocess.prepare --step shared  --dataset pol gossip --cls_ratio 0.4  # temporal P_c/P_u 분할
  python -m preprocess.prepare --step rec4mit --dataset pol gossip              # 캐시→P_u 조립: rec4mit/emb.npz
  python -m preprocess.prepare --step hdint   --dataset pol gossip              # 캐시→P_u 조립: hdint/emb.pkl
  python -m preprocess.prepare --step prism   --dataset pol gossip              # 캐시→P_u+P_c 조립: prism/*.pkl

파이프라인 순서: enrich → cache → shared → {rec4mit, hdint, prism}.
cache는 cls_ratio와 무관하므로 1회만. cls_ratio 튜닝 시 shared부터 재실행하면
캐시에서 조립만 하므로 빠르다 (BERT/KeyBERT 재실행 없음).
(enrich는 FakeNewsNet raw가 삭제돼 재실행 불가 — enriched CSV만 보존됨.)
"""

import argparse
import os

from .common import build_shared, DATASETS
from .embed import build_embeddings, build_cache
from .enrich import build_enriched

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # TNR/
PROC = os.path.join(ROOT, "processed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--step", required=True,
                   choices=["enrich", "cache", "shared", "rec4mit", "hdint", "prism"])
    p.add_argument("--dataset", nargs="+", default=["gossip", "pol"],
                   choices=["gossip", "pol"])
    p.add_argument("--cls_ratio", type=float, default=None,
                   help="temporal P_c/P_u 분할 비율 (shared 단계). 미지정 시 common.CLS_RATIO(0.40)")
    a = p.parse_args()

    for ds in a.dataset:
        print(f"\n===== [{a.step}] {ds} =====")
        if a.step == "enrich":
            st = build_enriched(ROOT, ds, DATASETS[ds]["raw_prefix"])
            print(f"  rows={st['rows']} (deduped {st['deduped']})  "
                  f"with_url={st['with_url']} (csv {st['url_from_csv']} + json {st['url_from_json']})  "
                  f"with_text={st['with_text']}")
            print(f"  -> {st['out']}")
        elif a.step == "cache":
            st = build_cache(ROOT, ds, PROC)
            print(f"  전체뉴스={st['news']}  content={st['content']}  "
                  f"keyword={st['keyword']}  cls={st['cls']}  with_text={st['with_text']}")
            print(f"  -> {os.path.join(PROC, ds, '_cache')}")
        elif a.step == "shared":
            kw = {} if a.cls_ratio is None else {"cls_ratio": a.cls_ratio}
            meta = build_shared(ROOT, ds, os.path.join(PROC, ds, "shared"), **kw)
            print(f"  news(P_u)={meta['num_news']}  P_c={meta['pc_size']}  "
                  f"cls_ratio={meta['cls_ratio']}  users={meta['num_users']}  "
                  f"instances={meta['num_instances']}")
            print(f"  veracity={meta['veracity_dist']}  polarity={meta['polarity_stats']}")
            print(f"  split={meta['split_sizes']}")
        else:
            st = build_embeddings(ROOT, ds, a.step, PROC)
            print(f"  {st}")
            print(f"  -> {os.path.join(PROC, ds, a.step)}")


if __name__ == "__main__":
    main()
