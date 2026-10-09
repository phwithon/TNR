"""
1회용 enrich 단계: FakeNewsNet raw에서 news_url + text를 뽑아 원본 CSV에 컬럼으로
박아 넣는다. 이후 전체 파이프라인은 enriched CSV만 읽으므로 raw 폴더 상시 의존이 사라진다.

입력 : data/원본/{gossip,pol}.csv  +  FakeNewsNet/{gossipcop,politifact}/{real,fake}/<id>/news content.json
출력 : data/{gossip,pol}_enriched.csv  (원본 컬럼 + news_url + text, news_id 중복 제거)

필드 용도:
  - news_url : HDInt 극성 라벨 (도메인→좌/중/우, 논문: mediabiasfactcheck 출처 기반)
  - text     : PRISM 기사 본문 임베딩 (논문: frozen BERT로 기사 본문 인코딩)
               REC4MIT/HDInt는 title+description만 쓰므로 text 불필요.

주의: 이 단계는 FakeNewsNet raw 폴더가 있어야 실행된다. 현재 raw는 삭제됐고
enriched CSV(data/*_enriched.csv)만 보존돼 있으므로 재실행은 raw 복원 시에만 가능.
"""

import glob
import json
import os
import re

import pandas as pd


def _clean(s: str) -> str:
    """줄바꿈·탭·연속공백을 단일 공백으로 정규화 (CSV 한 행=한 줄 보장)."""
    if not s:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def _load_csv_urls(raw_dir: str, prefix: str) -> dict:
    """FakeNewsNet/csv/{prefix}_*.csv (컬럼 id,news_url)에서 id→news_url 맵.
    스크랩 이전 '원본 URL 목록'이라 raw json url보다 커버리지 높음 (gossip 100%)."""
    url_map = {}
    for path in glob.glob(os.path.join(raw_dir, "csv", f"{prefix}_*.csv")):
        try:
            c = pd.read_csv(path, usecols=["id", "news_url"])
        except Exception:
            continue
        for _, r in c.iterrows():
            u = r.get("news_url")
            if pd.notna(u) and str(u).strip():
                url_map[str(r["id"])] = str(u).strip()
    return url_map


def _read_raw(raw_dir: str, prefix: str, news_id: str):
    """news_id의 news content.json에서 (url, text) 반환. 없으면 ('','')."""
    for label in ("real", "fake"):
        path = os.path.join(raw_dir, prefix, label, news_id, "news content.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                return (d.get("url") or "").strip(), _clean(d.get("text") or "")
            except Exception:
                return "", ""
    return "", ""


def build_enriched(root: str, dataset: str, raw_prefix: str) -> dict:
    """원본 CSV + raw → enriched CSV 저장. 통계 반환.

    news_url : FakeNewsNet csv 원본 URL 우선, 없으면 raw json url fallback.
    text     : raw json 기사 본문 (PRISM용).
    """
    src = os.path.join(root, "data", "원본", f"{dataset}.csv")
    df = pd.read_csv(src)
    df["news_id"] = df["news_id"].astype(str)
    for col in ("title", "description"):
        if col in df.columns:
            df[col] = df[col].apply(lambda v: _clean(v) if pd.notna(v) else v)

    before = len(df)
    df = df.drop_duplicates(subset="news_id", keep="first").reset_index(drop=True)
    deduped = before - len(df)

    raw_dir = os.path.join(root, "FakeNewsNet")
    csv_urls = _load_csv_urls(raw_dir, raw_prefix)

    urls, texts = [], []
    n_url = n_text = n_from_csv = n_from_json = 0
    for nid in df["news_id"]:
        json_url, text = _read_raw(raw_dir, raw_prefix, nid)
        url = csv_urls.get(nid, "") or json_url   # csv 우선, json fallback
        if url:
            n_url += 1
            n_from_csv += 1 if csv_urls.get(nid) else 0
            n_from_json += 0 if csv_urls.get(nid) else 1
        urls.append(url)
        texts.append(text)
        n_text += bool(text)
    df["news_url"] = urls
    df["text"] = texts

    out = os.path.join(root, "data", f"{dataset}_enriched.csv")
    df.to_csv(out, index=False)
    return {
        "dataset": dataset, "rows": len(df), "deduped": deduped,
        "with_url": n_url, "url_from_csv": n_from_csv, "url_from_json": n_from_json,
        "with_text": n_text, "out": out,
    }
