"""
뉴스 임베딩 — 2단계: (1) 전체 뉴스 임베딩 캐싱, (2) P_u/P_c로 조립.

핵심: 뉴스별 임베딩은 cls_ratio와 무관(기사 텍스트만 의존). cls_ratio는 P_c/P_u
배정만 바꾼다. 따라서 **전체 뉴스 임베딩을 1회 캐싱**(step=cache)해두고,
cls_ratio를 바꿀 때마다 캐시에서 **news_id로 조회해 조립**(step=rec4mit/hdint/prism)만 하면
BERT/KeyBERT를 재실행하지 않아 튜닝 루프가 빠르다.

캐시 (processed/{ds}/_cache/, 전체 뉴스·pad 없음·df 순서):
  content.npy  [T,1536]  ST(bert-base) title768 ⊕ desc768       (rec4mit+hdint 공용)
  keyword.npy  [T,2304]  KeyBERT 3키워드 × 768                  (hdint)
  cls.npy      [T,768]   BERT CLS (title+본문, max_len 512)       (prism)
  news_ids.json          행 → news_id 매핑
  keyword_strings.json   news_id → 추출 키워드 문자열 (검증용)

조립 산출 (processed/{ds}/{model}/):
  rec4mit/emb.npz   : {"emb": [M+1,1536]}            (M=|P_u|)
  hdint/emb.pkl     : {"content":[M+1,1536], "keyword":[M+1,2304]}
  prism/emb.pkl     : {"embedding":[M+1,768]}        (추천 라이브러리 = P_u)
  prism/cls_data.pkl: [{"embedding":[768],"label":0/1}]  (분류기셋 = P_c, disjoint)

인코더 규약 (§4.1.1): REC4MIT/HDInt = SentenceTransformer("bert-base-uncased") mean-pool,
title 32 / desc 128 / keyword 16, normalize; PRISM = transformers BertModel CLS, 512.
"""

import json
import os
import pickle

import numpy as np
import torch

from .common import load_news

LEN_TITLE = 32
LEN_DESC = 128
LEN_KW = 16
LEN_TEXT = 512
BERT_NAME = "bert-base-uncased"
NUM_KW = 3


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _all_fields(df, col):
    """df 행 순서로 col 텍스트 리스트 (결측 '')."""
    out = []
    for _, row in df.iterrows():
        v = row.get(col)
        out.append("" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v).strip())
    return out


# ───────────────── per-news 인코더 (pad 없음, 전체 뉴스) ─────────────────

def _st_content_rows(model, titles, descs):
    """title768 ⊕ desc768 = [T,1536]. desc 거의 없음(<=5자) → 0벡터."""
    def embed(texts, max_len):
        model.max_seq_length = max_len
        return model.encode(texts, batch_size=128, normalize_embeddings=True,
                            convert_to_numpy=True).astype(np.float32)
    T = len(titles)
    TT = embed([t or "unknown news" for t in titles], LEN_TITLE)   # [T,768]
    D = np.zeros((T, 768), dtype=np.float32)
    idx_desc = [i for i, d in enumerate(descs) if len(d) > 5]
    if idx_desc:
        D[idx_desc] = embed([descs[i] for i in idx_desc], LEN_DESC)
    return np.concatenate([TT, D], axis=1)                         # [T,1536]


def _keyword_rows(model, titles, descs):
    """KeyBERT(bert-base) top-3 (unigram) → 각 768 mean-pool ⊕ = [T,2304].
    반환: (keyword[T,2304], kw_strings[T][≤3])  — 문자열은 검증용."""
    from keybert import KeyBERT
    T = len(titles)
    kw_model = KeyBERT(model=model)
    docs = [((titles[i] or "") + ". " + (descs[i] or "")).strip() for i in range(T)]
    kws = kw_model.extract_keywords(docs, keyphrase_ngram_range=(1, 1),
                                    stop_words="english", top_n=NUM_KW)
    kw_strings, flat, pos = [], [], []
    for i in range(T):
        ks = kws[i] if isinstance(kws[i], list) else []
        words = [pair[0] for pair in ks[:NUM_KW]]
        kw_strings.append(words)
        for j, w in enumerate(words):
            flat.append(w); pos.append((i, j))
    keyword = np.zeros((T, NUM_KW * 768), dtype=np.float32)
    if flat:
        model.max_seq_length = LEN_KW
        E = model.encode(flat, batch_size=128, normalize_embeddings=True,
                         convert_to_numpy=True).astype(np.float32)
        for (i, j), v in zip(pos, E):
            keyword[i, j * 768:(j + 1) * 768] = v
    return keyword, kw_strings


def _bert_cls(texts, tokenizer, model, max_len, device, batch=64):
    """텍스트 리스트 → [len,768] CLS 임베딩. 빈 문자열은 영벡터."""
    out = np.zeros((len(texts), 768), dtype=np.float32)
    nonempty = [i for i, t in enumerate(texts) if t]
    for s in range(0, len(nonempty), batch):
        idxs = nonempty[s:s + batch]
        enc = tokenizer([texts[i] for i in idxs], padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt").to(device)
        with torch.no_grad():
            cls = model(**enc).last_hidden_state[:, 0, :].cpu().float().numpy()
        out[idxs] = cls
    return out


# ───────────────── (1) 전체 뉴스 임베딩 캐싱 (cls_ratio 무관) ─────────────────

def build_cache(root, dataset, proc_dir):
    """전체 뉴스 임베딩을 1회 계산해 캐싱. cls_ratio와 무관하므로 재사용."""
    from sentence_transformers import SentenceTransformer
    from transformers import BertModel, BertTokenizer
    df = load_news(root, dataset)
    ids = list(df["news_id"])
    titles = _all_fields(df, "title")
    descs = _all_fields(df, "description")
    texts = _all_fields(df, "text")
    cache_dir = os.path.join(proc_dir, dataset, "_cache")
    os.makedirs(cache_dir, exist_ok=True)
    dev = _device()

    st = SentenceTransformer(BERT_NAME, device=dev)
    content = _st_content_rows(st, titles, descs)        # [T,1536]
    keyword, kw_strings = _keyword_rows(st, titles, descs)   # [T,2304] + 문자열

    tok = BertTokenizer.from_pretrained(BERT_NAME)
    bert = BertModel.from_pretrained(BERT_NAME).to(dev).eval()
    cls_inputs = [(titles[i] + " " + texts[i]).strip() or "unknown news" for i in range(len(ids))]
    cls = _bert_cls(cls_inputs, tok, bert, LEN_TEXT, dev, batch=32)   # [T,768]

    np.save(os.path.join(cache_dir, "content.npy"), content)
    np.save(os.path.join(cache_dir, "keyword.npy"), keyword)
    np.save(os.path.join(cache_dir, "cls.npy"), cls)
    json.dump(ids, open(os.path.join(cache_dir, "news_ids.json"), "w", encoding="utf-8"))
    # 검증용: news_id → 추출 키워드 문자열 (임베딩이 어떤 단어인지 눈으로 확인)
    json.dump({ids[i]: kw_strings[i] for i in range(len(ids))},
              open(os.path.join(cache_dir, "keyword_strings.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    return {"news": len(ids), "content": list(content.shape),
            "keyword": list(keyword.shape), "cls": list(cls.shape),
            "with_text": sum(1 for t in texts if t)}


# ───────────────── (2) 캐시 → P_u/P_c 조립 (cls_ratio마다, 빠름) ─────────────────

def _load_cache(proc_dir, dataset):
    cd = os.path.join(proc_dir, dataset, "_cache")
    ids = json.load(open(os.path.join(cd, "news_ids.json"), encoding="utf-8"))
    return cd, {nid: i for i, nid in enumerate(ids)}


def _pu_table(arr, id2row, news2idx, dim):
    """캐시 배열 → P_u idx 1..M 테이블 [M+1,dim] (row0=pad)."""
    out = np.zeros((len(news2idx) + 1, dim), dtype=np.float32)
    for nid, idx in news2idx.items():
        out[idx] = arr[id2row[nid]]
    return out


def gen_rec4mit(proc_dir, dataset, news2idx, out_dir):
    cd, id2row = _load_cache(proc_dir, dataset)
    content = np.load(os.path.join(cd, "content.npy"))
    table = _pu_table(content, id2row, news2idx, 1536)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "emb.npz"), emb=table)
    return {"shape": list(table.shape)}


def gen_hdint(proc_dir, dataset, news2idx, out_dir):
    cd, id2row = _load_cache(proc_dir, dataset)
    content = np.load(os.path.join(cd, "content.npy"))
    keyword = np.load(os.path.join(cd, "keyword.npy"))
    ctable = _pu_table(content, id2row, news2idx, 1536)
    ktable = _pu_table(keyword, id2row, news2idx, 2304)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "emb.pkl"), "wb") as f:
        pickle.dump({"content": torch.from_numpy(ctable),
                     "keyword": torch.from_numpy(ktable)}, f)
    return {"content": list(ctable.shape), "keyword": list(ktable.shape)}


def gen_prism(proc_dir, dataset, news2idx, pc_news, out_dir):
    """추천 라이브러리(P_u) 임베딩 + 분류기셋(P_c, disjoint) cls_data. 논문 §3: P_c∩P_u=∅."""
    cd, id2row = _load_cache(proc_dir, dataset)
    cls = np.load(os.path.join(cd, "cls.npy"))
    table = _pu_table(cls, id2row, news2idx, 768)          # P_u 라이브러리
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "emb.pkl"), "wb") as f:
        pickle.dump({"embedding": torch.from_numpy(table)}, f)
    # cls_data = P_c (disjoint)
    cls_data = [{"embedding": torch.from_numpy(cls[id2row[nid]].copy()), "label": int(lab)}
                for nid, lab in pc_news.items()]
    with open(os.path.join(out_dir, "cls_data.pkl"), "wb") as f:
        pickle.dump(cls_data, f)
    return {"lib_shape": list(table.shape), "cls_size": len(cls_data),
            "cls_from": "P_c (disjoint)"}


# ───────────────── 디스패치 ─────────────────

def build_embeddings(root, dataset, model, proc_dir):
    """model ∈ {rec4mit, hdint, prism}. 캐시 + shared(P_u 인덱스)로 조립.
    선행 조건: step=cache (전체 임베딩) + step=shared (P_c/P_u 분할)."""
    shared_dir = os.path.join(proc_dir, dataset, "shared")
    cache_ids = os.path.join(proc_dir, dataset, "_cache", "news_ids.json")
    if not os.path.exists(cache_ids):
        raise SystemExit(f"캐시 없음 → 먼저 실행: "
                         f"python -m preprocess.prepare --step cache --dataset {dataset}")
    news2idx = json.load(open(os.path.join(shared_dir, "news_index.json"), encoding="utf-8"))
    out_dir = os.path.join(proc_dir, dataset, model)
    if model == "rec4mit":
        return gen_rec4mit(proc_dir, dataset, news2idx, out_dir)
    if model == "hdint":
        return gen_hdint(proc_dir, dataset, news2idx, out_dir)
    if model == "prism":
        pc_news = json.load(open(os.path.join(shared_dir, "pc_news.json"), encoding="utf-8"))
        return gen_prism(proc_dir, dataset, news2idx, pc_news, out_dir)
    raise ValueError(model)
