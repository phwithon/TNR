"""
극성(정치 성향) 라벨 복원 모듈.

공통 data/*.csv 에는 URL 컬럼이 없으므로, FakeNewsNet raw(news content.json)의
url 필드에서 도메인을 추출해 좌(0)/중립(1)/우(2) 라벨을 만든다.
도메인 테이블은 HDInt 원본(build_gossipcop_fnn)의 DOMAIN_POLARITY를 그대로 재사용.

URL이 없거나(=기사 json에 url 누락) 테이블에 없는 도메인은 중립(1) 기본값.
이는 HDInt 원본 lookup_polarity(default=1)와 동일한 동작.

실사용: enriched CSV의 news_url 컬럼 → extract_domain → lookup_polarity (common.build_shared).
build_polarity()는 FakeNewsNet raw에서 직접 뽑는 경로(enrich 단계 대안)로, 지금 파이프라인은
enriched CSV에 이미 news_url이 구워져 있어 사용하지 않는다.
"""

import json
import os
import re
from urllib.parse import urlparse

# ─── HDInt 원본 도메인→극성 테이블 (build_gossipcop_fnn(finetune).py) ───
DOMAIN_POLARITY = {
    # left (0)
    "cnn.com": 0, "abcnews.go.com": 0, "nytimes.com": 0,
    "washingtonpost.com": 0, "nbcnews.com": 0, "msnbc.com": 0,
    "cbsnews.com": 0, "politico.com": 0, "huffpost.com": 0,
    "huffingtonpost.com": 0, "theguardian.com": 0, "slate.com": 0,
    "vox.com": 0, "motherjones.com": 0, "thenation.com": 0,
    "thedailybeast.com": 0, "buzzfeed.com": 0, "vice.com": 0,
    "newsweek.com": 0, "rawstory.com": 0, "salon.com": 0,
    "people.com": 0, "eonline.com": 0, "vanityfair.com": 0,
    "rollingstone.com": 0, "mtv.com": 0,
    "elle.com": 0, "cosmopolitan.com": 0, "harpersbazaar.com": 0,
    "thecut.com": 0, "refinery29.com": 0, "glamour.com": 0,
    "bustle.com": 0, "vulture.com": 0, "thewrap.com": 0,
    "today.com": 0, "mercurynews.com": 0, "metro.co.uk": 0,
    "businessinsider.com": 0, "hollywoodreporter.com": 0, "independent.co.uk": 0,
    "nydailynews.com": 0, "elitedaily.com": 0, "townandcountrymag.com": 0,
    "inquisitr.com": 0, "toofab.com": 0, "teenvogue.com": 0,
    "latimes.com": 0, "vogue.com": 0, "allure.com": 0,
    "mcclatchydc.com": 0, "newslo.com": 0, "dailyworldupdate.us": 0,
    "states-tv.com": 0, "inverse.com": 0, "taxpolicycenter.org": 0,
    "kff.org": 0, "barackobama.com": 0, "hillaryhub.com": 0,
    "moveon.org": 0, "actionnetwork.org": 0,
    "clinton.senate.gov": 0, "teddeutch.house.gov": 0,
    "pbs.org": 0, "tampabay.com": 0, "boingboing.net": 0,
    "haaretz.com": 0, "aljazeera.com": 0, "themoscowtimes.com": 0,
    "thelastlineofdefense.org": 0, "mirror.co.uk": 0,
    # neutral (1)
    "reuters.com": 1, "apnews.com": 1, "bbc.com": 1, "bbc.co.uk": 1,
    "usatoday.com": 1, "thehill.com": 1, "time.com": 1,
    "bloomberg.com": 1, "forbes.com": 1, "wsj.com": 1,
    "npr.org": 1, "factcheck.org": 1, "snopes.com": 1, "politifact.com": 1,
    "usmagazine.com": 1, "ew.com": 1,
    "variety.com": 1, "deadline.com": 1, "tmz.com": 1, "pagesix.com": 1,
    "justjared.com": 1, "aceshowbiz.com": 1, "contactmusic.com": 1,
    "digitalspy.com": 1,
    # right (2)
    "foxnews.com": 2, "breitbart.com": 2, "dailycaller.com": 2,
    "washingtontimes.com": 2, "nypost.com": 2, "townhall.com": 2,
    "dailywire.com": 2, "theblaze.com": 2, "newsmax.com": 2,
    "nationalreview.com": 2, "dailymail.co.uk": 2, "express.co.uk": 2,
    "thegatewaypundit.com": 2, "babylonbee.com": 2, "thefederalistpapers.org": 2,
    "rare.us": 2, "madworldnews.com": 2, "lidblog.com": 2,
    "usaconservativereport.com": 2, "federalisttribune.com": 2,
    "nyeveningnews.com": 2, "dailyusaupdate.com": 2, "conservativepost.com": 2,
    "southernconservativextra.com": 2, "newsuptoday.com": 2, "usadailypost.us": 2,
    "thetrumpmedia.com": 2, "maga2020.us": 2, "therightists.com": 2,
    "conservative-headlines.com": 2, "patriotnetwork.one": 2,
    "freedomcrossroads.us": 2, "fedupthebook.com": 2, "defense-usa.xyz": 2,
    "republicanleader.house.gov": 2, "sasse.senate.gov": 2,
    "telegraph.co.uk": 2, "thesun.co.uk": 2,
}

NEUTRAL = 1  # 기본값 (URL 없음/테이블 미등재)


def extract_domain(url: str) -> str:
    """URL → 정규화된 도메인 (www. 제거, 포트 제거, web.archive.org 래핑 해제)."""
    if not url:
        return ""
    if "web.archive.org/web/" in url:
        m = re.search(r"web\.archive\.org/web/\d+[^/]*/(.+)", url)
        if m:
            url = m.group(1)
            if not url.startswith("http"):
                url = "http://" + url
    try:
        netloc = urlparse(url).netloc.lower()
        netloc = re.sub(r":\d+$", "", netloc)
        return re.sub(r"^www\.", "", netloc)
    except Exception:
        return ""


def lookup_polarity(domain: str, default: int = NEUTRAL) -> int:
    """도메인을 테이블과 대조 (서브도메인은 부모 도메인으로 fallback)."""
    if not domain:
        return default
    parts = domain.split(".")
    for i in range(len(parts) - 1):  # 최소 2라벨 유지 ('com' 단독 매칭 방지)
        cand = ".".join(parts[i:])
        if cand in DOMAIN_POLARITY:
            return DOMAIN_POLARITY[cand]
    return default


def _news_json_url(raw_dir: str, prefix: str, news_id: str) -> str:
    """FakeNewsNet raw에서 news_id의 url 필드를 읽어 반환 (없으면 '')."""
    for label in ("real", "fake"):
        path = os.path.join(raw_dir, prefix, label, news_id, "news content.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return (data.get("url") or "").strip()
            except Exception:
                return ""
    return ""


def build_polarity(news_ids, raw_dir: str, prefix: str):
    """
    news_id 리스트 → ({news_id: polarity(0/1/2)}, stats).
      raw_dir : FakeNewsNet raw 루트
      prefix  : 'gossipcop' | 'politifact'
    URL 없음/미등재 도메인 → 중립(1).
    """
    out = {}
    stats = {0: 0, 1: 0, 2: 0, "no_url": 0}
    for nid in news_ids:
        url = _news_json_url(raw_dir, prefix, str(nid))
        if not url:
            stats["no_url"] += 1
            out[str(nid)] = NEUTRAL
            continue
        pol = lookup_polarity(extract_domain(url))
        out[str(nid)] = pol
        stats[pol] += 1
    return out, stats
