"""
통합 모델 인터페이스.

세 모델(REC4MIT/HDInt/PRISM)은 학습 절차가 근본적으로 다르다
(이벤트 메모리 vs 2단 분리기 vs diffusion). 따라서 encode/predict를 억지로
통일하지 않고, **데이터·분할·후보·평가를 공유**하는 선에서 두 메서드만 요구한다:

  fit(train, val, shared, log)   : 각 모델의 원본 학습 루프를 그대로 수행
  rank_all(ctx_ids, ctx_mask, user) -> scores[B, N+1]
      : 전체 뉴스 라이브러리에 대한 점수 (idx0=패딩은 -inf). 통합 평가기가 사용.

이렇게 하면 "동일 데이터로 공정 비교"라는 목표를 달성하면서
각 재구현의 핵심(적대 loss, OT, diffusion)을 손상 없이 보존한다.
"""

import os
from abc import ABC, abstractmethod

import torch


class BaseRecModel(ABC):
    name: str = "base"

    def __init__(self, shared: dict, emb_dir: str, args: dict, device: torch.device):
        self.shared = shared
        self.emb_dir = emb_dir          # processed/{ds}/{model}
        self.args = args
        self.device = device
        self.num_news = shared["meta"]["num_news"]
        self.num_users = shared["meta"]["num_users"]

    @abstractmethod
    def fit(self, train: list, val: list, log) -> None:
        """원본 학습 루프. train/val = 인스턴스 리스트 [[u,[ctx],tgt], ...]."""

    @abstractmethod
    @torch.no_grad()
    def rank_all(self, ctx_ids: torch.Tensor, ctx_mask: torch.Tensor,
                 user: torch.Tensor) -> torch.Tensor:
        """
        ctx_ids [B,L], ctx_mask [B,L], user [B] →
        scores [B, N+1] (전체 뉴스에 대한 추천 점수, idx0 = -inf).
        """

    # 평가용 공통 헬퍼: 가짜뉴스 확률 P(fake) [N+1] (억제 지표 계산용, 선택적)
    def fake_prob(self) -> torch.Tensor | None:
        return None

    # 학습된 가중치 저장 (fit() 완료 후 호출). 단일 nn.Module 기준 기본 구현.
    # PRISM처럼 모듈이 여러 개면 어댑터에서 오버라이드.
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    # 저장된 가중치 복원 (학습 없이 평가/추론용). save()와 짝.
    # __init__에서 모델 구조·임베딩 테이블은 이미 구성되므로 state_dict만 주입.
    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.model.eval()
