# -*- coding: utf-8 -*-
"""
단계 6 — 비교 모델. 계획서 13.1절의 필수 6개를 **같은 인터페이스**로 맞춥니다.

    이름          계열        무엇을 보나
    naive        무조정      조정을 안 하면 어떻게 되는지의 하한
    slearner     회귀조정    처치를 변수 하나로 넣은 단일 모형
    tlearner     회귀조정    처치·대조를 따로 적합 (XGBoost ×2)
    cforest      트리        인과숲 — 이질성 추정의 산업 표준
    tarnet       표현학습    공유표현 + 두 머리 (CDT 이중머리의 정적 원형)
    cdt          시퀀스      제안 모델

공통 인터페이스
    m = Model(**hp)
    m.fit(std, outc, h=1)          # h = 관측창 인덱스 (0:6개월 1:12개월 2:24개월)
    tau = m.predict_tau(std, outc, h=1)   # (N,) 개인별 처치효과
    yhat = m.predict_y(std, outc, h=1, a=0/1)

**정적 모델은 「평탄화 버전」으로 넣습니다** — 마지막 시점의 상태만 씁니다.
이것이 「왜 시퀀스가 필요한가」의 직접 대조군입니다(계획서 13.2절 각주).
시퀀스 모델이 평탄화 모델을 못 이기면 이 논문의 전제가 무너집니다. 그 결과도 그대로 보고합니다.

═══════════════════════════════════════════════════════════════════════
icml50 의 CDT 를 옮기면서 고친 것 (원본 그대로 쓰면 안 되는 이유)

  A. **보상 손실의 마스크 차원이 안 맞았습니다.**
     정적 데이터에서 out_r·target_r 은 (B,1) 인데 mask.unsqueeze(-1) 은 (B,T,1) 이라
     브로드캐스트로 엉뚱한 값이 곱해졌습니다. 시점별로 마스크를 맞춰 곱하도록 고쳤습니다.
  B. **`hufExam` 전역 변수가 학습을 중간에 반환시켰습니다.** 디버그 찌꺼기라 없앴습니다.
  C. **완전히 가려진 행에서 softmax 가 NaN 을 냅니다.**
     단계 2 에서 상태를 1시점 지연시키며 M[:, :lag]=0 을 넣었으므로 실제로 생길 수 있습니다.
     행 전체가 -inf 면 그 행을 균등분포로 두도록 막았습니다.
  D. **행동이 범주(직종 37종)인데 Tanh 회귀로 복원했습니다.**
     범주는 교차엔트로피로 맞춰야 합니다. 이진 처치(IHDP·Jobs)와 37종 직종을
     같은 코드로 다루도록 행동 머리를 분류기로 바꿨습니다.
═══════════════════════════════════════════════════════════════════════
"""
import math
import warnings

import numpy as np

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════ 특징 만들기
def flatten_features(std, outc, use_last=True, extra_summary=True):
    """
    시퀀스 → 정적 모델이 먹을 (N, D) 표.

    처치 시점 t* 의 **직전** 상태를 씁니다(t* 자체가 아니라). 단계 2 의 지연과 같은 이유로,
    t* 시점 상태에는 그 달의 처치가 이미 반영돼 있을 수 있기 때문입니다.
    `extra_summary` 면 t* 이전 구간의 평균·기울기를 덧붙입니다 — 시퀀스 모델과의
    비교를 공정하게 하려는 것입니다(정적 모델에 이력 요약을 안 주면 이기는 게 당연해집니다).
    """
    X = std.X
    ts = np.asarray(outc['TSTAR'], int)
    N, T, D = X.shape
    t0 = np.clip(ts - 1, 0, T - 1)
    rows = np.arange(N)
    feats = [X[rows, t0, :]]
    if extra_summary:
        mean = np.zeros((N, D), np.float32)
        slope = np.zeros((N, D), np.float32)
        for i in range(N):
            e = max(int(ts[i]), 1)
            seg = X[i, :e, :]
            mean[i] = seg.mean(0)
            if e >= 2:
                slope[i] = seg[-1] - seg[0]
        feats += [mean, slope]
    feats.append(ts.reshape(-1, 1).astype(np.float32))       # 달력시간(P9 시간교란)
    return np.concatenate(feats, 1).astype(np.float32)


def _standardizer(F, train_mask):
    """
    train 기준 표준화. **분산이 0 인 열은 나누지 않습니다.**
    `sd + 1e-8` 로 두면 train 에서 상수인 열이 valid·test 에서 조금만 달라도
    |z| 가 1e7 단위로 튀어 신경망이 통째로 망가집니다(실제로 겪었습니다 — 14개 열).
    """
    mu = F[train_mask].mean(0)
    sd = F[train_mask].std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)


def action_index(std, mode='binary'):
    """
    행동을 0..A-1 정수로.

    **mode='binary' 가 기본입니다 — 표 2·3·4 의 반사실 대비가 여기서 정해집니다.**
    반사실은 `predict_y(a=1) - predict_y(a=0)`, 곧 **행동부호 1 과 0 의 대비**입니다.
    그러므로 행동 채널에는 「훈련을 받았는가(0/1)」가 들어가야 합니다.

    v73 까지는 `std.AIDX`(훈련 직종 코드)를 그대로 썼고, 그 결과가 이렇습니다:
      · empservicesyntv2 — AIDX 가 **전부 0**. `a.max() < 0` 이 아니라서 이진으로
        되돌아가지 못했고, 어휘 크기 1 짜리 임베딩이 만들어져 행동 토큰이 상수가
        되었습니다. τ 가 상수(-0.56, sd 0.49)로 붕괴한 원인입니다.
      · empservicereal — AIDX 는 30종 직종코드인데 **부호 1 은 한 번도 안 나옵니다**
        (분포 [999, 0, 259, 1, ...]). 학습된 적 없는 임베딩끼리 뺀 값이 τ 가 되어
        다른 모형이 +7~13 을 줄 때 홀로 -3.8 이 나왔습니다.
    두 자료 모두 `std.A` 에는 균형 잡힌 0/1 이 들어 있습니다(실 999/1001, v2 4060/3940).

    mode='code' 는 직종별 이질성 분석 전용입니다(τ_j = ŷ(직종 j) - ŷ(훈련없음)).
    어휘가 없는 자료에서는 자동으로 이진으로 되돌립니다.
    반환 어휘 크기는 최소 2 — `nn.Embedding(1, h)` 도 상수 토큰이라 못 씁니다.
    """
    a = None
    if mode == 'code':
        c = np.asarray(std.AIDX[..., 0])
        if c.max() > 0:
            a = c
    if a is None:
        a = np.asarray(std.A[..., 0])
        if a.max() < 0:
            a = np.asarray(std.AIDX[..., 0])
    a = np.clip(a, 0, None).astype(np.int64)
    return a, max(2, int(a.max()) + 1)


def _yh(outc, name, h):
    return np.asarray(outc[name][:, h], np.float64)


def _obs(outc, h):
    return np.asarray(outc['OBS'][:, h], bool)


def _fit_mask(std, outc, h, split, name='Y3'):
    """학습에 쓸 사람 = 그 분할 ∧ 우측절단 아님 ∧ 결과가 관측됨."""
    m = (std.SPLIT == split) & _obs(outc, h)
    y = _yh(outc, name, h)
    return m & np.isfinite(y)


# ══════════════════════════════════════════════ 공통 뼈대
class Base:
    name = 'base'

    def __init__(self, **hp):
        self.hp = hp
        self.fitted = False

    def fit(self, std, outc, h=1, y='Y3', split=0):
        raise NotImplementedError

    def predict_y(self, std, outc, h=1, a=0):
        raise NotImplementedError

    def predict_tau(self, std, outc, h=1):
        return self.predict_y(std, outc, h, a=1) - self.predict_y(std, outc, h, a=0)


class Naive(Base):
    """조정 없음. 훈련군 평균 − 대조군 평균을 모두에게 같은 값으로 줍니다."""
    name = 'naive'

    def fit(self, std, outc, h=1, y='Y3', split=0):
        m = _fit_mask(std, outc, h, split, y)
        yy, a = _yh(outc, y, h), np.asarray(outc['TREATED'], bool)
        self.m1 = float(np.nanmean(yy[m & a])) if (m & a).any() else np.nan
        self.m0 = float(np.nanmean(yy[m & ~a])) if (m & ~a).any() else np.nan
        self.fitted = True
        return self

    def predict_y(self, std, outc, h=1, a=0):
        return np.full(std.N, self.m1 if a else self.m0)


class SLearner(Base):
    """처치를 변수 하나로 넣은 단일 모형. 기본은 릿지(선형)."""
    name = 'slearner'

    def fit(self, std, outc, h=1, y='Y3', split=0):
        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        F = flatten_features(std, outc)
        a = np.asarray(outc['TREATED'], float).reshape(-1, 1)
        m = _fit_mask(std, outc, h, split, y)
        self.mdl = make_pipeline(StandardScaler(),
                                 RidgeCV(alphas=np.logspace(-3, 3, 13)))
        self.mdl.fit(np.hstack([F[m], a[m]]), _yh(outc, y, h)[m])
        self.fitted = True
        return self

    def predict_y(self, std, outc, h=1, a=0):
        F = flatten_features(std, outc)
        aa = np.full((std.N, 1), float(a))
        return self.mdl.predict(np.hstack([F, aa]))


class TLearner(Base):
    """처치군·대조군에 각각 따로 적합. XGBoost 두 개."""
    name = 'tlearner'

    def fit(self, std, outc, h=1, y='Y3', split=0):
        F = flatten_features(std, outc)
        a = np.asarray(outc['TREATED'], bool)
        m = _fit_mask(std, outc, h, split, y)
        yy = _yh(outc, y, h)
        try:
            from xgboost import XGBRegressor
            mk = lambda: XGBRegressor(
                n_estimators=self.hp.get('n_estimators', 400),
                max_depth=self.hp.get('max_depth', 4),
                learning_rate=self.hp.get('lr', 0.05),
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, n_jobs=self.hp.get('n_jobs', 4),
                random_state=self.hp.get('seed', 0), verbosity=0)
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            mk = lambda: GradientBoostingRegressor(random_state=self.hp.get('seed', 0))
        self.m1, self.m0 = mk(), mk()
        self.m1.fit(F[m & a], yy[m & a])
        self.m0.fit(F[m & ~a], yy[m & ~a])
        self.fitted = True
        return self

    def predict_y(self, std, outc, h=1, a=0):
        F = flatten_features(std, outc)
        return (self.m1 if a else self.m0).predict(F)


class CausalForest(Base):
    """
    인과숲 (Wager & Athey 2018). econml 이 있으면 그것을, 없으면
    **정직한 표본분할 T러너**로 대체하고 그 사실을 남깁니다.
    """
    name = 'cforest'

    def fit(self, std, outc, h=1, y='Y3', split=0):
        F = flatten_features(std, outc)
        a = np.asarray(outc['TREATED'], float)
        m = _fit_mask(std, outc, h, split, y)
        yy = _yh(outc, y, h)
        self.backend = 'econml'
        try:
            from econml.grf import CausalForest as _CF
            self.mdl = _CF(n_estimators=self.hp.get('n_estimators', 1000),
                           min_samples_leaf=self.hp.get('min_samples_leaf', 5),
                           max_depth=self.hp.get('max_depth', None),
                           honest=True, random_state=self.hp.get('seed', 0),
                           n_jobs=self.hp.get('n_jobs', 4))
            self.mdl.fit(F[m], a[m], yy[m])
        except ImportError:
            self.backend = 'sklearn-대체(econml 없음)'
            self.mdl = TLearner(**self.hp).fit(std, outc, h, y, split)
        self.fitted = True
        return self

    def predict_tau(self, std, outc, h=1):
        F = flatten_features(std, outc)
        if self.backend == 'econml':
            return np.asarray(self.mdl.predict(F)).ravel()
        return self.mdl.predict_tau(std, outc, h)

    def predict_y(self, std, outc, h=1, a=0):
        if self.backend != 'econml':
            return self.mdl.predict_y(std, outc, h, a)
        raise NotImplementedError('인과숲은 τ 만 냅니다. predict_tau 를 쓰십시오.')


# ══════════════════════════════════════════════ 신경망 (torch)
def _torch():
    import torch
    return torch


# ══════════════════════════════════════════════ 학습률 선택 (valid 전용)
LR_GRID = (3e-4, 1e-3, 3e-3, 1e-2)      # CDT
LR_GRID_T = (3e-4, 1e-3, 3e-3)          # TARNet — 기준모형에도 **같은 예산**을 줍니다
DROP_GRID = (0.0, 0.1)                  # CDT 드롭아웃


def safe_chunk(K, n_head=4, budget_gb=1.0, cap=4096, floor=16):
    """
    한 번에 통과시킬 사람 수. **어텐션은 토큰수의 제곱으로 큽니다.**

    토큰수 = 3K (시점마다 R·S·A 세 토큰). 사람 하나당 어텐션 텐서는
    대략 3K × 3K × n_head × 4바이트 이고, q@k · masked_fill · softmax 로
    서너 벌이 동시에 삽니다. K=120 이면 사람 하나에 약 8 MB —
    8,000명을 한 번에 넣으면 60 GB 가 필요합니다(12 GB 카드에서 K=60 부터 터졌습니다).
    """
    t = 3 * max(int(K), 1)
    per = t * t * int(n_head) * 4 * 4          # 바이트/사람
    return int(min(cap, max(floor, budget_gb * 1e9 / max(per, 1))))


def valid_uplift(tau_hat, std, outc, h, y, split=1):
    """
    **검증분할의 Qini.** 반사실 참값이 필요 없습니다 — 관측 결과 Y, 처치 A,
    그리고 τ̂ 의 **순위**만 있으면 계산됩니다. 그래서 test 를 열지 않고도
    「이 모형이 사람을 잘 줄 세우는가」를 고를 수 있습니다.

    왜 필요한가: 실데이터의 채점지표는 Qini 인데 v77 까지는 **valid 제곱오차**로
    골랐습니다. 사실적 예측이 같아도 τ̂ 의 순위는 전혀 다를 수 있습니다.
    재는 것과 고르는 것이 어긋나 있었습니다.
    """
    m = _fit_mask(std, outc, h, split, y)
    if m.sum() < 20:
        return np.nan
    a = np.asarray(outc['TREATED'], bool)[m]
    if a.all() or (~a).any() is False or a.sum() < 5 or (~a).sum() < 5:
        return np.nan
    return qini(np.asarray(tau_hat)[m], a, _yh(outc, y, h)[m])


def _grid_keys(loc, keys=('lr', 'dropout')):
    """값이 하나가 아니라 여러 개로 주어진 초매개변수 이름."""
    if loc.get('configs'):
        return ['configs']
    return [k for k in keys
            if k in loc and loc[k] is not None and not np.isscalar(loc[k])]


def _fit_score(cand, loc):
    """후보를 고르는 점수. 클수록 좋게 맞춰 돌려줍니다."""
    if loc.get('select') == 'qini':
        q = getattr(cand, 'valid_qini', None)
        if q is not None and np.isfinite(q):
            return float(q)
        return -np.inf
    bv = getattr(cand, 'best_valid', None)
    return -float(bv) if (bv is not None and np.isfinite(bv)) else -np.inf


def _fit_grid(model, loc, keys=('lr', 'dropout')):
    """
    후보 조합을 모두 돌려 **valid 손실이 가장 낮은 것**만 남깁니다. test 는 열지 않습니다.

    왜 필요한가: v73 은 IHDP valid 로 고른 lr=1e-2 를 네 자료에 그대로 썼습니다.
    IHDP(n=672·K=1)에서는 맞았지만 실데이터(n=1345·K=30·D=70)에서는 커서
    조기종료가 1~2에폭에 걸렸습니다. 자료마다 고르면 이 문제가 사라집니다.
    심사자가 반드시 묻는 「기준모형도 같은 예산으로 조율했는가」에 대한 답이기도 해서
    TARNet 에도 같은 장치를 답니다.
    """
    import itertools
    std, outc = loc['std'], loc['outc']
    run = loc.get('run')
    cfgs = loc.get('configs')
    if cfgs:
        # 후보를 **손으로 고른 목록**으로 줍니다. 격자를 다 곱하면 후보가 폭발합니다
        #   (lr4 × drop3 × hidden3 × layer2 × head2 = 144). 뜻이 있는 조합만 봅니다.
        gk = sorted({k for c in cfgs for k in c})
        combos = [tuple(c.get(k, loc.get(k)) for k in gk) for c in cfgs]
    else:
        gk = _grid_keys(loc, keys)
        combos = list(itertools.product(*[list(loc[k]) for k in gk]))
    base = {k: v for k, v in loc.items()
            if k not in ('self', 'model', 'std', 'outc', 'run', 'configs', '__class__')
            and k not in gk}
    best, best_kw, table, cand, bestsc = None, None, [], None, -np.inf
    for combo in combos:
        kw = dict(zip(gk, combo))
        cand = type(model)(**model.hp)
        cand.fit(std, outc, run=None, **base, **kw)
        sc = _fit_score(cand, loc)
        row = dict(kw)
        row['valid'] = getattr(cand, 'best_valid', None)
        row['valid_qini'] = getattr(cand, 'valid_qini', None)
        row['best_epoch'] = getattr(cand, 'best_epoch', None)
        table.append(row)
        if np.isfinite(sc) and (best is None or sc > bestsc):
            best, best_kw, bestsc = cand, kw, sc
    if best is None:                      # 전부 실패하면 마지막 후보라도 돌려줍니다
        best, best_kw = cand, kw
    model.__dict__.update({k: v for k, v in best.__dict__.items() if k != 'hp'})
    model.hp_chosen, model.hp_table = best_kw, table
    model.lr_chosen = best_kw.get('lr')
    if loc.get('verbose'):
        for r in table:
            print('    %-42s valid %8.5f  qini %10s  ep %s%s'
                  % (' '.join('%s=%s' % (k, r[k]) for k in gk),
                     r['valid'] if r['valid'] is not None else float('nan'),
                     ('%.1f' % r['valid_qini']) if r.get('valid_qini') is not None
                     and np.isfinite(r.get('valid_qini', np.nan)) else '-',
                     r['best_epoch'],
                     '  ←선택' if all(r[k] == best_kw[k] for k in gk) else ''))
    if run is not None:
        run.cfg['초매개_선택'] = best_kw
        run.cfg['선택기준'] = loc.get('select', 'loss')
        run.cfg['초매개_후보'] = table
        run._write_cfg()
    return model


class TARNet(Base):
    """
    Shalit et al. (2017). 공유표현 Φ(x) → 처치·대조 두 머리.
    CDT 이중머리의 정적 원형이라 반드시 넣습니다.
    """
    name = 'tarnet'

    def fit(self, std, outc, h=1, y='Y3', split=0, valid_split=1,
            epochs=200, patience=20, lr=LR_GRID_T, hidden=200, select='loss',
            configs=None, device=None, run=None, verbose=False):
        if _grid_keys(locals()):
            return _fit_grid(self, locals())
        torch = _torch()
        import torch.nn as nn
        from . import env as cenv
        from .runs import EarlyStop
        dev = device or cenv.pick_device()

        F = flatten_features(std, outc)
        mu, sd = _standardizer(F, std.SPLIT == 0)
        F = (F - mu) / sd
        self._norm = (mu, sd)
        a = np.asarray(outc['TREATED'], float)
        yy = _yh(outc, y, h)
        ytr = yy[_fit_mask(std, outc, h, split, y)]
        self._ymu, self._ysd = float(ytr.mean()), float(ytr.std() + 1e-8)

        def pack(sp):
            m = _fit_mask(std, outc, h, sp, y)
            return (torch.tensor(F[m], dtype=torch.float32, device=dev),
                    torch.tensor(a[m], dtype=torch.float32, device=dev),
                    torch.tensor((yy[m] - self._ymu) / self._ysd,
                                 dtype=torch.float32, device=dev))
        Xtr, Atr, Ytr = pack(split)
        Xva, Ava, Yva = pack(valid_split)
        # [고침 E] 재적합(검증을 학습에 합친 경우) 검증이 빈 집합이 됩니다. 옛 코드는
        #   빈 검증에서 vl=0.0 이 매 에폭 반복되어 vl<es.best 가 영원히 거짓 →
        #   **1에폭째 가중치가 최고로 복원**되는 사고가 났습니다(97판 D' 4.4438 동결의 원인).
        #   검증이 없으면 조기종료·최고복원을 모두 끄고 마지막 에폭 가중치를 씁니다.
        has_valid = int(Yva.numel()) > 0

        D = F.shape[1]
        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.phi = nn.Sequential(nn.Linear(D, hidden), nn.ELU(),
                                         nn.Linear(hidden, hidden), nn.ELU())
                mk = lambda: nn.Sequential(nn.Linear(hidden, hidden), nn.ELU(),
                                           nn.Linear(hidden, 1))
                self.h1, self.h0 = mk(), mk()
            def forward(self, x):
                z = self.phi(x)
                return self.h0(z).squeeze(-1), self.h1(z).squeeze(-1)

        cenv.set_seed(self.hp.get('seed', 0))
        net = _Net().to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
        es = EarlyStop(patience=patience)
        best = None
        bs = self.hp.get('batch', 256)
        for ep in range(epochs):
            net.train()
            perm = torch.randperm(len(Ytr), device=dev)
            tot = 0.0
            for b0 in range(0, len(Ytr), bs):
                j = perm[b0:b0 + bs]
                p0, p1 = net(Xtr[j])
                loss = (((1 - Atr[j]) * (p0 - Ytr[j]) ** 2).sum() +
                        (Atr[j] * (p1 - Ytr[j]) ** 2).sum()) / max(len(j), 1)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss) * len(j)
            loss = tot / max(len(Ytr), 1)
            net.eval()
            if has_valid:
                with torch.no_grad():
                    q0, q1 = net(Xva)
                    vl = float((((1 - Ava) * (q0 - Yva) ** 2).sum() +
                                (Ava * (q1 - Yva) ** 2).sum()) / max(len(Yva), 1))
            else:
                vl = float('nan')
            if run is not None:
                run.log(epoch=ep, train=float(loss), valid=vl)
            if has_valid:
                if es.best is None or vl < es.best:
                    best = {k: v.detach().clone() for k, v in net.state_dict().items()}
                if es.step(vl, ep):
                    break
        if best and has_valid:
            net.load_state_dict(best)
        self.net, self.dev = net.eval(), dev
        self.best_epoch = es.best_epoch if has_valid else ep
        self.best_valid = es.best if has_valid else None
        self.epochs_run = ep + 1
        self.used_final_weights = not has_valid
        self.fitted = True
        # 선택기준이 qini 면 검증 Qini 를 함께 남깁니다(고르는 쪽은 _fit_grid 가 봅니다).
        self.valid_qini = (float(valid_uplift(self.predict_tau(std, outc, h=h),
                                              std, outc, h, y, valid_split))
                           if select == 'qini' else None)
        return self

    def _fw(self, std, outc):
        torch = _torch()
        mu, sd = self._norm
        F = (flatten_features(std, outc) - mu) / sd
        with torch.no_grad():
            p0, p1 = self.net(torch.tensor(F, dtype=torch.float32, device=self.dev))
        return (p0.cpu().numpy() * self._ysd + self._ymu,
                p1.cpu().numpy() * self._ysd + self._ymu)

    def predict_y(self, std, outc, h=1, a=0):
        return self._fw(std, outc)[1 if a else 0]


# ══════════════════════════════════════════════ CDT
def build_cdt_modules():
    """torch 가 있을 때만 클래스를 만듭니다(윈도우 CPU 에서 import 만 해도 되게)."""
    torch = _torch()
    import torch.nn as nn
    import torch.nn.functional as Fn

    class MaskedCausalAttention(nn.Module):
        def __init__(self, h_dim, n_head, attn_pdrop=0.0, resid_pdrop=0.0, causal=True):
            super().__init__()
            assert h_dim % n_head == 0
            self.causal = bool(causal)          # 소거연구용 — False 면 양방향
            self.n_head, self.head_dim = n_head, h_dim // n_head
            self.c_attn = nn.Linear(h_dim, 3 * h_dim)
            self.c_proj = nn.Linear(h_dim, h_dim)
            self.ad, self.rd = nn.Dropout(attn_pdrop), nn.Dropout(resid_pdrop)

        def forward(self, x, mask=None):
            B, T, C = x.size()
            q, k, v = self.c_attn(x).split(C, dim=2)
            sh = lambda z: z.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            q, k, v = sh(q), sh(k), sh(v)
            causal = (torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
                      if self.causal
                      else torch.ones(T, T, device=x.device, dtype=torch.bool))
            keep = causal.view(1, 1, T, T)
            if mask is not None:
                if mask.dim() == 2:
                    mask = mask.unsqueeze(1).unsqueeze(1)
                keep = keep & (mask > 0)
            # [고침 C] 행 전체가 막히면 softmax 가 NaN 을 냅니다.
            #   단계 2 의 상태 지연으로 M 앞쪽이 0 이라 실제로 생깁니다.
            #   그런 행은 자기 자신만 보게 열어 둡니다(대각선).
            dead = ~keep.any(-1, keepdim=True)
            eye = torch.eye(T, device=x.device, dtype=torch.bool).view(1, 1, T, T)
            keep = keep | (dead & eye)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            att = att.masked_fill(~keep, float('-inf'))
            att = self.ad(Fn.softmax(att, dim=-1))
            y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
            return self.rd(self.c_proj(y))

    class CausalBlock(nn.Module):
        def __init__(self, h, n_head, resid_pdrop=0.0, attn_pdrop=0.0, causal=True):
            super().__init__()
            self.ln1, self.ln2 = nn.LayerNorm(h), nn.LayerNorm(h)
            self.attn = MaskedCausalAttention(h, n_head, attn_pdrop, resid_pdrop, causal)
            self.mlp = nn.Sequential(nn.Linear(h, 4 * h), nn.GELU(),
                                     nn.Linear(4 * h, h), nn.Dropout(resid_pdrop))

        def forward(self, x, mask=None):
            x = x + self.attn(self.ln1(x), mask=mask)
            return x + self.mlp(self.ln2(x))

    class CDTNet(nn.Module):
        """
        (R, S, A) 세 토큰을 시점마다 쌓아 인과 어텐션으로 읽습니다.

        MOT — 학습·추론 모두에서 R 토큰의 값을 0 으로 가립니다. 가리지 않으면
        모형이 R 을 그대로 베껴 항등사상으로 붕괴합니다(소거연구 표 5 의 첫 줄).
        이중머리 — 결과 = 기저(상태만) + 상호작용(상태×행동). 예후가 처치신호를
        덮지 않도록 구조로 분리합니다.
        """
        def __init__(self, state_dim, n_action, hidden=128, K=30, n_layer=3,
                     n_head=4, max_ep_len=4096, resid_pdrop=0.0, attn_pdrop=0.0,
                     dual=True, causal=True, head='additive'):
            super().__init__()
            self.K, self.hidden, self.n_action = K, hidden, n_action
            self.dual = bool(dual)              # 소거연구용 — False 면 단일 머리
            self.head = str(head)               # 'additive' | 'twin'
            self.s_emb = nn.Linear(state_dim, hidden)
            # [고침 D] 행동은 범주입니다. Tanh 회귀가 아니라 임베딩·분류기로 다룹니다.
            self.a_emb = nn.Embedding(n_action, hidden)
            self.r_emb = nn.Linear(1, hidden)
            self.t_emb = nn.Embedding(max_ep_len, hidden)
            self.ln = nn.LayerNorm(hidden)
            self.blocks = nn.ModuleList([
                CausalBlock(hidden, n_head, resid_pdrop, attn_pdrop, causal)
                for _ in range(n_layer)])
            self.ln_f = nn.LayerNorm(hidden)
            self.head_action = nn.Linear(hidden, n_action)          # 성향(행동 복원)
            self.head_base = nn.Linear(hidden, 1)                   # 기저 예후
            self.head_inter = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                            nn.Linear(hidden, 1))   # 처치 상호작용
            # 'twin' — 처치 팔마다 결과머리를 따로 둡니다(TARNet 을 시퀀스 표현 위에 올린 꼴).
            #   τ = y_1(s) - y_0(s) 가 **상태표현만의 매끈한 함수**가 되어, 마지막 행동
            #   토큰 하나를 뒤집어 얻던 τ 보다 훨씬 덜 흔들립니다. 실데이터에서 CDT 의 τ 가
            #   시드마다 상쇄되어 앙상블이 오히려 나빠졌던 문제(Qini 989→222)를 겨눈 것입니다.
            #   s_rep 는 인과 어텐션 순서(R,S,A) 때문에 그 시점의 행동을 못 봅니다 — 누출 없음.
            self.head_arm = nn.Linear(hidden, n_action)

        def forward(self, s, a, rtg, ts, mask=None, mask_return=True):
            s, a, ts = s[:, -self.K:], a[:, -self.K:], ts[:, -self.K:]
            a_idx = a
            rtg = rtg[:, -self.K:]
            if mask is not None:
                mask = mask[:, -self.K:]
            r_in = torch.zeros_like(rtg) if mask_return else rtg
            te = self.t_emb(ts)
            se, ae, re = self.s_emb(s) + te, self.a_emb(a) + te, self.r_emb(r_in) + te
            B, T, H = se.shape
            x = self.ln(torch.stack((re, se, ae), dim=2).view(B, T * 3, H))
            m3 = None
            if mask is not None:
                m3 = mask.unsqueeze(-1).expand(B, T, 3).reshape(B, T * 3)
            for blk in self.blocks:
                x = blk(x, mask=m3)
            x = self.ln_f(x).view(B, T, 3, H)
            s_rep, a_rep = x[:, :, 1, :], x[:, :, 2, :]
            if self.head == 'twin':
                out = self.head_arm(s_rep).gather(-1, a_idx.unsqueeze(-1))
            elif self.dual:
                out = self.head_base(s_rep) + self.head_inter(a_rep)
            else:
                out = self.head_inter(a_rep)         # 단일 머리 = 행동 토큰만 읽음
            return self.head_action(s_rep), out      # (B,T,A) 로짓 · (B,T,1) 결과

    return CDTNet


class CDT(Base):
    """제안 모델. 반사실은 t* 의 행동만 a 로 바꾸고 나머지는 그대로 두어 얻습니다."""
    name = 'cdt'

    def fit(self, std, outc, h=1, y='Y3', split=0, valid_split=1,
            epochs=200, patience=30, lr=LR_GRID, hidden=128, K=30, n_layer=3,
            n_head=4, lambda_action=0.3, batch=128, warmup=0.1, action='binary',
            dropout=DROP_GRID, mot=True, dual=True, causal=True, state_drop=None,
            head='additive', select='loss', eval_every=5, configs=None,
            device=None, run=None, verbose=False):
        # lr 은 **자료마다 valid 손실로 고릅니다**(test 는 안 봅니다). v73 은 IHDP 에서 고른
        # 1e-2 하나를 네 자료에 그대로 썼고, 실데이터에서 조기종료가 1~2에폭에 걸렸습니다
        # (best_epoch 2 · τ 가 다른 모형과 부호까지 반대). 자료마다 표본수·차원·창길이가
        # 다르니 학습률 하나가 맞을 이유가 없습니다.
        if _grid_keys(locals()):
            return _fit_grid(self, locals())
        torch = _torch()
        from . import env as cenv
        from .runs import EarlyStop
        dev = device or cenv.pick_device()
        cenv.set_seed(self.hp.get('seed', 0))

        N, T, D = std.X.shape
        K = min(int(K), int(T))          # 창이 궤적보다 길 이유가 없습니다(IHDP·Jobs 는 T=1)
        ts = np.asarray(outc['TSTAR'], int)
        AIDX, A = action_index(std, mode=action)
        self.K, self.A, self.dev, self.h, self.yname = K, A, dev, h, y
        self.mot, self.n_head = bool(mot), int(n_head)
        self.pred_chunk = safe_chunk(K, n_head)

        Xm = std.X[std.SPLIT == 0].reshape(-1, D)
        self._mu, self._sd = _standardizer(Xm, np.ones(len(Xm), bool))
        yy = _yh(outc, y, h)
        ytr = yy[_fit_mask(std, outc, h, split, y)]
        self._ymu, self._ysd = float(ytr.mean()), float(ytr.std() + 1e-8)

        # state_drop — 이름에 이 조각이 든 상태 열을 0(=train 평균)으로 막습니다.
        #   `상태_생애_*누적경력기간`·`효과잔존량` 처럼 **이미 생애 전체를 요약해 둔 열**을
        #   빼면, 그 정보는 오직 시퀀스로만 되살릴 수 있습니다. K 절단(그림 3)이
        #   평평하게 나온 까닭이 여기 있는지를 가르는 실험입니다.
        _cols = list((getattr(std, 'meta', None) or {}).get('state_cols') or [])
        _keep = np.ones(D, np.float32)
        if state_drop:
            _pat = (state_drop,) if isinstance(state_drop, str) else tuple(state_drop)
            for _d, _c in enumerate(_cols[:D]):
                if any(_t in _c for _t in _pat):
                    _keep[_d] = 0.0
        self.state_kept = int(_keep.sum())

        def window(idx):
            """사람마다 t* 로 끝나는 길이 K 창을 잘라 옵니다."""
            S = np.zeros((len(idx), K, D), np.float32)
            Ai = np.zeros((len(idx), K), np.int64)
            Ms = np.zeros((len(idx), K), np.float32)
            Ts = np.zeros((len(idx), K), np.int64)
            for j, i in enumerate(idx):
                e = int(ts[i]) + 1
                b = max(0, e - K)
                L = e - b
                S[j, K - L:] = ((std.X[i, b:e] - self._mu) / self._sd) * _keep
                Ai[j, K - L:] = AIDX[i, b:e]
                Ms[j, K - L:] = std.M[i, b:e]
                Ts[j, K - L:] = np.arange(b, e)
            return (torch.tensor(S, device=dev), torch.tensor(Ai, device=dev),
                    torch.tensor(Ms, device=dev), torch.tensor(Ts, device=dev))

        tr = np.where(_fit_mask(std, outc, h, split, y))[0]
        va = np.where(_fit_mask(std, outc, h, valid_split, y))[0]
        Str, Atr, Mtr, Ttr = window(tr)
        Sva, Ava, Mva, Tva = window(va)
        Ytr = torch.tensor((yy[tr] - self._ymu) / self._ysd, dtype=torch.float32, device=dev)
        Yva = torch.tensor((yy[va] - self._ymu) / self._ysd, dtype=torch.float32, device=dev)
        # [고침 E] TARNet 과 같은 보호 — 검증이 비면 조기종료·최고복원 없이 끝까지 돕니다.
        has_valid = len(va) > 0

        # 학습 묶음도 K 에 맞춰 줄입니다. 역전파는 층마다 활성값을 붙들고 있어
        # 순전파보다 서너 배를 더 씁니다. K=120 에서 batch 128 은 12 GB 카드가 못 버팁니다.
        batch = int(min(int(batch), safe_chunk(K, n_head, budget_gb=0.5)))
        CDTNet = build_cdt_modules()
        # 드롭아웃 — 실데이터(n=1345)에서 3층 트랜스포머는 몇 에폭 만에 과적합합니다.
        # 0 으로 두면 valid 가 곧장 올라가 조기종료가 1~2에폭에 걸립니다. valid 로 고릅니다.
        net = CDTNet(D, A, hidden=hidden, K=K, n_layer=n_layer, n_head=n_head,
                     resid_pdrop=float(dropout), attn_pdrop=float(dropout),
                     dual=dual, causal=causal, head=head).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        # 워밍업 뒤 코사인 감쇠. 학습률이 높으면 첫 에폭에서 valid 가 튀어 올라
        # 조기종료가 1~2에폭 만에 걸립니다(72판까지 실제로 그랬습니다).
        _steps = max(1, (len(tr) + batch - 1) // batch) * epochs
        _warm = max(1, int(_steps * warmup))
        def _lr_at(k):
            if k < _warm:
                return (k + 1) / _warm
            p = (k - _warm) / max(1, _steps - _warm)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
        sch = torch.optim.lr_scheduler.LambdaLR(opt, _lr_at)
        _min_ep = max(1, int(round(epochs * warmup)))     # 워밍업이 끝나는 에폭
        # select='qini' — **재는 것과 고르는 것을 맞춥니다.** 실데이터의 채점지표는
        #   Qini(줄 세우기)인데 사실적 제곱오차로 고르면 어긋납니다. Qini 는 반사실이
        #   필요 없어 검증분할에서 그대로 계산됩니다 — test 를 열지 않습니다.
        _by_qini = (select == 'qini')
        es = EarlyStop(patience=patience, lower_better=not _by_qini)
        _yva_raw = yy[va]
        _ava = np.asarray(outc['TREATED'], bool)[va]
        _can_q = _by_qini and _ava.sum() >= 5 and (~_ava).sum() >= 5

        def _qini_va():
            """검증분할 사람들의 τ̂ 로 Qini. 마지막 시점의 행동만 1/0 으로 바꿉니다."""
            net.eval()
            with torch.no_grad():
                _o = []
                for _a in (1, 0):
                    _A2 = Ava.clone(); _A2[:, -1] = _a
                    _, _r = net(Sva, _A2, rtg0.expand(len(va), K, 1), Tva,
                                mask=Mva, mask_return=mot)
                    _o.append(_r[:, -1, 0].float().cpu().numpy())
            return qini(_o[0] - _o[1], _ava, _yva_raw)
        self.curve = []
        best = None
        ce = torch.nn.CrossEntropyLoss(reduction='none')
        rtg0 = torch.zeros((1, K, 1), device=dev)

        def step(S, Ai, Ms, Ts, Y, train=True):
            net.train(train)
            # MOT 소거(mot=False)면 R 토큰에 **관측 결과 그대로**를 넣습니다.
            # 그러면 모형이 R 을 베껴 항등사상으로 붕괴하는 것을 표 5 에서 보입니다.
            _rtg = (rtg0.expand(len(S), K, 1) if mot
                    else Y.view(-1, 1, 1).expand(len(S), K, 1))
            logit, rhat = net(S, Ai, _rtg, Ts, mask=Ms, mask_return=mot)
            # [고침 D] 행동은 교차엔트로피
            la = (ce(logit.reshape(-1, self.A), Ai.reshape(-1)).reshape(Ai.shape) * Ms).sum() \
                 / Ms.sum().clamp(min=1)
            # [고침 A] 결과 손실은 **마지막 유효 시점 하나**에 대해서만.
            #   원본은 (B,1) 예측에 (B,T,1) 마스크를 곱해 차원이 어긋났습니다.
            lr_ = ((rhat[:, -1, 0] - Y) ** 2).mean()
            return la, lr_

        for ep in range(epochs):
            perm = torch.randperm(len(tr), device=dev)
            tot = 0.0
            for b0 in range(0, len(tr), batch):
                idx = perm[b0:b0 + batch]
                la, lr_ = step(Str[idx], Atr[idx], Mtr[idx], Ttr[idx], Ytr[idx], True)
                loss = lr_ + lambda_action * la
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step(); sch.step()
                tot += float(loss) * len(idx)
            # valid 도 묶어서 통과시킵니다. 통째로 넣으면 K 가 클 때 여기서 먼저 터집니다.
            with torch.no_grad():
                _vb, _acc = self.pred_chunk, 0.0
                for _b0 in range(0, len(va), _vb):
                    _sl = slice(_b0, _b0 + _vb)
                    _, _l = step(Sva[_sl], Ava[_sl], Mva[_sl], Tva[_sl], Yva[_sl], False)
                    _acc += float(_l) * len(Yva[_sl])
            vl = (_acc / len(va)) if has_valid else float('nan')
            _q = _qini_va() if (_can_q and (ep % max(1, int(eval_every)) == 0
                                            or ep == epochs - 1)) else None
            self.curve.append((ep, tot / max(len(tr), 1), vl, _q))
            if verbose and (ep % 10 == 0 or ep == epochs - 1):
                print('    ep %3d  train %.4f  valid %.4f  lr %.2e'
                      % (ep, tot / max(len(tr), 1), vl, sch.get_last_lr()[0]), flush=True)
            if run is not None:
                run.log(epoch=ep, train=tot / max(len(tr), 1), valid=vl)
            if not has_valid:
                continue          # [고침 E] 검증 없음 — 채점·조기종료 생략, 마지막 에폭 사용
            _sc = vl
            if _by_qini:
                if _q is None or not np.isfinite(_q):
                    continue                      # 이 에폭은 채점 안 함 — 조기종료도 안 셈
                _sc = _q
            if es.best is None or (_sc > es.best if _by_qini else _sc < es.best):
                best = {k: v.detach().clone() for k, v in net.state_dict().items()}
                self._best_loss = vl
            _stop = es.step(_sc, ep)
            # 워밍업 동안에는 학습률이 아직 올라가는 중이라 valid 가 나빠지는 것이 정상입니다.
            # 그 구간의 「나빠짐」을 인내심에 세면 학습이 시작도 하기 전에 멈춥니다.
            if ep + 1 == _min_ep:
                es.bad = 0
                _stop = False
            if _stop and ep + 1 > _min_ep:
                break
        if best and has_valid:
            net.load_state_dict(best)
        self.net = net.eval()
        self._window = window
        self.best_epoch = es.best_epoch if has_valid else ep
        if has_valid:
            self.best_valid = getattr(self, '_best_loss', es.best) if _by_qini else es.best
        else:
            self.best_valid = None
        self.valid_qini = float(_qini_va()) if _can_q else (
            float(es.best) if (_by_qini and es.best is not None) else None)
        self.epochs_run = ep + 1
        self.used_final_weights = not has_valid
        # 에폭 한계에서 멈췄으면 아직 더 배울 수 있다는 뜻입니다.
        self.hit_limit = bool(self.best_epoch is not None and
                              self.best_epoch >= epochs - 1)
        self.fitted = True
        return self

    def predict_y_action(self, std, outc, a_code, chunk=None):
        """
        t* 의 행동만 a_code 로 바꾼 반사실 결과. 나머지 이력은 그대로.

        **묶어서 통과시킵니다.** 8,000명을 한 번에 넣으면 K=60 부터 12 GB 카드가 터집니다
        (어텐션이 (3K)² 로 커집니다). 그래도 부족하면 묶음을 절반으로 줄여 다시 잡습니다.
        """
        torch = _torch()
        N = int(std.N)
        c = int(chunk or getattr(self, 'pred_chunk', None) or
                safe_chunk(self.K, getattr(self, 'n_head', 4)))
        out = np.empty(N, np.float32)
        i = 0
        while i < N:
            idx = np.arange(i, min(i + c, N))
            try:
                S, Ai, Ms, Ts = self._window(idx)
                Ai = Ai.clone()
                Ai[:, -1] = int(a_code)         # t* 의 행동만 교체
                # 추론 때 R 은 언제나 0 입니다(테스트에서는 결과를 모릅니다).
                with torch.no_grad():
                    _, rhat = self.net(
                        S, Ai, torch.zeros((len(idx), self.K, 1), device=self.dev),
                        Ts, mask=Ms, mask_return=getattr(self, 'mot', True))
                out[idx] = rhat[:, -1, 0].float().cpu().numpy()
                del S, Ai, Ms, Ts, rhat
            except RuntimeError as e:
                if 'out of memory' not in str(e).lower() or c <= 16:
                    raise
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                c = max(16, c // 2)
                print('  추론 VRAM 부족 — 묶음을 %d 로 줄여 다시 시도합니다' % c, flush=True)
                continue
            i += len(idx)
        self.pred_chunk = c
        return out.astype(np.float64) * self._ysd + self._ymu

    def predict_y(self, std, outc, h=1, a=0):
        return self.predict_y_action(std, outc, 1 if a else 0)

    def predict_tau_action(self, std, outc, a_code):
        return self.predict_y_action(std, outc, a_code) - self.predict_y_action(std, outc, 0)


MODELS = {m.name: m for m in (Naive, SLearner, TLearner, CausalForest, TARNet, CDT)}
REQUIRED = ('naive', 'slearner', 'tlearner', 'cforest', 'tarnet', 'cdt')

# ══════════════════════════════════════════════ 채점
# 데이터마다 **낼 수 있는 지표가 다릅니다**(계획서 13.2절).
#   ihdp   √PEHE · ε_ATE            (반사실 정답이 설계로 존재)
#   jobs   ε_ATT · 정책위험          (무작위 부분표본이 정답을 줌). PEHE 는 금지
#   syntv2 √PEHE (참값 MU_Y3)
#   real   Qini / AUUC              (반사실 정답 없음). PEHE·ATT오차 금지
def pehe(tau_hat, mu0, mu1, mask=None):
    t = np.asarray(mu1).ravel() - np.asarray(mu0).ravel()
    e = (np.asarray(tau_hat).ravel() - t) ** 2
    if mask is not None:
        e = e[mask]
    return float(np.sqrt(np.nanmean(e)))


def eps_ate(tau_hat, mu0, mu1, mask=None):
    t = np.asarray(mu1).ravel() - np.asarray(mu0).ravel()
    th = np.asarray(tau_hat).ravel()
    if mask is not None:
        t, th = t[mask], th[mask]
    return float(abs(np.nanmean(th) - np.nanmean(t)))


def eps_att(tau_hat, treated, rct, y, att_true):
    """
    Jobs — 무작위 부분표본(RCT=1)에서 ATT 를 재고 정답 0.077940 과 견줍니다.
    관측 표본으로 학습한 모형을 무작위 표본에만 적용하는 것이 요점입니다.
    """
    m = (np.asarray(rct).ravel() > 0) & (np.asarray(treated).ravel() > 0)
    if m.sum() == 0:
        return np.nan
    return float(abs(np.nanmean(np.asarray(tau_hat).ravel()[m]) - att_true))


def policy_risk(tau_hat, treated, rct, y):
    """정책 위험 — 「τ̂>0 이면 처치」 정책을 무작위 표본에서 채점합니다."""
    r = np.asarray(rct).ravel() > 0
    t = np.asarray(treated).ravel() > 0
    yy = np.asarray(y).ravel()
    pi = np.asarray(tau_hat).ravel() > 0
    a = r & pi & t
    b = r & (~pi) & (~t)
    if r.sum() == 0:
        return np.nan
    p = float(pi[r].mean())
    # τ̂ 가 상수면 「전원 처치」나 「전원 비처치」가 되어 한쪽이 빕니다.
    # 그때는 그쪽 항만으로 계산합니다(NaN 을 내면 naive·slearner 가 표에서 사라집니다).
    v1 = yy[a].mean() if a.sum() else 0.0
    v0 = yy[b].mean() if b.sum() else 0.0
    if a.sum() == 0 and p > 0:
        return np.nan
    if b.sum() == 0 and p < 1:
        return np.nan
    return float(1.0 - (v1 * p + v0 * (1 - p)))


def _uplift_order(tau_hat, seed=None):
    """[고침 F] τ̂ 내림차순 순위. seed 가 주어지면 **동점을 그 시드의 무작위 순서로** 깹니다.

    np.argsort 는 안정 정렬이라 동점 — 특히 상수 예측기(naive·slearner)의 8,000개
    전원 동점 — 이 **데이터 원순서 그대로** 늘어섭니다. 그래서 상수 모형이
    Qini ≈ 0 이 아니라 원순서의 우연에 따라 큰 음수를 받았고(실데이터 slearner
    −65.65 가 시드 5개 내내 ±0.0000), 논문 IX-A 의 「상수는 무작위보다 나쁘다」는
    서술이 이 인공물 위에 서 있었습니다. 동점이 없는 모형은 순위가 안 바뀝니다."""
    t = np.asarray(tau_hat).ravel()
    if seed is None:
        return np.argsort(-t)
    jit = np.random.default_rng(seed).permutation(len(t))
    return np.lexsort((jit, -t))


def uplift_curve(tau_hat, treated, y, n_bins=10, seed=None):
    """τ̂ 내림차순으로 정렬해 상위 k% 에서의 (처치군평균 − 대조군평균) × 인원."""
    o = _uplift_order(tau_hat, seed)
    t = np.asarray(treated).ravel()[o] > 0
    yy = np.asarray(y).ravel()[o]
    n = len(o)
    xs, ys = [], []
    for k in range(1, n_bins + 1):
        m = int(n * k / n_bins)
        tt, cc = t[:m], ~t[:m]
        if tt.sum() == 0 or cc.sum() == 0:
            xs.append(k / n_bins); ys.append(np.nan); continue
        xs.append(k / n_bins)
        ys.append((np.nanmean(yy[:m][tt]) - np.nanmean(yy[:m][cc])) * m)
    return np.array(xs), np.array(ys)


def auuc(tau_hat, treated, y, n_bins=10, tie_seeds=(0, 1, 2, 3, 4)):
    """[고침 F] 동점 무작위(시드 고정) 5회 평균 — 동점 없는 모형은 값이 안 변합니다."""
    vals = []
    for _sd in tie_seeds:
        x, u = uplift_curve(tau_hat, treated, y, n_bins, seed=_sd)
        ok = np.isfinite(u)
        if ok.sum() > 1:
            vals.append(float(np.trapezoid(u[ok], x[ok])))
    return float(np.mean(vals)) if vals else np.nan


def qini(tau_hat, treated, y, n_bins=10, tie_seeds=(0, 1, 2, 3, 4)):
    """Qini = AUUC − 무작위 배정선 아래 면적. [고침 F] 동점 무작위 처리 포함.
    전원 동점(상수 예측기)이면 정의상 0 을 돌려줍니다 — 정보 없는 순위입니다."""
    t = np.asarray(tau_hat).ravel()
    if np.nanstd(t) < 1e-12:
        return 0.0
    vals = []
    for _sd in tie_seeds:
        x, u = uplift_curve(tau_hat, treated, y, n_bins, seed=_sd)
        ok = np.isfinite(u)
        if ok.sum() <= 1 or not np.isfinite(u[-1]):
            continue
        vals.append(float(np.trapezoid(u[ok], x[ok])) - float(u[-1]) * 0.5)
    return float(np.mean(vals)) if vals else np.nan


ATT_JOBS = 0.07794018617548037


def score(name, tau_hat, std, outc, h=1, y='Y3', split=2, truth=None):
    """데이터셋에 맞는 지표만 냅니다. 못 내는 것은 아예 안 넣습니다."""
    m = (std.SPLIT == split)
    out = {'데이터': name, '분할': {0: 'train', 1: 'valid', 2: 'test'}[split],
           'n': int(m.sum()), 'τ 평균': round(float(np.nanmean(tau_hat[m])), 4),
           'τ sd': round(float(np.nanstd(tau_hat[m])), 4)}
    if name == 'ihdp':
        out['√PEHE'] = round(pehe(tau_hat, std.MU0[:, 0, 0], std.MU1[:, 0, 0], m), 4)
        out['ε_ATE'] = round(eps_ate(tau_hat, std.MU0[:, 0, 0], std.MU1[:, 0, 0], m), 4)
    elif name == 'jobs':
        out['ε_ATT'] = round(eps_att(tau_hat, outc['TREATED'], std.RCT[:, 0, 0],
                                     std.Y[:, 0, 0], ATT_JOBS), 4)
        out['정책위험'] = round(policy_risk(tau_hat, outc['TREATED'], std.RCT[:, 0, 0],
                                          std.Y[:, 0, 0]), 4)
    elif name == 'empservicesyntv2' and truth is not None:
        # 참값은 **훈련받은 사람에게만** 있습니다(생성기가 그때 t* 를 저장).
        # 대조군은 학습 표본에는 넣되 채점에서는 뺍니다 → ATT 성격의 PEHE 입니다.
        a = truth['AOBS']
        mu1 = truth['MUA3'][np.arange(std.N), a, h]
        mu0 = truth['MUA3'][:, 0, h]
        mm = (m & np.asarray(outc['TREATED'], bool)
              & np.isfinite(mu1) & np.isfinite(mu0))
        out['채점표본'] = '훈련군만(참값 존재) %d명' % int(mm.sum())
        out['√PEHE'] = round(pehe(tau_hat, mu0, mu1, mm), 4)
        out['ε_ATE'] = round(eps_ate(tau_hat, mu0, mu1, mm), 4)
    else:
        yy = _yh(outc, y, h)
        ok = m & _obs(outc, h) & np.isfinite(yy)
        out['AUUC'] = round(auuc(tau_hat[ok], outc['TREATED'][ok], yy[ok]), 2)
        out['Qini'] = round(qini(tau_hat[ok], outc['TREATED'][ok], yy[ok]), 2)
    return out
