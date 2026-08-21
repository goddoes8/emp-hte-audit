# -*- coding: utf-8 -*-
"""
데이터 표준화 계층.

모든 데이터셋(IHDP, Jobs, EmpServiceSynt, EmpServiceReal)을 **하나의 스키마**로 변환해
`datastd/` 에 `.npz` 로 저장하고, 실험 코드는 오직 이 표준 파일만 읽는다.

────────────────────────────────────────────────────────────────────────
표준 스키마  (N=궤적 수, T=시점 수, Ds=상태 차원)
────────────────────────────────────────────────────────────────────────
  X       (N, T, Ds) float32   상태 (전처리 적용됨 — meta 참조)
  A       (N, T, 1)  float32   ★ 흡수 처치 1{t ≥ 첫 훈련시점}. CATE = f(·,A=1) − f(·,A=0)
  AEVT    (N, T, 1)  float32   사건 처치 1{그 시점에 훈련 시작}
  AIDX    (N, T, 1)  int32     훈련직종 코드. 없으면 -1
  APERSON (N, T, 1)  float32   개인 단위 처치 플래그(istreated). 참고용
  Y      (N, T, 1)   float32   결과
  M      (N, T)      uint8     유효 마스크 (패딩·손실계산 제외 시점은 0)
  TS     (N, T)      int32     타임스텝 (0..T-1)
  MU0    (N, T, 1)   float32   반사실 정답 E[Y|do(A=0)]. 없으면 NaN
  MU1    (N, T, 1)   float32   반사실 정답 E[Y|do(A=1)]. 없으면 NaN
  RCT    (N, T, 1)   float32   무작위배정 표본 지시자 (Jobs 전용). 없으면 NaN
  PS     (N, T, 1)   float32   성향점수. 없으면 NaN
  SPLIT  (N,)        uint8     0=train 1=valid 2=test  ← 파일에 고정
  meta   ()          str(json) 메타데이터

설계 원칙
  1. **분할을 파일에 박는다.** 실행마다 분할이 달라지지 않으므로,
     테스트셋으로 조기종료하는 사고가 구조적으로 불가능해진다.
  2. **정규화는 저장하지 않는다.** 원값을 저장하고, 학습 시 train 분할에서만
     통계를 계산해 적용한다.
  3. **범주형은 고정 어휘로 인코딩한다.** `pd.factorize` 는 등장 순서에 따라
     코드가 달라지므로 쓰지 않는다. 어휘는 meta 에 저장해 재현성을 보장한다.
  4. **정답이 없으면 NaN 을 넣는다.** 없는 것을 만들어내지 않는다.
  5. **처치를 세 가지로 나눠 저장한다.** `istreated` 는 개인 단위 상수(120시점 내내 동일)
     이므로 시점별 처치가 아니다. 실제 시점별 처치는 `정책_훈련직종 > 0` 사건이다.
     A(흡수) / AEVT(사건) / APERSON(개인) 을 모두 저장해 단계 2에서 비교한다.
  6. **상태 전처리는 train 분할 통계로만 한다.** 죽은 열 제거 기준과 클리핑 경계를
     train 에서 계산하므로 누수가 없다. 수행 내역은 meta 에 남긴다.
"""
from __future__ import annotations

import os, json, hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import (RAW, DIR_STD, SPLIT_SEED, VAL_FRAC, SPLIT_RATIO, STATE_LAG,
                     TRAJ_LEN, SPLIT_NAMES, ensure_dirs)

SCHEMA_VERSION = 3
ARRAY_KEYS = ('X', 'A', 'AEVT', 'AIDX', 'APERSON', 'Y', 'M', 'TS',
              'MU0', 'MU1', 'RCT', 'PS', 'SPLIT')

__all__ = ['StdData', 'build', 'save_std', 'load_std', 'verify_std',
           'list_std', 'describe', 'StdSeqDataset', 'DATASETS']

DATASETS = {
    'ihdp':           dict(n_realization=100, seq=False),
    'jobs':           dict(n_realization=10,  seq=False),
    'empservicesynt': dict(n_realization=1,   seq=True),
    'empservicereal': dict(n_realization=1,   seq=True),
    'empservicesyntv2': dict(n_realization=1, seq=True),   # v2  이질성·교란 검증용
    'empservicesyntsyn7': dict(n_realization=1, seq=True), # syn7 규칙01 (tools/gen_syn_keis.py)
    'empservicesyntsyn8': dict(n_realization=1, seq=True), # syn8 규칙02 (tools/gen_syn_keis2.py)
}


# ══════════════════════════════════════════════════════════ 컨테이너

@dataclass
class StdData:
    X: np.ndarray; A: np.ndarray; AEVT: np.ndarray; AIDX: np.ndarray
    APERSON: np.ndarray; Y: np.ndarray
    M: np.ndarray; TS: np.ndarray
    MU0: np.ndarray; MU1: np.ndarray; RCT: np.ndarray; PS: np.ndarray
    SPLIT: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def N(self): return self.X.shape[0]
    @property
    def T(self): return self.X.shape[1]
    @property
    def Ds(self): return self.X.shape[2]
    @property
    def has_mu(self): return bool(np.isfinite(self.MU0).any())
    @property
    def has_rct(self): return bool(np.isfinite(self.RCT).any())

    def idx(self, split):
        code = {'train': 0, 'valid': 1, 'test': 2}[split] if isinstance(split, str) else split
        return np.flatnonzero(self.SPLIT == code)

    def summary(self):
        rows = []
        for c, nm in SPLIT_NAMES.items():
            i = self.idx(c)
            if len(i) == 0:
                rows.append({'split': nm, 'N': 0}); continue
            m = self.M[i].astype(bool)
            rows.append({
                'split': nm, 'N': len(i),
                '유효시점': int(m.sum()),
                '처치율': round(float(self.A[i][..., 0][m].mean()), 4),
                'Y평균': round(float(self.Y[i][..., 0][m].mean()), 4),
                'Y표준편차': round(float(self.Y[i][..., 0][m].std()), 4),
            })
        return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════ 유틸

def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()[:16]


def _nan(shape):
    return np.full(shape, np.nan, dtype=np.float32)


def _split_by_unit(n, ratio, seed):
    """개인(궤적) 단위 무작위 분할. 반환: (n,) uint8"""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tr = int(round(n * ratio[0]))
    n_va = int(round(n * ratio[1]))
    s = np.full(n, 2, dtype=np.uint8)
    s[perm[:n_tr]] = 0
    s[perm[n_tr:n_tr + n_va]] = 1
    return s


def encode_fixed(series, vocab=None):
    """
    고정 어휘 범주 인코딩. `pd.factorize` 와 달리 등장 순서에 의존하지 않는다.
    'c00' 처럼 없음/기준을 뜻하는 값이 있으면 항상 0 을 받도록 맨 앞에 둔다.
    """
    if vocab is None:
        vals = sorted(pd.unique(series.astype(str)))
        base = [v for v in vals if v in ('c00', 'c0', '0', 'none', 'None')]
        rest = [v for v in vals if v not in base]
        vocab = base + rest
    lut = {v: i for i, v in enumerate(vocab)}
    codes = series.astype(str).map(lut)
    if codes.isna().any():
        missing = sorted(set(series.astype(str)[codes.isna()]))
        raise ValueError(f'어휘에 없는 값: {missing[:10]}')
    return codes.to_numpy(np.int64), vocab


def preprocess_states(X, SPLIT, cols, drop_dead=True, tame=True,
                      skew_thr=5.0, q=(0.001, 0.999), min_nonzero=0.005,
                      copy_in=True):
    """
    상태 전처리. 기준은 **train 분할에서만** 계산한다(누수 방지).

    1) 죽은 열 제거 — train 전 시점에서 분산 0 인 열
    2) 초희소 열 제거 — train 에서 0 이 아닌 비율이 min_nonzero 미만인 열.
       (예: 직종 c12 경력 보유자가 표본에 27명뿐이면 표준화 후 |z|>100 이 되어
        Linear 층의 그래디언트를 튀게 만든다. 정보량 대비 해가 크다.)
    3) 극단 희소열 완화 — |왜도| > skew_thr 인 열에 부호보존 log1p 후
       train 분위수로 클리핑.

    반환: (X_new, cols_new, report)
    """
    X = np.asarray(X, dtype=np.float32)
    if not copy_in:
        pass
    else:
        X = X.copy()
    cols = list(cols)
    tr = np.flatnonzero(SPLIT == 0)
    D = X.shape[2]
    rep = {'n_in': D}

    def _col(j):                       # train 분할의 j 번째 열만 1차원으로 (열 하나 = 수 MB)
        return X[tr, :, j].reshape(-1).astype(np.float64)   # 누적오차 방지

    # 1) 죽은 열 · 2) 초희소 열 — 마스크를 먼저 다 만들고 배열 복사는 딱 한 번만 한다
    rep['dropped'], rep['dropped_sparse'] = [], []
    keep = np.ones(D, bool)
    if drop_dead:
        for j in range(D):
            if _col(j).std() < 1e-8:
                keep[j] = False
        rep['dropped'] = [c for c, k in zip(cols, keep) if not k]
    if min_nonzero:
        for j in range(D):
            if not keep[j]:
                continue
            v = float((_col(j) != 0).mean())
            if v < min_nonzero:
                keep[j] = False
                rep['dropped_sparse'].append((cols[j], round(v, 5)))
    if not keep.all():
        X = X[:, :, keep]
        cols = [c for c, k in zip(cols, keep) if k]
        D = X.shape[2]

    # 3) 극단 왜도 열만 부호보존 log1p → train 분위수로 클리핑 (전부 제자리 연산)
    rep['logged'], rep['clipped'] = [], {}
    if tame and D:
        for j in range(D):
            f = _col(j)
            mu, sd = f.mean(), f.std() + 1e-12
            if abs((((f - mu) / sd) ** 3).mean()) > skew_thr:
                sub = X[:, :, j]
                np.copyto(sub, np.sign(sub) * np.log1p(np.abs(sub)))
                rep['logged'].append(cols[j])
        n_clip = 0
        for j in range(D):
            f = _col(j)
            lo, hi = np.quantile(f, q[0]), np.quantile(f, q[1])
            if hi > lo:
                np.clip(X[:, :, j], lo, hi, out=X[:, :, j])
                n_clip += 1
        rep['clipped'] = {'q': list(q), 'n_cols': int(n_clip)}

    zmax = 0.0
    for j in range(D):
        f = _col(j)
        mu, sd = f.mean(), f.std()
        sd = 1.0 if sd < 1e-8 else sd
        zmax = max(zmax, float(np.abs((f - mu) / sd).max()))
    rep.update(n_out=D, n_dropped=len(rep['dropped']),
               n_dropped_sparse=len(rep['dropped_sparse']),
               min_nonzero=min_nonzero, n_logged=len(rep['logged']),
               zmax_after=round(zmax, 2))
    return X, cols, rep


# ══════════════════════════════════════════════════════════ 빌더 — IHDP

def _build_ihdp(realization=0, use_1000=False):
    ftr = RAW['ihdp1000_train' if use_1000 else 'ihdp_train']
    fte = RAW['ihdp1000_test' if use_1000 else 'ihdp_test']
    parts, splits = [], []

    for path, is_train in ((ftr, True), (fte, False)):
        raw = np.load(path)
        r = realization % raw['x'].shape[2]
        x = raw['x'][:, :, r].astype(np.float32)
        t = raw['t'][:, r].astype(np.float32)
        y = raw['yf'][:, r].astype(np.float32)
        mu0 = raw['mu0'][:, r].astype(np.float32)
        mu1 = raw['mu1'][:, r].astype(np.float32)
        n = x.shape[0]
        parts.append((x, t, y, mu0, mu1))
        if is_train:                       # train 파일 안에서 검증 분리
            s = _split_by_unit(n, (1 - VAL_FRAC, VAL_FRAC, 0.0),
                               SPLIT_SEED + realization)
            s[s == 2] = 1                  # 잔여는 valid 로
        else:
            s = np.full(n, 2, dtype=np.uint8)
        splits.append(s)

    x = np.concatenate([p[0] for p in parts])[:, None, :]           # (N,1,25)
    t = np.concatenate([p[1] for p in parts])[:, None, None]
    y = np.concatenate([p[2] for p in parts])[:, None, None]
    mu0 = np.concatenate([p[3] for p in parts])[:, None, None]
    mu1 = np.concatenate([p[4] for p in parts])[:, None, None]
    N = x.shape[0]

    meta = dict(dataset='ihdp', realization=int(realization),
                source=[os.path.basename(ftr), os.path.basename(fte)],
                source_sha=[_sha256(ftr), _sha256(fte)],
                note='시퀀스 길이 1 (횡단면). 반사실 정답 mu0/mu1 존재.')
    return StdData(
        X=x, A=t, AEVT=t.copy(), AIDX=np.full((N, 1, 1), -1, np.int32),
        APERSON=t.copy(), Y=y,
        M=np.ones((N, 1), np.uint8), TS=np.zeros((N, 1), np.int32),
        MU0=mu0, MU1=mu1, RCT=_nan((N, 1, 1)), PS=_nan((N, 1, 1)),
        SPLIT=np.concatenate(splits), meta=meta)


# ══════════════════════════════════════════════════════════ 빌더 — Jobs

def _build_jobs(realization=0):
    """
    ⚠ 원 코드는 npz 의 'e' 를 True ITE 로 보고 mu0/mu1 을 역산했으나,
       Jobs 벤치마크의 'e' 는 **무작위배정(NSW 실험) 표본 지시자**(0/1)다.
       Jobs 에는 반사실 정답이 없다. 따라서 MU0/MU1 은 NaN 으로 두고
       'e' 는 RCT 로 저장한다. (PEHE 는 이 데이터에서 계산할 수 없다.)
    """
    parts, splits = [], []
    for key, is_train in (('jobs_train', True), ('jobs_test', False)):
        path = RAW[key]
        raw = np.load(path, allow_pickle=True, encoding='latin1')
        r = realization % raw['x'].shape[2]
        x = raw['x'][:, :, r].astype(np.float32)
        t = raw['t'][:, r].astype(np.float32)
        y = raw['yf'][:, r].astype(np.float32)
        e = raw['e'][:, r].astype(np.float32)
        n = x.shape[0]
        parts.append((x, t, y, e))
        if is_train:
            s = _split_by_unit(n, (1 - VAL_FRAC, VAL_FRAC, 0.0),
                               SPLIT_SEED + 1000 + realization)
            s[s == 2] = 1
        else:
            s = np.full(n, 2, dtype=np.uint8)
        splits.append(s)

    x = np.concatenate([p[0] for p in parts])[:, None, :]
    t = np.concatenate([p[1] for p in parts])[:, None, None]
    y = np.concatenate([p[2] for p in parts])[:, None, None]
    e = np.concatenate([p[3] for p in parts])[:, None, None]
    N = x.shape[0]

    meta = dict(dataset='jobs', realization=int(realization),
                source=[os.path.basename(RAW['jobs_train']), os.path.basename(RAW['jobs_test'])],
                source_sha=[_sha256(RAW['jobs_train']), _sha256(RAW['jobs_test'])],
                note=("시퀀스 길이 1. 반사실 정답 없음(MU0/MU1=NaN). "
                      "'e'는 무작위배정 표본 지시자이며 RCT 로 저장. "
                      "평가지표는 Policy Risk 와 ATT(RCT 부분표본) 만 유효."))
    return StdData(
        X=x, A=t, AEVT=t.copy(), AIDX=np.full((N, 1, 1), -1, np.int32),
        APERSON=t.copy(), Y=y,
        M=np.ones((N, 1), np.uint8), TS=np.zeros((N, 1), np.int32),
        MU0=_nan((N, 1, 1)), MU1=_nan((N, 1, 1)), RCT=e, PS=_nan((N, 1, 1)),
        SPLIT=np.concatenate(splits), meta=meta)


# ══════════════════════════════════════════════════════════ 빌더 — EmpService

def _build_empservice(which, traj_len=TRAJ_LEN, preprocess=True):
    """
    which : 'empservicesynt' | 'empservicereal' | 'empservicesyntv2'

    처치 정의 (검증 결과 반영)
      · `istreated` 는 **개인 단위 상수**다(2,000명 전원이 120시점 내내 동일).
        "이 사람이 언젠가 훈련을 받는가" 이지 "이 시점에 처치받았는가" 가 아니다.
        → APERSON 으로 보관.
      · 실제 시점별 처치는 `정책_훈련직종 > 0` 사건이다.
        → AEVT (사건), A (흡수: 첫 사건 이후 계속 1).
      · 훈련은 M=0(실직·훈련 중) 시점에 일어나고 결과인 로그임금은 M=1(재취업)
        시점에 관측된다. 종단 인과 설계로 정합적인 구조다.
    """
    # [84] 실데이터는 코호트를 갈아끼울 수 있습니다(2,000명 ↔ 확대판).
    from . import config as _cf
    path = _cf.real_path() if which == 'empservicereal' else RAW[which]
    import pyarrow.parquet as _pq
    _names = list(_pq.ParquetFile(path).schema_arrow.names)
    _state = [c for c in _names if c.startswith('상태_')]

    # [84] **열 화이트리스트** — 코호트를 키울 때 변수 정의까지 바뀌면 안 됩니다.
    #   새 생성기는 상태열이 111개(상태_현재실직기간 추가)인데 기존 분석은 70개를 씁니다.
    #   목록을 주면 그 열만 그 순서대로 씁니다. 없는 열이 있으면 **이름을 찍고 멈춥니다** —
    #   조용히 다른 변수 조합으로 도는 것이 제일 위험합니다.
    _keep = getattr(_cf, 'STATE_KEEP', {}).get(which)
    if _keep:
        _miss = [c for c in _keep if c not in _state]
        if _miss:
            raise KeyError(
                '%s 에 없는 상태열이 %d개입니다: %s\n'
                '  파일 %s\n'
                '  STATE_KEEP 목록과 코호트가 안 맞습니다. 생성기를 확인하십시오.'
                % (which, len(_miss), _miss[:8], path))
        _drop = [c for c in _state if c not in set(_keep)]
        _state = list(_keep)
        print('  [열 맞춤] %s — 상태열 %d개만 사용 (%d개 제외: %s%s)'
              % (which, len(_state), len(_drop), ', '.join(_drop[:3]),
                 ' …' if len(_drop) > 3 else ''))
    _need = [c for c in _names if not c.startswith('상태_')]
    df = pd.read_parquet(path, columns=_need)      # 상태열은 아래에서 묶음으로 읽는다

    if len(df) % traj_len != 0:
        raise ValueError(f'행 수 {len(df):,} 가 traj_len {traj_len} 의 배수가 아닙니다.')
    N = len(df) // traj_len

    # ---- 블록 구조 검증: 각 120행 블록이 한 사람인지
    pv = df['persona'].to_numpy().reshape(N, traj_len)
    if not (pv[:, :1] == pv).all():
        raise ValueError('120행 블록 안에서 persona 가 일정하지 않습니다. 정렬을 확인하십시오.')

    # 상태열은 12개씩 나눠 읽어 미리 잡아 둔 float32 배열에 바로 채운다.
    # (한꺼번에 읽으면 960,000행 × 116열 DataFrame 이 통째로 메모리에 올라가
    #  8GB 노트북에서 스와핑이 일어난다.)
    state_cols = list(_state)
    vocabs = {}
    X = np.empty((len(df), len(state_cols)), np.float32)
    CH = 12
    for i0 in range(0, len(state_cols), CH):
        chunk = state_cols[i0:i0 + CH]
        dch = pd.read_parquet(path, columns=chunk)
        for k, c in enumerate(chunk):
            col = dch[c]
            if (col.dtype == object or str(col.dtype) in ('string', 'str')
                    or str(col.dtype).startswith('large_string')):
                code, vocab = encode_fixed(col)
                vocabs[c] = vocab
                X[:, i0 + k] = code
            else:
                X[:, i0 + k] = pd.to_numeric(col, errors='coerce').to_numpy(np.float32)
        del dch
    X = X.reshape(N, traj_len, len(state_cols))

    n_nan = int(np.isnan(X).sum())
    if n_nan:
        raise ValueError(f'상태에 결측 {n_nan}개. 원자료를 확인하십시오.')

    APERSON = df['istreated'].to_numpy(np.float32).reshape(N, traj_len, 1)
    aidx, act_vocab = encode_fixed(df['정책_훈련직종'])
    AIDX = aidx.astype(np.int32).reshape(N, traj_len, 1)
    AEVT = (AIDX > 0).astype(np.float32)                       # 훈련 시작 사건
    A = (np.cumsum(AEVT[..., 0], axis=1) > 0).astype(np.float32)[..., None]  # 흡수 처치
    Y = pd.to_numeric(df['보상_로그임금'], errors='coerce').to_numpy(np.float32).reshape(N, traj_len, 1)

    if '보상_손실함수계산여부' in df.columns:
        M = df['보상_손실함수계산여부'].to_numpy(np.uint8).reshape(N, traj_len)
    else:
        M = np.ones((N, traj_len), np.uint8)
    M = (M.astype(bool) & np.isfinite(Y[..., 0])).astype(np.uint8)
    Y = np.nan_to_num(Y, nan=0.0)

    if {'True_Mu0', 'True_Mu1'} <= set(df.columns):
        MU0 = df['True_Mu0'].to_numpy(np.float32).reshape(N, traj_len, 1)
        MU1 = df['True_Mu1'].to_numpy(np.float32).reshape(N, traj_len, 1)
    else:
        MU0 = _nan((N, traj_len, 1)); MU1 = _nan((N, traj_len, 1))

    TS = np.tile(np.arange(traj_len, dtype=np.int32), (N, 1))
    SPLIT = _split_by_unit(N, SPLIT_RATIO, SPLIT_SEED + (0 if 'synt' in which else 7))

    # ── 단계 2: 상태 1시점 지연 ──────────────────────────────────
    # 행 t 의 상태를 X_{t-1} 로 바꾼다. 행동 A_t 와 보상 Y_t 는 그대로 둔다.
    # 그래야 「그 달의 행동이 그 달의 상태에 이미 반영된」 누출이 사라진다.
    # 지연으로 정의가 사라지는 앞쪽 L 시점은 M=0 으로 막는다.
    lag = int(STATE_LAG.get(which, 0))
    if lag > 0:
        X[:, lag:, :] = X[:, :-lag, :]
        X[:, :lag, :] = X[:, lag:lag + 1, :]      # 앞쪽은 첫 유효 상태로 채움(M=0 이라 학습에 안 쓰임)
        M[:, :lag] = 0

    prep = {'applied': False}
    if preprocess:
        X, state_cols, prep = preprocess_states(X, SPLIT, state_cols, copy_in=False)
        prep['applied'] = True

    meta = dict(dataset=which, realization=0,
                source=[os.path.basename(path)], source_sha=[_sha256(path)],
                traj_len=traj_len, n_person=int(N),
                state_cols=state_cols, action_vocab=act_vocab, state_vocabs=vocabs,
                persona_kind=('type' if pd.unique(pv[:, 0]).size < N else 'individual'),
                preprocess=prep,
                state_lag=lag,
                n_treated_person=int((APERSON[:, 0, 0] > 0).sum()),
                n_event_person=int((AEVT.sum(1)[:, 0] > 0).sum()),
                note=(f'상태 {lag}시점 지연 적용(단계 2). ' if lag else '') +
                     ('A=흡수처치 1{t≥첫 훈련시점}, AEVT=사건, AIDX=훈련직종, '
                      'APERSON=istreated(개인 상수). M 은 보상_손실함수계산여부 반영. '
                      '상태는 train 기준 전처리 적용(meta.preprocess).'))
    return StdData(X=X, A=A, AEVT=AEVT, AIDX=AIDX, APERSON=APERSON, Y=Y, M=M, TS=TS,
                   MU0=MU0, MU1=MU1, RCT=_nan((N, traj_len, 1)), PS=_nan((N, traj_len, 1)),
                   SPLIT=SPLIT, meta=meta)


# ══════════════════════════════════════════════════════════ 공개 API

def build(name, realization=0, **kw):
    if name == 'ihdp':
        return _build_ihdp(realization, **kw)
    if name == 'jobs':
        return _build_jobs(realization)
    if name in ('empservicesynt', 'empservicereal', 'empservicesyntv2',
                'empservicesyntsyn7', 'empservicesyntsyn8'):
        return _build_empservice(name, **kw)
    raise ValueError(f'알 수 없는 데이터셋: {name}. 가능: {list(DATASETS)}')


def std_path(name, realization=0):
    """
    표준 npz 의 경로.

    [84] 실데이터는 코호트를 갈아끼울 수 있으므로 **파일 이름에 코호트를 넣습니다.**
    안 그러면 코호트를 바꿔도 `load_std` 가 옛 npz 를 그대로 읽어, 새 자료로 돌린 줄
    알고 옛 숫자를 보게 됩니다. 이름을 나누면 두 코호트가 나란히 살고 섞이지 않습니다.
    """
    sub = os.path.join(DIR_STD, name)
    n_rep = DATASETS[name]['n_realization']
    tag = ''
    if name == 'empservicereal':
        from . import config as _cf
        tag = ('_%s' % _cf.REAL_COHORT) if getattr(_cf, 'REAL_COHORT', None) else ''
    fn = f'{name}{tag}.npz' if n_rep == 1 else f'{name}{tag}_r{realization:03d}.npz'
    return os.path.join(sub, fn)


def save_std(std, name=None, realization=None):
    name = name or std.meta['dataset']
    realization = std.meta.get('realization', 0) if realization is None else realization
    path = std_path(name, realization)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta = dict(std.meta)
    meta.update(schema_version=SCHEMA_VERSION,
                shape=dict(N=std.N, T=std.T, Ds=std.Ds),
                split_counts={SPLIT_NAMES[c]: int((std.SPLIT == c).sum()) for c in (0, 1, 2)},
                has_mu=std.has_mu, has_rct=std.has_rct)
    np.savez_compressed(path, meta=json.dumps(meta, ensure_ascii=False),
                        **{k: getattr(std, k) for k in ARRAY_KEYS})
    return path


def std_meta(name, realization=0):
    """`.npz` 의 meta 만 읽는다(배열은 안 읽으므로 빠르다). 없으면 None."""
    path = std_path(name, realization)
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            return json.loads(str(z['meta']))
    except Exception:
        return None


def std_is_current(name, realization=0):
    """
    저장된 `.npz` 가 **지금 코드가 만들 결과와 같은 규격인지** 본다.
    스키마 판이 올라갔거나 상태 지연(STATE_LAG)이 바뀌었으면 False → 다시 만들어야 한다.
    (원자료는 그대로인데 코드만 바뀐 경우를 잡는 장치. 단계 2에서 실제로 겪은 사고다.)
    """
    m = std_meta(name, realization)
    if m is None:
        return False
    if int(m.get('schema_version', 0)) != int(SCHEMA_VERSION):
        return False
    if int(m.get('state_lag', 0)) != int(STATE_LAG.get(name, 0)):
        return False
    return True


def load_std(name, realization=0):
    path = std_path(name, realization)
    if not os.path.exists(path):
        raise FileNotFoundError(f'표준 파일 없음: {path}\n먼저 build → save_std 를 실행하십시오.')
    z = np.load(path, allow_pickle=False)
    arrs = {k: z[k] for k in ARRAY_KEYS}
    return StdData(meta=json.loads(str(z['meta'])), **arrs)


def verify_std(std, name=None):
    """저장물이 스키마를 지키는지 검사. 실패 목록을 반환(빈 리스트면 정상)."""
    err = []
    N, T, Ds = std.N, std.T, std.Ds
    want = {'X': (N, T, Ds), 'A': (N, T, 1), 'AEVT': (N, T, 1), 'AIDX': (N, T, 1),
            'APERSON': (N, T, 1), 'Y': (N, T, 1),
            'M': (N, T), 'TS': (N, T), 'MU0': (N, T, 1), 'MU1': (N, T, 1),
            'RCT': (N, T, 1), 'PS': (N, T, 1), 'SPLIT': (N,)}
    for k, sh in want.items():
        a = getattr(std, k)
        if tuple(a.shape) != sh:
            err.append(f'{k} 형상 {tuple(a.shape)} != {sh}')
    if np.isnan(std.X).any():           err.append('X 에 NaN')
    if np.isnan(std.A).any():           err.append('A 에 NaN')
    for k in ('A', 'AEVT', 'APERSON'):
        v = getattr(std, k)
        if not np.isin(v, (0.0, 1.0)).all():
            err.append(f'{k} 가 이진이 아님: {np.unique(v)[:6]}')
    absorb = (np.cumsum(std.AEVT[..., 0], axis=1) > 0).astype(np.float32)
    if not np.array_equal(absorb, std.A[..., 0]):
        err.append('A 가 AEVT 의 누적(흡수 처치)과 불일치')
    if std.M.sum() == 0:                err.append('유효 시점이 0')
    if not np.isin(std.SPLIT, (0, 1, 2)).all(): err.append('SPLIT 값 오류')
    for c, nm in SPLIT_NAMES.items():
        if (std.SPLIT == c).sum() == 0: err.append(f'{nm} 분할이 비어 있음')
    tr, va, te = (set(np.flatnonzero(std.SPLIT == c)) for c in (0, 1, 2))
    if tr & va or tr & te or va & te:   err.append('분할이 겹침')
    if std.has_mu:
        ite = (std.MU1 - std.MU0)[..., 0][std.M.astype(bool)]
        if not np.isfinite(ite).all():  err.append('MU 에 부분 결측')
    return err


def list_std():
    ensure_dirs()
    rows = []
    for name, spec in DATASETS.items():
        for r in range(spec['n_realization']):
            p = std_path(name, r)
            if os.path.exists(p):
                rows.append({'dataset': name, 'realization': r,
                             'MB': round(os.path.getsize(p) / 1e6, 2),
                             'path': os.path.relpath(p, DIR_STD)})
    return pd.DataFrame(rows)


def describe(std):
    m = std.meta
    lines = [f"[{m['dataset']}] realization={m.get('realization', 0)}  "
             f"N={std.N:,}  T={std.T}  Ds={std.Ds}",
             f"  반사실정답(MU) {'있음' if std.has_mu else '없음'} | "
             f"RCT지시자 {'있음' if std.has_rct else '없음'} | "
             f"훈련직종 {'있음' if (std.AIDX > 0).any() else '없음'}",
             f"  처치  A(흡수) {std.A.mean():.3f} | AEVT(사건) {std.AEVT.mean():.4f} | "
             f"APERSON(개인) {std.APERSON[:, 0, 0].mean():.3f}",]
    pp = m.get('preprocess') or {}
    if pp.get('applied'):
        lines += [f"  전처리 상태열 {pp['n_in']}→{pp['n_out']} "
                  f"(상수 제거 {pp['n_dropped']}, 초희소 제거 {pp.get('n_dropped_sparse', 0)}, "
                  f"log1p {pp['n_logged']}, |z|max {pp['zmax_after']})",]
    lines += [
             f"  {m.get('note', '')}"]
    return '\n'.join(lines) + '\n' + std.summary().to_string(index=False)


# ══════════════════════════════════════════════════════════ 학습용 Dataset

class StdSeqDataset:
    """
    표준 npz → 모델이 기대하는 배치 형태로 변환.
    배치 키: states, actions, rewards, rtg, timesteps, mask, mu_0, mu_1

    정규화 통계는 **train 분할에서만** 계산해 넘겨야 한다(누수 방지).
    """
    def __init__(self, std, split='train', context_len=None,
                 state_mean=None, state_std=None, gamma=1.0):
        self.std, self.split = std, split
        self.idx = std.idx(split)
        self.K = context_len or std.T
        self.gamma = gamma

        if state_mean is None:
            # 모델은 M=0 시점도 **입력**으로 소비하므로 통계는 전 시점에서 낸다.
            # (M=1 시점만 쓰면 그때 거의 상수인 열의 std 가 0에 가까워져 |z| 가 폭발한다.)
            tr = std.idx('train')
            flat = std.X[tr].reshape(-1, std.Ds)
            state_mean = flat.mean(0)
            state_std = flat.std(0)
            # 상수 열(std≈0)을 1e-6 으로 나누면 값이 폭발한다. 1.0 으로 두어 중심화만 한다.
            state_std = np.where(state_std < 1e-8, 1.0, state_std)
        self.state_mean = np.asarray(state_mean, np.float32)
        self.state_std = np.asarray(state_std, np.float32)
        self.const_cols = np.flatnonzero(self.state_std == 1.0) if state_std is not None else np.array([])

    def get_state_stats(self):
        return self.state_mean, self.state_std

    def dead_state_cols(self):
        """train 분할 전 시점에서 분산이 0인 상태 열."""
        tr = self.std.idx('train')
        sd = self.std.X[tr].reshape(-1, self.std.Ds).std(0)
        j = np.flatnonzero(sd < 1e-8)
        names = self.std.meta.get('state_cols')
        return [(int(i), names[i] if names else f'col{i}') for i in j]

    def __len__(self):
        return len(self.idx)

    def item_numpy(self, i):
        """torch 없이도 검증할 수 있도록 numpy 단계를 분리."""
        j = self.idx[i]
        s = (self.std.X[j] - self.state_mean) / self.state_std
        a, y, m = self.std.A[j], self.std.Y[j], self.std.M[j]
        ts = self.std.TS[j]
        K = min(self.K, self.std.T)
        s, a, y, m, ts = s[-K:], a[-K:], y[-K:], m[-K:], ts[-K:]

        rtg = np.zeros_like(y)                      # return-to-go (감가 누적)
        run = 0.0
        for t in range(len(y) - 1, -1, -1):
            run = float(y[t, 0]) * float(m[t]) + self.gamma * run
            rtg[t, 0] = run

        out = dict(
            states=np.ascontiguousarray(s, np.float32),
            actions=np.ascontiguousarray(a, np.float32),
            rewards=np.ascontiguousarray(y, np.float32),
            rtg=np.ascontiguousarray(rtg, np.float32),
            timesteps=np.ascontiguousarray(ts, np.int64),
            mask=np.ascontiguousarray(m, np.float32),
        )
        out['a_event'] = np.ascontiguousarray(self.std.AEVT[j][-K:], np.float32)
        out['a_person'] = np.ascontiguousarray(self.std.APERSON[j][-K:], np.float32)
        if self.std.has_mu:
            out['mu_0'] = np.ascontiguousarray(self.std.MU0[j][-K:], np.float32)
            out['mu_1'] = np.ascontiguousarray(self.std.MU1[j][-K:], np.float32)
        if self.std.has_rct:
            out['rct'] = np.ascontiguousarray(self.std.RCT[j][-K:], np.float32)
        return out

    def __getitem__(self, i):
        import torch
        return {k: torch.from_numpy(v) for k, v in self.item_numpy(i).items()}


# ══════════════════════════════════════════════════════════ v2 반사실 정답

TRUTH_ACTIONS = tuple(f'c{i:02d}' for i in range(37))
TRUTH_HORIZONS = (6, 12, 24)


def truth_path(name='empservicesyntv2', realization=0):
    return os.path.join(DIR_STD, name, f'{name}_truth.npz')


def build_truth(name='empservicesyntv2'):
    """datasynt/*_truth.parquet + *_person.parquet → (N, 37, 3) 배열.
    사람 순서는 표준 npz 의 사람 블록 순서(persona 오름차순)와 같습니다."""
    tr = pd.read_parquet(RAW[name + '_truth'])
    pe = pd.read_parquet(RAW[name + '_person'])
    persons = sorted(pe['persona'].unique())
    pidx = {p: i for i, p in enumerate(persons)}
    aidx = {a: i for i, a in enumerate(TRUTH_ACTIONS)}
    hidx = {h: i for i, h in enumerate(TRUTH_HORIZONS)}
    N = len(persons)
    out = {k: np.full((N, len(TRUTH_ACTIONS), len(TRUTH_HORIZONS)), np.nan, np.float32)
           for k in ('MUA1', 'MUA2', 'MUA3')}
    i = tr['persona'].map(pidx).to_numpy()
    j = tr['action'].map(aidx).to_numpy()
    k = tr['horizon'].map(hidx).to_numpy()
    out['MUA1'][i, j, k] = tr['MU_Y1'].to_numpy(np.float32)
    out['MUA2'][i, j, k] = tr['MU_Y2'].to_numpy(np.float32)
    out['MUA3'][i, j, k] = tr['MU_Y3'].to_numpy(np.float32)
    pe = pe.set_index('persona').loc[persons]
    out['TSTAR'] = pe['tstar'].to_numpy(np.int32)
    out['AOBS'] = np.array([aidx[a] for a in pe['훈련직종']], np.int32)
    out['GENDER'] = (pe['성별'] == '여').to_numpy(np.int8)
    out['AGE0'] = pe['나이'].to_numpy(np.int16)
    out['EDU'] = pe['학력'].to_numpy(np.int16)
    out['U'] = pe['U'].to_numpy(np.float32)
    out['V'] = pe['V'].to_numpy(np.float32)
    out['ACTIONS'] = np.array(TRUTH_ACTIONS)
    out['HORIZONS'] = np.array(TRUTH_HORIZONS, np.int32)
    return out


def save_truth(d, name='empservicesyntv2'):
    p = truth_path(name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez_compressed(p, **d)
    return p


def load_truth(name='empservicesyntv2'):
    z = np.load(truth_path(name), allow_pickle=False)
    return {k: z[k] for k in z.files}
