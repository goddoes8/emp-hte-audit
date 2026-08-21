# -*- coding: utf-8 -*-
"""
기술분석(descriptive analysis) 도구.

표준화된 `datastd/*.npz` 와 원본 parquet 을 모두 훑어
변수별 분포·결측·특이값·종속변수와의 관계를 표로 만든다.
결과는 그대로 논문 Data 절과 부록 표로 쓴다.

원칙
  · 모든 통계는 **train 분할**에서 낸다. 시험지를 보고 쓴 글은 못 쓴다.
  · 결과 Y 와의 관계는 **M=1(유효 시점)** 에서만 계산한다.
  · 표를 화면에 다 뿌리지 않고 CSV 로도 저장한다.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

__all__ = ['raw_null_report', 'leakage_check', 'duplicate_columns', 'variable_profile', 'categorical_table',
           'treatment_balance', 'outcome_summary', 'time_profile',
           'outlier_report', 'correlation_with_y', 'run_all', 'save_tables']

Q = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


# ────────────────────────────────────────────── 원본 결측

def raw_null_report(path, max_rows=25):
    """표준화 **이전** 원본 파일의 결측·자료형. 표준화 후에는 결측이 0이므로 원본을 본다."""
    df = pd.read_parquet(path) if str(path).endswith('.parquet') else pd.read_csv(path)
    rows = []
    for c in df.columns:
        s = df[c]
        rows.append({
            '변수': c,
            '자료형': str(s.dtype),
            '결측수': int(s.isna().sum()),
            '결측률%': round(100 * s.isna().mean(), 4),
            '고유값수': int(s.nunique(dropna=True)),
            '최빈값': (s.mode(dropna=True).iloc[0] if s.notna().any() else None),
            '최빈비율%': round(100 * s.value_counts(normalize=True, dropna=True).iloc[0], 2)
                          if s.notna().any() else None,
        })
    out = pd.DataFrame(rows)
    out.attrs['n_rows'] = len(df)
    out.attrs['n_cols'] = df.shape[1]
    return out


# ────────────────────────────────────────────── 변수 프로파일

def _flat(std, split='train', mask_only=False):
    idx = std.idx(split)
    X = std.X[idx]
    if mask_only:
        m = std.M[idx].astype(bool)
        return X[m]
    return X.reshape(-1, std.Ds)


def variable_profile(std, split='train'):
    """상태 변수 한 칸씩: 분포·결측·0비율·왜도·첨도·특이값 비율."""
    cols = std.meta.get('state_cols') or [f'X{i}' for i in range(std.Ds)]
    F = _flat(std, split)
    n = F.shape[0]
    rows = []
    for j, c in enumerate(cols):
        v = F[:, j]
        finite = np.isfinite(v)
        vv = v[finite]
        mu, sd = vv.mean(), vv.std()
        q = np.quantile(vv, Q) if vv.size else [np.nan] * len(Q)
        z = (vv - mu) / (sd if sd > 1e-12 else 1.0)
        iq1, iq3 = np.quantile(vv, [.25, .75])
        iqr = iq3 - iq1
        rows.append({
            '변수': c,
            '고유값수': int(np.unique(vv).size),
            '결측수': int((~finite).sum()),
            '0비율%': round(100 * float((vv == 0).mean()), 2),
            '평균': mu, '표준편차': sd, '최소': vv.min(), '최대': vv.max(),
            'p1': q[0], 'p5': q[1], 'p25': q[2], '중위': q[3], 'p75': q[4], 'p95': q[5], 'p99': q[6],
            '왜도': float(((vv - mu) ** 3).mean() / (sd ** 3 + 1e-12)),
            '첨도': float(((vv - mu) ** 4).mean() / (sd ** 4 + 1e-12) - 3),
            '특이값%(|z|>3)': round(100 * float((np.abs(z) > 3).mean()), 2),
            '특이값%(IQR)': round(100 * float(((vv < iq1 - 1.5 * iqr) | (vv > iq3 + 1.5 * iqr)).mean()), 2),
            '최대|z|': round(float(np.abs(z).max()), 1),
        })
    out = pd.DataFrame(rows)
    out.attrs['n_obs'] = n
    return out


def outlier_report(std, split='train', top=15):
    """특이값이 많은 변수 상위 목록. 학습 불안정의 원인을 짚는 데 쓴다."""
    p = variable_profile(std, split)
    return (p[['변수', '0비율%', '왜도', '첨도', '특이값%(|z|>3)', '특이값%(IQR)', '최대|z|']]
            .sort_values('최대|z|', ascending=False).head(top).reset_index(drop=True))


# ────────────────────────────────────────────── 종속변수와의 관계

def correlation_with_y(std, split='train', top=20):
    """결과 Y 와의 피어슨·스피어만 상관 (M=1 시점만)."""
    cols = std.meta.get('state_cols') or [f'X{i}' for i in range(std.Ds)]
    idx = std.idx(split)
    m = std.M[idx].astype(bool)
    X = std.X[idx][m]
    y = std.Y[idx][..., 0][m]
    ry = pd.Series(y).rank().to_numpy()
    rows = []
    for j, c in enumerate(cols):
        v = X[:, j]
        if v.std() < 1e-12:
            rows.append({'변수': c, '피어슨r': np.nan, '스피어만r': np.nan}); continue
        rows.append({
            '변수': c,
            '피어슨r': float(np.corrcoef(v, y)[0, 1]),
            '스피어만r': float(np.corrcoef(pd.Series(v).rank().to_numpy(), ry)[0, 1]),
        })
    out = pd.DataFrame(rows)
    out['|피어슨r|'] = out['피어슨r'].abs()
    return out.sort_values('|피어슨r|', ascending=False).head(top).reset_index(drop=True)


def categorical_table(std, col, split='train', max_cat=30):
    """
    범주형(고유값이 적은) 변수 한 개: 항목별 빈도 · 처치율 · 결과 평균.
    '이 항목에 속한 사람은 훈련을 얼마나 받고, 월급이 얼마인가' 를 한 표로 본다.
    """
    cols = std.meta.get('state_cols') or [f'X{i}' for i in range(std.Ds)]
    j = cols.index(col)
    idx = std.idx(split)
    m = std.M[idx].astype(bool)
    v = std.X[idx][m][:, j]
    y = std.Y[idx][..., 0][m]
    a = std.A[idx][..., 0][m]
    u = np.unique(v)
    if u.size > max_cat:
        raise ValueError(f'{col} 은 고유값 {u.size}개로 범주형이 아닙니다.')
    vocab = (std.meta.get('state_vocabs') or {}).get(col)
    rows = []
    gm = y.mean()
    for k in u:
        s = v == k
        lab = vocab[int(k)] if (vocab and 0 <= int(k) < len(vocab)) else k
        rows.append({
            '항목': lab, '코드': k, '건수': int(s.sum()),
            '비율%': round(100 * float(s.mean()), 2),
            '처치율%': round(100 * float(a[s].mean()), 2),
            'Y평균': round(float(y[s].mean()), 4),
            'Y표준편차': round(float(y[s].std()), 4),
            'Y평균-전체평균': round(float(y[s].mean() - gm), 4),
        })
    return pd.DataFrame(rows).sort_values('건수', ascending=False).reset_index(drop=True)


# ────────────────────────────────────────────── 처치 균형

def treatment_balance(std, split='train', treat='A', top=None):
    """
    처치군과 대조군의 변수별 차이(표준화 평균차 SMD).
    |SMD| > 0.1 이면 두 집단이 원래 달랐다는 뜻 → 선택편향의 증거.
    """
    cols = std.meta.get('state_cols') or [f'X{i}' for i in range(std.Ds)]
    idx = std.idx(split)
    m = std.M[idx].astype(bool)
    X = std.X[idx][m]
    a = getattr(std, treat)[idx][..., 0][m]
    t, c = a == 1, a == 0
    rows = []
    for j, col in enumerate(cols):
        v = X[:, j]
        mt, mc = v[t].mean(), v[c].mean()
        sp = np.sqrt((v[t].var() + v[c].var()) / 2)
        rows.append({'변수': col, '처치군평균': mt, '대조군평균': mc,
                     'SMD': float((mt - mc) / (sp + 1e-9))})
    out = pd.DataFrame(rows)
    out['|SMD|'] = out['SMD'].abs()
    out = out.sort_values('|SMD|', ascending=False).reset_index(drop=True)
    out.attrs['n_treat'] = int(t.sum()); out.attrs['n_ctrl'] = int(c.sum())
    out.attrs['mean_abs_smd'] = round(float(out['|SMD|'].mean()), 4)
    out.attrs['n_imbalanced'] = int((out['|SMD|'] > 0.1).sum())
    return out if top is None else out.head(top)


def leakage_check(std, split='train', smd_thr=0.8, auc_thr=0.9):
    """
    처치 이후에 생기는 변수(post-treatment variable)가 상태에 섞였는지 검사.

    훈련을 받아야만 값이 생기는 변수를 상태 X 에 넣으면, 모델이 X 만 보고
    처치 여부를 알아버린다. 그러면 반사실 추정이 무의미해진다.
    (예: '훈련직종 효과잔존량' 은 훈련을 받아야 0 이 아니게 된다.)

    판정: |SMD| 가 크거나, 그 변수 하나로 처치를 거의 완벽히 맞히면 의심.
    """
    cols = std.meta.get('state_cols') or [f'X{i}' for i in range(std.Ds)]
    idx = std.idx(split); m = std.M[idx].astype(bool)
    X = std.X[idx][m]; a = std.A[idx][..., 0][m]
    t, c = a == 1, a == 0
    ry = pd.Series(a).rank().to_numpy()
    rows = []
    for j, col in enumerate(cols):
        v = X[:, j]
        if v.std() < 1e-12:
            continue
        sp = np.sqrt((v[t].var() + v[c].var()) / 2)
        smd = float((v[t].mean() - v[c].mean()) / (sp + 1e-9))
        # 단변량 AUC = 순위상관에서 환산
        r = float(np.corrcoef(pd.Series(v).rank().to_numpy(), ry)[0, 1])
        n1, n0 = int(t.sum()), int(c.sum())
        auc = 0.5 + r * np.sqrt(3 * n1 * n0 / ((n1 + n0) * (n1 + n0 + 1))) / np.sqrt(n1 * n0) * np.sqrt(n1 * n0)
        auc = float(np.clip(0.5 + r / 2, 0, 1))          # 근사
        rows.append({'변수': col, 'SMD': round(smd, 3), '|SMD|': round(abs(smd), 3),
                     '단변량AUC': round(max(auc, 1 - auc), 3),
                     '대조군 0비율%': round(100 * float((v[c] == 0).mean()), 2),
                     '처치군 0비율%': round(100 * float((v[t] == 0).mean()), 2)})
    out = pd.DataFrame(rows)
    out['누출의심'] = (out['|SMD|'] >= smd_thr) | (out['단변량AUC'] >= auc_thr)
    return out.sort_values('|SMD|', ascending=False).reset_index(drop=True)


def duplicate_columns(std, split='train', thr=0.999):
    """서로 상관이 1에 가까운 중복 변수 쌍. 같은 정보를 두 번 넣는 셈이다."""
    cols = std.meta.get('state_cols') or [f'X{i}' for i in range(std.Ds)]
    F = _flat(std, split)
    keep = F.std(0) > 1e-12
    idxs = np.flatnonzero(keep)
    C = np.corrcoef(F[:, keep].T)
    pairs = []
    for i in range(len(idxs)):
        for j in range(i + 1, len(idxs)):
            if abs(C[i, j]) >= thr:
                pairs.append({'변수A': cols[idxs[i]], '변수B': cols[idxs[j]],
                              '상관': round(float(C[i, j]), 5)})
    return pd.DataFrame(pairs)


# ────────────────────────────────────────────── 결과 변수

def outcome_summary(std):
    """결과 Y 를 분할별·처치별로 요약."""
    rows = []
    for sp in ('train', 'valid', 'test'):
        idx = std.idx(sp); m = std.M[idx].astype(bool)
        y = std.Y[idx][..., 0][m]; a = std.A[idx][..., 0][m]
        for nm, s in (('전체', np.ones_like(a, bool)), ('처치군', a == 1), ('대조군', a == 0)):
            if s.sum() == 0:
                continue
            yy = y[s]
            rows.append({'분할': sp, '집단': nm, '건수': int(s.sum()),
                         '평균': round(float(yy.mean()), 4), '표준편차': round(float(yy.std()), 4),
                         '최소': round(float(yy.min()), 3), 'p25': round(float(np.quantile(yy, .25)), 3),
                         '중위': round(float(np.median(yy)), 3), 'p75': round(float(np.quantile(yy, .75)), 3),
                         '최대': round(float(yy.max()), 3)})
    out = pd.DataFrame(rows)
    # 단순 차이 (인과효과 아님 — 선택편향 포함)
    tr = out[(out['분할'] == 'train')].set_index('집단')
    if {'처치군', '대조군'} <= set(tr.index):
        out.attrs['naive_diff'] = round(float(tr.loc['처치군', '평균'] - tr.loc['대조군', '평균']), 4)
    return out


def time_profile(std, split='train'):
    """시점별 패턴 — 취업률(M), 처치율(A), 훈련 시작(AEVT), 결과 평균."""
    if std.T <= 1:
        return pd.DataFrame()
    idx = std.idx(split)
    M = std.M[idx].astype(bool); A = std.A[idx][..., 0]
    AE = std.AEVT[idx][..., 0]; Y = std.Y[idx][..., 0]
    rows = []
    for t in range(std.T):
        m = M[:, t]
        rows.append({'t': t, '취업률%': round(100 * float(m.mean()), 2),
                     '처치율%': round(100 * float(A[:, t].mean()), 2),
                     '훈련시작%': round(100 * float(AE[:, t].mean()), 3),
                     'Y평균': round(float(Y[m, t].mean()), 4) if m.any() else np.nan,
                     '유효건수': int(m.sum())})
    return pd.DataFrame(rows)


# ────────────────────────────────────────────── 일괄 실행

def run_all(std, split='train', cat_max=12):
    """한 데이터셋에 대한 기술분석 묶음."""
    cols = std.meta.get('state_cols') or [f'X{i}' for i in range(std.Ds)]
    prof = variable_profile(std, split)
    cat_cols = [c for c, u in zip(cols, prof['고유값수']) if u <= cat_max]
    res = {
        'profile': prof,
        'leakage': leakage_check(std, split),
        'duplicates': duplicate_columns(std, split),
        'outliers': outlier_report(std, split),
        'corr_y': correlation_with_y(std, split),
        'balance': treatment_balance(std, split),
        'outcome': outcome_summary(std),
        'time': time_profile(std, split),
        'categorical': {c: categorical_table(std, c, split) for c in cat_cols},
        'cat_cols': cat_cols,
    }
    return res


def save_tables(res, outdir, prefix):
    os.makedirs(outdir, exist_ok=True)
    saved = []
    for k in ('profile', 'outliers', 'corr_y', 'balance', 'outcome', 'time',
              'leakage', 'duplicates'):
        df = res.get(k)
        if df is None or (hasattr(df, 'empty') and df.empty):
            continue
        p = os.path.join(outdir, f'{prefix}_{k}.csv')
        df.to_csv(p, index=False, encoding='utf-8-sig')
        saved.append(p)
    if res.get('categorical'):
        p = os.path.join(outdir, f'{prefix}_categorical.csv')
        pd.concat([d.assign(변수=c) for c, d in res['categorical'].items()],
                  ignore_index=True).to_csv(p, index=False, encoding='utf-8-sig')
        saved.append(p)
    return saved
