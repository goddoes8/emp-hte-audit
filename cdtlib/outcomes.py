# -*- coding: utf-8 -*-
"""
단계 3 — 결과변수 3종 × 관측창 3종.

    Y1 취업가능성   t*+h 시점에 취업해 있는가            (0/1)
    Y2 예상임금     t*+h 시점의 로그임금 | 취업          (취업 아니면 결측)
    Y3 기대소득     t*+h 시점의 취업 × 임금(만원)        (미취업이면 0)

**정의를 합성데이터의 참값과 한 글자도 다르지 않게 맞췄습니다.**
`tools/gen_empservicesynt.py` 의 반사실 참값은

    MU_Y1[i,a,h] = P(t*+h 에 취업)
    MU_Y2[i,a,h] = E[로그임금 | t*+h 에 취업]
    MU_Y3[i,a,h] = E[취업 × 임금(만원)]   ← t*+h 「시점」의 값이지 창 누적이 아님

이므로 관측 결과도 **시점값**으로 만듭니다. 창 안 아무 때나 취업했는가(any) 로 만들면
PEHE 를 참값과 견줄 수 없습니다.

처치 시점 t*
    훈련군 — 관측창 안 첫 훈련 시작 월(AEVT 가 처음 1인 달)
    대조군 — 훈련이 없으니 t* 가 없습니다. 훈련군 t* 의 경험분포에서 뽑아
             **위약 처치시점**을 부여합니다(시드 고정). 이렇게 해야 두 군이 같은
             달력시간 분포 위에서 비교됩니다(P9 시간 교란 대응).
    합성 v2 — 생성기가 사람마다 t* 를 정해 두었고 참값이 그 t* 기준이므로,
             정답 파일의 TSTAR 를 그대로 씁니다. 새로 뽑으면 참값과 어긋납니다.

우측 절단
    t*+h 가 관측창(T=120) 밖이면 그 사람의 그 관측창 결과는 **없습니다.**
    0 으로 채우지 않습니다. NaN 으로 두고 관측 가능 인원을 따로 셉니다.
"""
import numpy as np

HORIZONS = (6, 12, 24)

# 보상_로그임금을 만원으로 되돌리는 방법. 데이터마다 다릅니다.
#   실데이터·v1 — 03 이 만원에 log1p 를 씌웠습니다        → expm1
#   합성 v2·syn7·syn8 — 생성기가 로그임금을 직접 냅니다   → exp
#   (참값 MU_Y3 도 exp 로 계산되므로 여기서 맞춰야 채점이 성립합니다)
WAGE_INV = {
    'empservicereal':     'expm1',
    'empservicesynt':     'expm1',
    'empservicesyntv2':   'exp',
    'empservicesyntsyn7': 'exp',
    'empservicesyntsyn8': 'exp',
}


def wage_won(y, name):
    """로그임금 → 만원."""
    return np.expm1(y) if WAGE_INV.get(name, 'expm1') == 'expm1' else np.exp(y)


def treat_time(std, seed=20260821):
    """
    처치 시점 t* (N,) 과 처치 여부 (N,).

    훈련군은 첫 훈련 사건 월, 대조군은 훈련군 t* 분포에서 뽑은 위약 시점입니다.
    """
    evt = std.AEVT[..., 0] > 0
    treated = evt.any(1)
    tstar = np.where(treated, evt.argmax(1), -1).astype(np.int32)
    if treated.any() and (~treated).any():
        rng = np.random.default_rng(seed)
        tstar[~treated] = rng.choice(tstar[treated], size=int((~treated).sum()),
                                     replace=True)
    return tstar, treated


def build(std, name=None, horizons=HORIZONS, tstar=None, seed=20260821):
    """
    반환 dict
        Y1 (N,H) 0/1 · Y2 (N,H) 취업 아니면 NaN · Y3 (N,H) 만원
        OBS(N,H) True 면 관측 가능(우측절단 아님)
        TSTAR (N,) · TREATED (N,) · AIDX0 (N,) 관측된 훈련직종(대조군 0)
    """
    name = name or std.meta['dataset']
    T = std.T
    ts, treated = treat_time(std, seed)
    if tstar is not None:
        # 합성 v2 — 참값의 t* 를 그대로 씁니다. 새로 뽑으면 참값과 어긋납니다.
        # 다만 생성기는 **훈련받은 사람에게만** t* 를 저장해 나머지는 -1 입니다.
        # 그대로 두면 분석 표본이 전원 훈련군이 되어 「처치 vs 비처치」 대비가
        # 성립하지 않습니다(정적 모델의 τ 가 상수 0 이 되고 naive 는 NaN 이 됩니다).
        # 그래서 t* 가 없는 사람에게는 위약 시점을 부여해 **학습 표본에 넣습니다.**
        # 채점은 참값이 있는 훈련군에서만 합니다(models.score 가 그렇게 자릅니다).
        ts = np.asarray(tstar, np.int32).copy()
        _miss = ts < 0
        if _miss.any() and (~_miss).any():
            _rng = np.random.default_rng(seed)
            ts[_miss] = _rng.choice(ts[~_miss], size=int(_miss.sum()), replace=True)
        treated = np.asarray(tstar, np.int32) >= 0
    H = len(horizons)
    N = std.N

    emp = std.M.astype(bool)                   # 그 달에 취업(보상이 관측)했는가
    ylog = std.Y[..., 0]

    Y1 = np.full((N, H), np.nan, np.float32)
    Y2 = np.full((N, H), np.nan, np.float32)
    Y3 = np.full((N, H), np.nan, np.float32)
    OBS = np.zeros((N, H), bool)
    rows = np.arange(N)
    for k, h in enumerate(horizons):
        t = ts + h
        ok = (ts >= 0) & (t < T)               # 우측 절단
        OBS[:, k] = ok
        tt = np.clip(t, 0, T - 1)
        e = emp[rows, tt] & ok
        Y1[ok, k] = e[ok].astype(np.float32)
        w = ylog[rows, tt]
        Y2[np.where(ok & e)[0], k] = w[ok & e]
        Y3[ok, k] = np.where(e[ok], wage_won(w[ok], name), 0.0)

    aidx = std.AIDX[..., 0]
    a0 = np.where(ts >= 0, aidx[rows, np.clip(ts, 0, T - 1)], 0).astype(np.int32)
    a0 = np.where(treated, a0, 0)
    return dict(Y1=Y1, Y2=Y2, Y3=Y3, OBS=OBS, TSTAR=ts,
                TREATED=treated.astype(np.int8), AIDX0=a0,
                HORIZONS=np.asarray(horizons, np.int32))


def summary(out, horizons=HORIZONS):
    """관측 가능 인원·우측절단·평균을 표로."""
    import pandas as pd
    tr = out['TREATED'].astype(bool)
    rows = []
    for k, h in enumerate(horizons):
        ok = out['OBS'][:, k]
        rows.append({
            '관측창(개월)': h,
            '관측가능': int(ok.sum()),
            '우측절단': int((~ok).sum()),
            '훈련군': int((ok & tr).sum()),
            '대조군': int((ok & ~tr).sum()),
            'Y1 취업률': round(float(np.nanmean(out['Y1'][ok, k])), 4),
            'Y2 임금(취업)': round(float(np.nanmean(out['Y2'][ok, k])), 4),
            'Y2 결측률': round(float(np.isnan(out['Y2'][ok, k]).mean()), 4),
            'Y3 기대소득': round(float(np.nanmean(out['Y3'][ok, k])), 2),
        })
    return pd.DataFrame(rows)


def naive_diff(out, horizons=HORIZONS):
    """조정 없는 단순 차이(훈련군 − 대조군). 조정 전 하한을 보여 주는 값입니다."""
    import pandas as pd
    tr = out['TREATED'].astype(bool)
    rows = []
    for k, h in enumerate(horizons):
        ok = out['OBS'][:, k]
        r = {'관측창(개월)': h}
        for nm in ('Y1', 'Y2', 'Y3'):
            v = out[nm][:, k]
            a = np.nanmean(v[ok & tr]) if (ok & tr).any() else np.nan
            b = np.nanmean(v[ok & ~tr]) if (ok & ~tr).any() else np.nan
            r[nm + ' 훈련군'] = round(float(a), 4)
            r[nm + ' 대조군'] = round(float(b), 4)
            r[nm + ' 단순차이'] = round(float(a - b), 4)
        rows.append(r)
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════ Lee (2009) 경계
def lee_bounds(y_t, e_t, y_c, e_c):
    """
    임금은 **취업한 사람에게만** 관측됩니다. 훈련이 취업률을 바꾸면 두 군의
    「임금이 보이는 사람」 구성이 달라져, 단순 비교는 인과효과가 아닙니다.

    Lee(2009) 는 언제나 취업했을 사람(always-taker)에 대한 효과를 **구간**으로 줍니다.
    취업률이 높은 쪽에서 비율 q = (p_hi − p_lo)/p_hi 만큼을 잘라
    아래쪽을 자르면 상한, 위쪽을 자르면 하한이 됩니다.

    인수: y_t/y_c 는 취업자의 임금(결측 제외), e_t/e_c 는 각 군의 취업률.
    반환: (하한, 상한, 잘라낸 비율 q, 어느 쪽을 잘랐나)
    """
    y_t = np.asarray(y_t, float); y_t = y_t[np.isfinite(y_t)]
    y_c = np.asarray(y_c, float); y_c = y_c[np.isfinite(y_c)]
    if len(y_t) < 5 or len(y_c) < 5 or not np.isfinite(e_t) or not np.isfinite(e_c):
        return np.nan, np.nan, np.nan, '표본부족'
    if e_t <= 0 or e_c <= 0:
        return np.nan, np.nan, np.nan, '취업률 0'

    if e_t >= e_c:                       # 훈련군이 더 많이 취업 → 훈련군을 자름
        q = (e_t - e_c) / e_t
        hi, lo, who = y_t, y_c, '훈련군'
    else:
        q = (e_c - e_t) / e_c
        hi, lo, who = y_c, y_t, '대조군'
    if q <= 0:
        m = hi.mean() - lo.mean()
        return (m, m, 0.0, '자를 것 없음') if who == '훈련군' else (-m, -m, 0.0, '자를 것 없음')

    s = np.sort(hi)
    k = int(np.floor(q * len(s)))
    if k >= len(s) - 1:
        return np.nan, np.nan, q, '자를 비율이 너무 큼'
    trim_low_kept  = s[k:]               # 아래쪽 k 개를 버림 → 평균이 커짐 → 상한
    trim_high_kept = s[:len(s) - k]      # 위쪽 k 개를 버림  → 평균이 작아짐 → 하한
    a, b = trim_high_kept.mean() - lo.mean(), trim_low_kept.mean() - lo.mean()
    if who == '대조군':                  # 자른 쪽이 대조군이면 부호를 뒤집습니다
        a, b = -b, -a
    return float(a), float(b), float(q), who


def lee_table(out, horizons=HORIZONS):
    """관측창별 Y2 의 Lee 경계."""
    import pandas as pd
    tr = out['TREATED'].astype(bool)
    rows = []
    for k, h in enumerate(horizons):
        ok = out['OBS'][:, k]
        yt = out['Y2'][ok & tr, k]; yc = out['Y2'][ok & ~tr, k]
        et = float(np.nanmean(out['Y1'][ok & tr, k])) if (ok & tr).any() else np.nan
        ec = float(np.nanmean(out['Y1'][ok & ~tr, k])) if (ok & ~tr).any() else np.nan
        lo, hi, q, who = lee_bounds(yt, et, yc, ec)
        naive = (np.nanmean(yt) - np.nanmean(yc)) if len(yt) and len(yc) else np.nan
        rows.append({'관측창(개월)': h,
                     '취업률 훈련군': round(et, 4), '취업률 대조군': round(ec, 4),
                     '단순차이(편의 있음)': round(float(naive), 4),
                     'Lee 하한': round(lo, 4) if np.isfinite(lo) else None,
                     'Lee 상한': round(hi, 4) if np.isfinite(hi) else None,
                     '잘라낸 비율 q': round(q, 4) if np.isfinite(q) else None,
                     '자른 쪽': who})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════ 저장·적재
def out_path(name, realization=0):
    import os
    from .config import DIR_STD
    return os.path.join(DIR_STD, name, f'{name}_outcomes.npz')


def save(out, name, realization=0):
    import os
    p = out_path(name, realization)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez_compressed(p, **out)
    return p


def load(name, realization=0):
    z = np.load(out_path(name, realization), allow_pickle=False)
    return {k: z[k] for k in z.files}

# ══════════════════════════════════════════════ 정적 데이터(IHDP·Jobs)
def build_static(std):
    """
    IHDP·Jobs 는 시점이 하나입니다(T=1). 「길이 1 궤적」으로 승격해 두었으므로
    같은 로더·같은 모델 코드를 쓰되, 결과는 그 한 시점의 Y 하나뿐입니다.
    관측창 개념이 없으므로 H=1 로 두고 t*=0 을 씁니다.
    """
    N = std.N
    y = std.Y[:, 0, 0].astype(np.float32)
    tr = (std.A[:, 0, 0] > 0)
    one = np.ones((N, 1), bool)
    return dict(Y1=y.reshape(N, 1), Y2=y.reshape(N, 1), Y3=y.reshape(N, 1),
                OBS=one, TSTAR=np.zeros(N, np.int32),
                TREATED=tr.astype(np.int8),
                AIDX0=std.AIDX[:, 0, 0].astype(np.int32),
                HORIZONS=np.array([1], np.int32))


def for_dataset(std, name=None, realization=0):
    """데이터셋에 맞는 결과 묶음을 돌려줍니다. 시점이 하나면 정적 판으로."""
    import os
    name = name or std.meta['dataset']
    if std.T <= 1:
        return build_static(std)
    p = out_path(name, realization)
    return load(name, realization) if os.path.exists(p) else build(std, name)
