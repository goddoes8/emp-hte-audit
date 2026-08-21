# -*- coding: utf-8 -*-
"""
syn7 — `synthetic_데이타생성규칙.xlsx` 를 그대로 구현한 합성데이터 생성기
=========================================================================
`icml_dgen_syn6_keis.ipynb` 의 후속(버전 6 → 7)입니다.

syn6 은 규칙표의 계수 11개 중 **9개를 쓰지 않고** 코드에 박아 넣은 상수로 대체했습니다.
syn7 은 규칙표를 xlsx 에서 직접 읽어 그대로 씁니다.

규칙표의 구조 (검증 완료)
-------------------------
60개 계수 전부가 다음을 만족합니다.

    계수 = log( (기저만원 + 증가만원) / 기저만원 ),   Intercept 계수 = log(기저만원)

즉 규칙표는 **만원 단위 임금의 로그를 종속변수로 하는 가법 선형모형**입니다.
예) P1 Intercept 5.5215 = log(250),  Act_Train 0.1823 = log(300/250).

    로그임금(t) = Intercept
                + Time_Year        × 경과연수
                + 훈련효과(t)                        ← Act_Train_Code + Last_Train(Decay)×경과월
                + Act_Placement    × 1{알선으로 취업}
                + Exp_Matched      × 1{현직종 = 타겟직종}   (아니면 Not_Exp_Matched)
                + Exp_General      × 총경력(년)
                + Company_Size     × 1{기업규모 ≥ SIZE_CUT}
                + Contract_Type    × 1{계약근로여부 = 'cN'}
                + Gap_Duration     × 직전 실직기간(년)
                + Residual         ~ N(0, σ)

규칙표에 적혀 있지 않아 syn6 을 승계한 부분
-------------------------------------------
· 고용 전이확률 (취업→실직 0.2 / 실직→취업 0.1)
· 훈련 수강 확률 (실업 0.05 / 취업 0.005, 직전 훈련 후 12개월 경과 조건)
· 알선 확률 (실업 중 0.3)
이 셋은 규칙표의 범위 밖입니다. `PROC` 로 바꿀 수 있게 열어 두었습니다.

해석이 필요한 부분 (SPEC 으로 전부 조절 가능)
---------------------------------------------
규칙표는 계수만 주고 「무엇에 곱하는가」는 적어 두지 않았습니다. 아래가 기본 해석이며,
근거를 함께 적었습니다. 다르게 보시면 SPEC 만 바꾸면 됩니다.

· TIME_UNIT='year'      — 변수명이 Time_**Year** 이고 +3만원/년이 자연스럽습니다
· DECAY_UNIT='month'    — 연 단위로 보면 P1 훈련효과가 22.8**년** 지속됩니다(비현실).
                          월 단위면 22.8개월이고, 페르소나별로 22.8/9.4/4.8/2.5/2.4개월이 되어
                          「고효율 훈련형이 오래 간다」는 페르소나 이름과 맞습니다
· GAP_UNIT='year'       — P5 가 -50만원입니다. 월 단위면 한 달 쉬고 -50만원이라 비현실적입니다
· EXP_MATCH='indicator' — 규칙표에 `Not Exp_Matched` 라는 **대립 범주**가 값과 함께 있습니다.
                          기울기라면 「아닌 경우」에 별도 계수가 붙을 이유가 없습니다
· SIZE='indicator'      — +20만원이 단일 프리미엄으로 읽힙니다. 1~11 코드에 기울기로 곱하면
                          최대 +0.85 로그(2.3배)가 되어 과합니다
· CONTRACT='indicator'  — 계약근로여부 cN(무기) 에 프리미엄

실행
----
    cd confieee2026
    python tools/gen_syn_keis.py --xlsx <규칙표.xlsx> --n 2000
"""
from __future__ import annotations
import os, sys, json, math, argparse, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CONF = os.path.dirname(HERE)
sys.path.insert(0, CONF)
from cdtlib import jobcode as jc                                       # noqa: E402
from gen_empservicesynt import OCC2IND, N_OCC, N_IND                   # noqa: E402

# [86] 폴더 경로는 cdtlib/config.py 한 곳에서만 정합니다
from cdtlib.config import DIR_SYNT                              # noqa: E402

# ══════════════════════════════════════════════════ 해석 규약
SPEC = dict(
    TIME_UNIT='year',        # 'year' | 'month'
    DECAY_UNIT='month',      # 'month' | 'year'
    GAP_UNIT='year',         # 'year'  | 'month'
    EXP_MATCH='indicator',   # 'indicator' | 'slope'   (slope 면 타겟직종 누적경력(년)에 곱함)
    SIZE='indicator',        # 'indicator' | 'scaled'  (scaled 면 (규모-1)/10 에 곱함)
    SIZE_CUT=6,
    CONTRACT='indicator',
    TRAIN_FLOOR0=True,       # 훈련효과가 감쇠로 음수가 되지 않게 0 에서 절단
)

# 규칙표 밖 — syn6 승계
PROC = dict(
    P_SEP=0.20, P_HIRE=0.10,
    P_TRAIN_UNEMP=0.05, P_TRAIN_EMP=0.005, TRAIN_GATE_MONTHS=12,
    P_PLACE=0.30, P_TARGET_JOB=0.80,
    T=120, DECAY_PERIOD=60.0,
)

REGIONS = ['csu', 'cky', 'cch', 'cho', 'cy']
EMPL_TYPES = ['c1', 'c2', 'cy']
CONTRACT_TYPES = ['cN', 'cY']
VARS = ['Intercept', 'Residual', 'Act_Train_Code', 'Act_Placement', 'Time_Year',
        'Exp_Matched', 'Not Exp_Matched', 'Exp_General', 'Last_Train (Decay)',
        'Company_Size', 'Contract_Type', 'Gap_Duration']


# ══════════════════════════════════════════════════ 규칙표 읽기
def read_rules(xlsx):
    """규칙 xlsx → {persona: {변수: (평균, 표준편차)}} + 타겟직종 + 검증표."""
    import openpyxl
    ws = openpyxl.load_workbook(xlsx, data_only=True).worksheets[0]
    rows = list(ws.iter_rows(min_row=1, values_only=True))
    # 헤더 위치 자동 탐색
    start = next(i for i, r in enumerate(rows)
                 if r and any(str(c).strip() == 'Intercept' for c in r if c))
    cur, rec = None, []
    for r in rows[start:]:
        pid, pname, gubun, var, val, mu, sd = (list(r) + [None] * 7)[:7]
        muw, sdw = (list(r) + [None] * 9)[7:9]
        if pid:
            cur = str(pid).strip()
        if var is None:
            continue
        rec.append(dict(P=cur, 구분=gubun, 변수=str(var).strip(), 값=(val or ''),
                        계수=mu, 계수sd=sd, 만원=muw, 만원sd=sdw))
    df = pd.DataFrame(rec)
    df = df[df['변수'].isin(VARS)].copy()

    RULES, TARGET = {}, {}
    for p, g in df.groupby('P', sort=True):
        d = {}
        for _, r in g.iterrows():
            d[r['변수']] = (float(r['계수'] or 0.0), float(r['계수sd'] or 0.0))
            if r['변수'] == 'Act_Train_Code':
                TARGET[p] = int(''.join(ch for ch in str(r['값']) if ch.isdigit()) or 0)
        missing = [v for v in VARS if v not in d]
        if missing:
            raise ValueError(f'{p} 에 없는 변수: {missing}')
        RULES[p] = d

    # 검증: 계수 = log((기저+증가)/기저)
    chk = []
    for p, g in df.groupby('P', sort=True):
        base = float(g.loc[g['변수'] == 'Intercept', '만원'].iloc[0])
        for _, r in g.iterrows():
            if r['만원'] is None:
                continue
            want = (math.log(base) if r['변수'] == 'Intercept'
                    else math.log((base + float(r['만원'])) / base))
            chk.append([p, r['변수'], float(r['계수']), float(r['만원']), base,
                        round(want, 4), abs(float(r['계수']) - want) < 5e-4])
    chk = pd.DataFrame(chk, columns=['페르소나', '변수', '규칙표계수', '증가만원',
                                     '기저만원', '재계산계수', '일치'])
    return RULES, TARGET, chk


# ══════════════════════════════════════════════════ 사람별 계수 추첨
def draw_beta(rule, rng):
    b = {}
    for k, (mu, sd) in rule.items():
        b[k] = float(rng.normal(mu, sd)) if sd > 0 else float(mu)
    b['Residual_sd'] = rule['Residual'][1]           # Residual 은 잡음의 표준편차
    b['Residual'] = 0.0
    return b


def train_effect(b, months_since):
    """훈련효과 = Act_Train_Code + Last_Train(Decay) × 경과.  기본은 0 에서 절단."""
    if months_since is None:
        return 0.0
    m = months_since if SPEC['DECAY_UNIT'] == 'month' else months_since / 12.0
    v = b['Act_Train_Code'] + b['Last_Train (Decay)'] * m
    return max(0.0, v) if SPEC['TRAIN_FLOOR0'] else v


def log_wage(b, st, t, target, months_since_train, with_noise, eps):
    """규칙표 그대로의 가법 로그임금. 미취업이면 정의되지 않으므로 호출하지 않는다."""
    tt = (t / 12.0) if SPEC['TIME_UNIT'] == 'year' else float(t)
    w = b['Intercept'] + b['Time_Year'] * tt
    w += train_effect(b, months_since_train)
    if st['placed']:
        w += b['Act_Placement']
    if SPEC['EXP_MATCH'] == 'indicator':
        w += b['Exp_Matched'] if st['job'] == target else b['Not Exp_Matched']
    else:
        w += b['Exp_Matched'] * st['exp_target_y'] + b['Not Exp_Matched'] * st['exp_other_y']
    w += b['Exp_General'] * st['exp_total_y']
    if SPEC['SIZE'] == 'indicator':
        w += b['Company_Size'] * (1.0 if st['size'] >= SPEC['SIZE_CUT'] else 0.0)
    else:
        w += b['Company_Size'] * ((st['size'] - 1) / 10.0)
    w += b['Contract_Type'] * (1.0 if st['contract'] == 'cN' else 0.0)
    g = st['prev_gap_m'] / (12.0 if SPEC['GAP_UNIT'] == 'year' else 1.0)
    w += b['Gap_Duration'] * g
    return w + (eps if with_noise else 0.0)


# ══════════════════════════════════════════════════ 생성
def build(xlsx, N=2000, seed=20260821, outdir=DIR_SYNT, tag='syn7'):
    t0 = time.time()
    RULES, TARGET, CHK = read_rules(xlsx)
    personas = sorted(RULES)
    rng = np.random.default_rng(seed)
    T = PROC['T']

    recs, per = [], []
    for pid in range(N):
        p = personas[pid % len(personas)]
        b = draw_beta(RULES[p], rng)
        target = TARGET[p]
        gender = int(rng.integers(0, 2))
        age0 = int(rng.integers(24, 55))
        edu = int(rng.integers(1, 6))
        region0 = REGIONS[int(rng.integers(0, len(REGIONS)))]

        st = dict(emp=1, job=target, size=int(rng.integers(1, 12)),
                  region=region0, etype=EMPL_TYPES[int(rng.integers(0, 3))],
                  contract=CONTRACT_TYPES[int(rng.integers(0, 2))],
                  gap_m=0, prev_gap_m=0, placed=False,
                  exp_total_y=0.0, exp_target_y=0.0, exp_other_y=0.0)
        expocc = np.zeros(N_OCC + 1); expind = np.zeros(N_IND + 1)
        resocc = np.zeros(N_OCC + 1)
        last_train_t = None           # 마지막 훈련 이수 시점
        n_train = 0
        placed_pending = False

        for t in range(1, T + 1):
            since = None if last_train_t is None else (t - last_train_t)

            # ── ① 상태 기록 (행동 이전, 전이 이전)  ← 누출이 생길 수 없는 순서
            row = {
                'persona': f't{pid:05d}',
                '메모리': t, '상태_년': (t - 1) // 12 + 1,
                '상태_년범주': f'c{2015 + (t - 1) // 12}',
                '상태_분기범주': f'c{((t - 1) % 12) // 3 + 1}',
                '상태_성별_남': int(gender == 0), '상태_성별_여': int(gender == 1),
                '상태_나이': (age0 + (t - 1) // 12) / 100.0,
                '상태_기업규모': st['size'] if st['emp'] else 0,
                '상태_지역범주': st['region'] if st['emp'] else 'c00',
                '상태_고용형태구분코드': st['etype'] if st['emp'] else 'c00',
                '상태_계약근로여부': st['contract'] if st['emp'] else 'c00',
                '상태_현사업장직종': f'c{st["job"]:02d}' if st['emp'] else 'c00',
                '상태_현사업장산업': f'c{OCC2IND[st["job"]]:02d}' if st['emp'] else 'c00',
                '상태_현재실직기간': st['gap_m'] / 12.0,
            }
            for j in range(N_OCC + 1):
                row[f'상태_생애_훈련직종c{j:02d}효과잔존량'] = resocc[j]
            row['상태_생애_훈련직종c00효과잔존량'] = float(resocc[1:].sum())
            for j in range(N_OCC + 1):
                row[f'상태_생애_직종c{j:02d}누적경력기간'] = expocc[j] / 12.0
            row['상태_생애_직종c00누적경력기간'] = st['exp_total_y']
            for j in range(N_IND + 1):
                row[f'상태_생애_산업c{j:02d}누적경력기간'] = expind[j] / 12.0
            row['상태_생애_산업c00누적경력기간'] = st['exp_total_y']

            # ── ② 행동
            act_train, act_place = 0, 0
            gate = (last_train_t is None) or (t - last_train_t > PROC['TRAIN_GATE_MONTHS'])
            if gate:
                pr = PROC['P_TRAIN_UNEMP'] if st['emp'] == 0 else PROC['P_TRAIN_EMP']
                if rng.random() < pr:
                    act_train = (target if rng.random() < PROC['P_TARGET_JOB']
                                 else int(rng.integers(1, N_OCC + 1)))
            if st['emp'] == 0 and rng.random() < PROC['P_PLACE']:
                act_place = 1
            row['정책_훈련직종'] = f'c{act_train:02d}' if act_train > 0 else 'c00'

            # ── ③ 전이
            if act_train > 0:
                last_train_t = t
                n_train += 1
            if act_place == 1:
                placed_pending = True

            if st['emp'] == 1:
                if rng.random() < PROC['P_SEP']:
                    st.update(emp=0, job=0, size=0, region='c00',
                              etype='c00', contract='c00', placed=False)
            else:
                if rng.random() < PROC['P_HIRE']:
                    st.update(emp=1, job=target, size=int(rng.integers(1, 12)),
                              region=REGIONS[int(rng.integers(0, len(REGIONS)))],
                              etype=EMPL_TYPES[int(rng.integers(0, 3))],
                              contract=CONTRACT_TYPES[int(rng.integers(0, 2))],
                              placed=placed_pending)
                    st['prev_gap_m'] = st['gap_m']
                    placed_pending = False

            if st['emp'] == 1:
                st['gap_m'] = 0
                expocc[st['job']] += 1; expind[OCC2IND[st['job']]] += 1
                st['exp_total_y'] += 1 / 12.0
                if st['job'] == target: st['exp_target_y'] += 1 / 12.0
                else:                   st['exp_other_y'] += 1 / 12.0
            else:
                st['gap_m'] += 1

            # 훈련 잔존량: 이수한 달의 '다음' 기록부터 1.0 에서 시작해 감쇠.
            # ①에서 이미 기록을 마친 뒤 여기서 올리므로 훈련 시작 월의 상태에는 들어가지 않는다.
            resocc *= 0.5 ** (1.0 / PROC['DECAY_PERIOD'])
            if act_train > 0:
                resocc[act_train] = 1.0
            resocc[0] = 0.0

            # ── ④ 보상 (전이 이후)
            since_now = None if last_train_t is None else (t - last_train_t)
            eps = float(rng.normal(0, b['Residual_sd']))
            if st['emp'] == 1:
                y = log_wage(b, st, t, target, since_now, True, eps)
                mu0 = log_wage(b, st, t, target, None, False, 0.0)          # 훈련 이력 없음
                mu1 = log_wage(b, st, t, target, 0, False, 0.0)             # 이번 달 타겟훈련 이수
                mu_obs = log_wage(b, st, t, target, since_now, False, 0.0)  # 관측 이력 그대로
            else:
                y = mu0 = mu1 = mu_obs = 0.0
            row['보상_로그임금'] = y
            row['보상_손실함수계산여부'] = 1 if st['emp'] == 1 else 0
            row['True_Mu0'] = mu0
            row['True_Mu1'] = mu1
            row['True_Mu_obs'] = mu_obs
            recs.append(row)

        per.append(dict(persona=f't{pid:05d}', 페르소나=p, 타겟직종=f'c{target:02d}',
                        성별='남' if gender == 0 else '여', 나이=age0, 학력=edu,
                        훈련횟수=n_train,
                        **{f'b_{k}': v for k, v in b.items() if k != 'Residual'}))

    df = pd.DataFrame(recs)
    ever = df.groupby('persona')['정책_훈련직종'].transform(lambda s: (s != 'c00').any())
    df.insert(1, 'istreated', ever.astype(int))       # ★ 파생 지시자 (처치 아님)

    base = ['persona', 'istreated', '메모리', '상태_년', '상태_년범주', '상태_분기범주',
            '상태_성별_남', '상태_성별_여', '상태_나이', '상태_기업규모', '상태_지역범주',
            '상태_고용형태구분코드', '상태_계약근로여부', '상태_현사업장직종',
            '상태_현사업장산업', '상태_현재실직기간', '정책_훈련직종',
            '보상_로그임금', '보상_손실함수계산여부']
    cols = (base
            + [f'상태_생애_훈련직종c{i:02d}효과잔존량' for i in range(N_OCC + 1)]
            + [f'상태_생애_직종c{i:02d}누적경력기간' for i in range(N_OCC + 1)]
            + [f'상태_생애_산업c{i:02d}누적경력기간' for i in range(N_IND + 1)]
            + ['True_Mu0', 'True_Mu1', 'True_Mu_obs'])
    df = df[cols]
    dper = pd.DataFrame(per)

    os.makedirs(outdir, exist_ok=True)
    f1 = os.path.join(outdir, f'empservicesynt_{tag}.parquet')
    f2 = os.path.join(outdir, f'empservicesynt_{tag}_person.parquet')
    f3 = os.path.join(outdir, f'empservicesynt_{tag}_meta.json')
    f4 = os.path.join(outdir, f'empservicesynt_{tag}_규칙검증.csv')
    df.to_parquet(f1, index=False, compression='snappy')
    dper.to_parquet(f2, index=False, compression='snappy')
    CHK.to_csv(f4, index=False, encoding='utf-8-sig')
    json.dump(dict(version='syn7', tag=tag, seed=seed, N=N, T=T,
                   xlsx=os.path.basename(xlsx), spec=SPEC, proc=PROC,
                   rules={p: {k: list(v) for k, v in d.items()} for p, d in RULES.items()},
                   target=TARGET, 규칙표_계수검증=bool(CHK['일치'].all())),
              open(f3, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('  생성 완료 %.1fs  %s  %s' % (time.time() - t0, df.shape, f1))
    return df, dper, CHK, RULES, TARGET


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--xlsx', required=True)
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20260821)
    ap.add_argument('--tag', default='syn7')
    ap.add_argument('--out', default=DIR_SYNT)
    a = ap.parse_args()
    build(a.xlsx, a.n, a.seed, a.out, a.tag)
