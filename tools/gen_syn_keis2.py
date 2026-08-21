# -*- coding: utf-8 -*-
"""
syn8 — `synthetic_데이타생성규칙02.xlsx` 구현
=============================================
`gen_syn_keis.py`(syn7, 규칙01)의 후속입니다. syn7 은 그대로 두었으므로
규칙01 판을 다시 만들 때는 그쪽을 쓰십시오.

규칙02 가 규칙01 에 더한 것
---------------------------
· 변수 4개  Gender_Female · Age_Year · Act_Train_Female · Act_Train_Age
· 페르소나 3종  P6 성별민감(용접) · P7 연령민감(IT) · P8 성별민감(보건)
· Sheet2  모집단(인구비중·여성비율·연령범위)과 공정(전이·수강 확률)

P1~P5 는 새 변수 4개가 전부 0 이므로 **규칙01 과 같은 결과**가 나옵니다.

임금식
------
    로그임금(t) = Intercept
                + Time_Year        × 경과연수
                + Gender_Female    × 1{여성}
                + Age_Year         × (나이 − 30)
                + 훈련효과(t)
                + Act_Placement    × 1{알선으로 취업}
                + Exp_Matched      × 1{현직종 = 타겟직종}  (아니면 Not Exp_Matched)
                + Exp_General      × 총경력(년)
                + Company_Size     × 1{기업규모 ≥ 6}
                + Contract_Type    × 1{계약근로여부 = 'cN'}
                + Gap_Duration     × 직전 실직기간(년)
                + Residual

    훈련효과(t) = max(0,  Act_Train_Code
                        + Act_Train_Female × 1{여성}
                        + Act_Train_Age    × (나이 − 30)
                        + Last_Train(Decay) × 훈련 후 경과개월 )

    ※ Act_Train_Female 은 **가산항**입니다. P6 은 이 값이 −Act_Train_Code 와 같아
      여성의 훈련효과가 정확히 0 이 됩니다.

    python tools/gen_syn_keis2.py --xlsx docs/synthetic_데이타생성규칙02.xlsx --n 2000
"""
from __future__ import annotations
import os, sys, json, math, argparse, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CONF = os.path.dirname(HERE)
sys.path.insert(0, CONF); sys.path.insert(0, HERE)
from gen_empservicesynt import OCC2IND, N_OCC, N_IND                    # noqa: E402
from gen_syn_keis import SPEC                                           # noqa: E402

# [86] 폴더 경로는 cdtlib/config.py 한 곳에서만 정합니다
from cdtlib.config import DIR_SYNT                              # noqa: E402
VARS = ['Intercept', 'Residual', 'Act_Train_Code', 'Act_Train_Female', 'Act_Train_Age',
        'Act_Placement', 'Time_Year', 'Exp_Matched', 'Not Exp_Matched', 'Exp_General',
        'Last_Train (Decay)', 'Company_Size', 'Contract_Type', 'Gap_Duration',
        'Gender_Female', 'Age_Year']
REGIONS = ['csu', 'cky', 'cch', 'cho', 'cy']
EMPL_TYPES = ['c1', 'c2', 'cy']
CONTRACT_TYPES = ['cN', 'cY']


# ══════════════════════════════════════════════════ 규칙표
def read_rules(xlsx):
    import openpyxl
    wb = openpyxl.load_workbook(xlsx, data_only=True)
    ws = wb['Sheet1'] if 'Sheet1' in wb.sheetnames else wb.worksheets[0]
    cur, rec = None, []
    for r in ws.iter_rows(values_only=True):
        r = list(r) + [None] * 9
        if r[0] and str(r[0]).strip().startswith('P') and len(str(r[0]).strip()) <= 3:
            cur = str(r[0]).strip()
        if r[3] and str(r[3]).strip() in VARS:
            rec.append(dict(P=cur, 변수=str(r[3]).strip(), 값=r[4],
                            계수=r[5], sd=r[6], 만원=r[7], 만원sd=r[8]))
    df = pd.DataFrame(rec)
    RULES, TARGET = {}, {}
    for p, g in df.groupby('P', sort=True):
        d = {r['변수']: (float(r['계수'] or 0), float(r['sd'] or 0)) for _, r in g.iterrows()}
        miss = [v for v in VARS if v not in d]
        if miss:
            raise ValueError(f'{p} 에 없는 변수: {miss}  (규칙02 형식이 맞습니까?)')
        RULES[p] = d
        v = g.loc[g['변수'] == 'Act_Train_Code', '값'].iloc[0]
        TARGET[p] = int(''.join(c for c in str(v) if c.isdigit()))

    # 계수 ↔ 만원 정합성
    chk = []
    for p, g in df.groupby('P', sort=True):
        b = float(g.loc[g['변수'] == 'Intercept', '만원'].iloc[0])
        tm = float(g.loc[g['변수'] == 'Act_Train_Code', '만원'].iloc[0])
        for _, r in g.iterrows():
            w = float(r['만원'])
            if r['변수'] == 'Intercept':
                want, how = math.log(b), 'log(기저)'
            elif r['변수'] == 'Act_Train_Female':
                want = math.log((b + w) / b) - math.log((b + tm) / b); how = '여성-남성 로그차'
            elif r['변수'] == 'Act_Train_Age':
                want = math.log((b + tm + w) / b) - math.log((b + tm) / b); how = '1세당 로그차'
            else:
                want, how = math.log((b + w) / b), 'log((기저+증가)/기저)'
            chk.append([p, r['변수'], float(r['계수']), w, b, round(want, 4),
                        abs(float(r['계수']) - want) < 5e-4, how])
    CHK = pd.DataFrame(chk, columns=['페르소나', '변수', '규칙표계수', '증가만원', '기저만원',
                                     '재계산계수', '일치', '검증식'])

    # Sheet2 — 모집단·공정
    POP, PROC = {}, {}
    if 'Sheet2' in wb.sheetnames:
        w2 = wb['Sheet2']; mode = None
        for r in w2.iter_rows(values_only=True):
            r = list(r) + [None] * 7
            a = str(r[0]).strip() if r[0] else ''
            if a.startswith('[모집단'): mode = 'pop'; continue
            if a.startswith('[공정'): mode = 'proc'; continue
            if a.startswith('[실데이터'): mode = None; continue
            if mode == 'pop' and a in RULES:
                POP[a] = dict(share=float(r[3]), female=float(r[4]),
                              age_min=int(r[5]), age_max=int(r[6]))
            elif mode == 'proc' and a and r[1] is not None and a != '항목':
                PROC[a] = float(r[1])
    if not POP:
        POP = {p: dict(share=1 / len(RULES), female=0.5, age_min=24, age_max=54) for p in RULES}
    return RULES, TARGET, CHK, POP, PROC


# ══════════════════════════════════════════════════ 임금식
def draw_beta(rule, rng):
    b = {k: (float(rng.normal(mu, sd)) if sd > 0 else float(mu)) for k, (mu, sd) in rule.items()}
    b['Residual_sd'] = rule['Residual'][1]; b['Residual'] = 0.0
    return b


def train_effect(b, months_since, female, age):
    if months_since is None:
        return 0.0
    m = months_since if SPEC['DECAY_UNIT'] == 'month' else months_since / 12.0
    v = (b['Act_Train_Code'] + b['Act_Train_Female'] * female
         + b['Act_Train_Age'] * (age - 30.0) + b['Last_Train (Decay)'] * m)
    return max(0.0, v) if SPEC['TRAIN_FLOOR0'] else v


def log_wage(b, st, t, target, since, female, age, eps):
    w = b['Intercept'] + b['Time_Year'] * (t / 12.0 if SPEC['TIME_UNIT'] == 'year' else t)
    w += b['Gender_Female'] * female
    w += b['Age_Year'] * (age - 30.0)
    w += train_effect(b, since, female, age)
    if st['placed']:
        w += b['Act_Placement']
    w += b['Exp_Matched'] if st['job'] == target else b['Not Exp_Matched']
    w += b['Exp_General'] * st['exp_total_y']
    w += b['Company_Size'] * (1.0 if st['size'] >= SPEC['SIZE_CUT'] else 0.0)
    w += b['Contract_Type'] * (1.0 if st['contract'] == 'cN' else 0.0)
    w += b['Gap_Duration'] * (st['prev_gap_m'] / (12.0 if SPEC['GAP_UNIT'] == 'year' else 1.0))
    return w + eps


# ══════════════════════════════════════════════════ 생성
def build(xlsx, N=2000, seed=20260821, outdir=DIR_SYNT, tag='syn8'):
    t0 = time.time()
    RULES, TARGET, CHK, POP, PR = read_rules(xlsx)
    rng = np.random.default_rng(seed)
    T = int(PR.get('T', 120)); DEC = float(PR.get('DECAY_PERIOD', 60.0))
    GATE = int(PR.get('TRAIN_GATE_MONTHS', 9)); FSD = float(PR.get('TRAIN_FRAILTY_SD', 0.0))

    ps = list(RULES); share = np.array([POP[p]['share'] for p in ps], float); share /= share.sum()
    who = rng.choice(len(ps), size=N, p=share)

    recs, per = [], []
    for pid in range(N):
        p = ps[who[pid]]; b = draw_beta(RULES[p], rng); target = TARGET[p]
        female = 1.0 if rng.random() < POP[p]['female'] else 0.0
        age0 = int(rng.integers(POP[p]['age_min'], POP[p]['age_max'] + 1))
        edu = int(rng.integers(1, 6))
        frail = float(np.exp(rng.normal(0, FSD) - FSD * FSD / 2)) if FSD > 0 else 1.0

        st = dict(emp=1, job=target, size=int(rng.integers(1, 12)),
                  region=REGIONS[int(rng.integers(0, 5))],
                  etype=EMPL_TYPES[int(rng.integers(0, 3))],
                  contract=CONTRACT_TYPES[int(rng.integers(0, 2))],
                  gap_m=0, prev_gap_m=0, placed=False, exp_total_y=0.0)
        expocc = np.zeros(N_OCC + 1); expind = np.zeros(N_IND + 1); resocc = np.zeros(N_OCC + 1)
        last_t, n_tr, pend = None, 0, False

        for t in range(1, T + 1):
            age = age0 + (t - 1) // 12
            row = {'persona': f't{pid:05d}', '메모리': t, '상태_년': (t - 1) // 12 + 1,
                   '상태_년범주': f'c{2015 + (t - 1) // 12}',
                   '상태_분기범주': f'c{((t - 1) % 12) // 3 + 1}',
                   '상태_성별_남': int(female == 0), '상태_성별_여': int(female == 1),
                   '상태_나이': age / 100.0,
                   '상태_기업규모': st['size'] if st['emp'] else 0,
                   '상태_지역범주': st['region'] if st['emp'] else 'c00',
                   '상태_고용형태구분코드': st['etype'] if st['emp'] else 'c00',
                   '상태_계약근로여부': st['contract'] if st['emp'] else 'c00',
                   '상태_현사업장직종': f'c{st["job"]:02d}' if st['emp'] else 'c00',
                   '상태_현사업장산업': f'c{OCC2IND[st["job"]]:02d}' if st['emp'] else 'c00',
                   '상태_현재실직기간': st['gap_m'] / 12.0}
            for j in range(N_OCC + 1):
                row[f'상태_생애_훈련직종c{j:02d}효과잔존량'] = resocc[j]
            row['상태_생애_훈련직종c00효과잔존량'] = float(resocc[1:].sum())
            for j in range(N_OCC + 1):
                row[f'상태_생애_직종c{j:02d}누적경력기간'] = expocc[j] / 12.0
            row['상태_생애_직종c00누적경력기간'] = st['exp_total_y']
            for j in range(N_IND + 1):
                row[f'상태_생애_산업c{j:02d}누적경력기간'] = expind[j] / 12.0
            row['상태_생애_산업c00누적경력기간'] = st['exp_total_y']

            act = 0
            if (last_t is None) or (t - last_t > GATE):
                pr = PR.get('P_TRAIN_UNEMP', .014) if st['emp'] == 0 else PR.get('P_TRAIN_EMP', .0014)
                if rng.random() < min(1.0, pr * frail):
                    act = target if rng.random() < PR.get('P_TARGET_JOB', .8) else int(rng.integers(1, N_OCC + 1))
            if st['emp'] == 0 and rng.random() < PR.get('P_PLACE', .3):
                pend = True
            row['정책_훈련직종'] = f'c{act:02d}' if act > 0 else 'c00'

            if act > 0:
                last_t = t; n_tr += 1
            if st['emp'] == 1:
                if rng.random() < PR.get('P_SEP', .2):
                    st.update(emp=0, job=0, size=0, region='c00', etype='c00',
                              contract='c00', placed=False)
            else:
                if rng.random() < PR.get('P_HIRE', .1):
                    st['prev_gap_m'] = st['gap_m']
                    st.update(emp=1, job=target, size=int(rng.integers(1, 12)),
                              region=REGIONS[int(rng.integers(0, 5))],
                              etype=EMPL_TYPES[int(rng.integers(0, 3))],
                              contract=CONTRACT_TYPES[int(rng.integers(0, 2))], placed=pend)
                    pend = False
            if st['emp'] == 1:
                st['gap_m'] = 0; expocc[st['job']] += 1
                expind[OCC2IND[st['job']]] += 1; st['exp_total_y'] += 1 / 12.0
            else:
                st['gap_m'] += 1

            resocc *= 0.5 ** (1.0 / DEC)
            if act > 0:
                resocc[act] = 1.0
            resocc[0] = 0.0

            since = None if last_t is None else (t - last_t)
            eps = float(rng.normal(0, b['Residual_sd']))
            if st['emp'] == 1:
                y = log_wage(b, st, t, target, since, female, age, eps)
                mu0 = log_wage(b, st, t, target, None, female, age, 0.0)
                mu1 = log_wage(b, st, t, target, 0, female, age, 0.0)
                mob = log_wage(b, st, t, target, since, female, age, 0.0)
            else:
                y = mu0 = mu1 = mob = 0.0
            row.update({'보상_로그임금': y, '보상_손실함수계산여부': int(st['emp'] == 1),
                        'True_Mu0': mu0, 'True_Mu1': mu1, 'True_Mu_obs': mob})
            recs.append(row)

        per.append(dict(persona=f't{pid:05d}', 페르소나=p, 타겟직종=f'c{target:02d}',
                        성별='여' if female else '남', 나이=age0, 학력=edu,
                        수강성향=round(frail, 4), 훈련횟수=n_tr,
                        참_훈련효과=round(train_effect(b, 0, female, age0), 4),
                        **{f'b_{k}': v for k, v in b.items() if k != 'Residual'}))

    df = pd.DataFrame(recs)
    ever = df.groupby('persona')['정책_훈련직종'].transform(lambda s: (s != 'c00').any())
    df.insert(1, 'istreated', ever.astype(int))
    base = ['persona', 'istreated', '메모리', '상태_년', '상태_년범주', '상태_분기범주',
            '상태_성별_남', '상태_성별_여', '상태_나이', '상태_기업규모', '상태_지역범주',
            '상태_고용형태구분코드', '상태_계약근로여부', '상태_현사업장직종', '상태_현사업장산업',
            '상태_현재실직기간', '정책_훈련직종', '보상_로그임금', '보상_손실함수계산여부']
    cols = (base + [f'상태_생애_훈련직종c{i:02d}효과잔존량' for i in range(N_OCC + 1)]
            + [f'상태_생애_직종c{i:02d}누적경력기간' for i in range(N_OCC + 1)]
            + [f'상태_생애_산업c{i:02d}누적경력기간' for i in range(N_IND + 1)]
            + ['True_Mu0', 'True_Mu1', 'True_Mu_obs'])
    df = df[cols]; dper = pd.DataFrame(per)

    os.makedirs(outdir, exist_ok=True)
    f1 = os.path.join(outdir, f'empservicesynt_{tag}.parquet')
    df.to_parquet(f1, index=False, compression='snappy')
    dper.to_parquet(os.path.join(outdir, f'empservicesynt_{tag}_person.parquet'),
                    index=False, compression='snappy')
    CHK.to_csv(os.path.join(outdir, f'empservicesynt_{tag}_규칙검증.csv'),
               index=False, encoding='utf-8-sig')
    json.dump(dict(version='syn8', tag=tag, seed=seed, N=N, T=T,
                   xlsx=os.path.basename(xlsx), spec=SPEC, pop=POP, proc=PR,
                   rules={p: {k: list(v) for k, v in d.items()} for p, d in RULES.items()},
                   target=TARGET, 규칙표_계수검증=bool(CHK['일치'].all())),
              open(os.path.join(outdir, f'empservicesynt_{tag}_meta.json'), 'w',
                   encoding='utf-8'), ensure_ascii=False, indent=1)
    print('  생성 완료 %.1fs  %s  →  %s' % (time.time() - t0, df.shape, f1))
    return df, dper, CHK, RULES, TARGET, POP, PR


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--xlsx', default=os.path.join(CONF, 'docs', 'synthetic_데이타생성규칙02.xlsx'))
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20260821)
    ap.add_argument('--tag', default='syn8')
    ap.add_argument('--out', default=DIR_SYNT)
    a = ap.parse_args()
    build(a.xlsx, a.n, a.seed, a.out, a.tag)
