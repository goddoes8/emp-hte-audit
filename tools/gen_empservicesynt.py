# -*- coding: utf-8 -*-
"""
empservicesynt v2 — 반사실 정답이 있는 준합성 고용서비스 데이터 생성기
=====================================================================
v1(icml_datagen03.ipynb)의 결함 11건을 모두 고쳐 다시 설계했습니다.
설계 근거와 결함 대응은 docs/05_합성데이터_v2_설계.md 를 보십시오.

핵심 설계
---------
1. 처치는 `정책_훈련직종` 하나뿐 (c00 = 훈련 없음, c01~c36 = 훈련직종 36종).
   `istreated` 는 「관측기간 중 1회 이상 훈련」의 파생 지시자이며 처치가 아닙니다.
   (실데이터의 istreated 도 같은 정의임을 확인했습니다: 평균 0.5, 1,001/2,000명)
2. 참 효과는 성별 × 훈련직종 × 연령 × 관련경력의 함수입니다.
   취업효과 τ_emp 와 임금효과 τ_wage 를 따로 둡니다. 두 채널의 부호·크기가 달라
   Y1(취업)/Y2(임금)/Y3(기대소득)을 나눠 볼 필요가 실제로 생깁니다.
3. 잠재교란자를 둘로 나눕니다.
     U — 매달의 고용·임금 실현에 영향을 주므로 **이력에 흔적을 남깁니다.**
         관측 이력이 길수록 U 가 잘 드러납니다  → 가정 1(이력의 대리 충분성)이
         참인 영역. 문맥길이 K 소거연구(그림 3)의 정답이 여기서 나옵니다.
     V — 훈련 선택과 처치 이후 결과에만 영향을 주고 **이력에 흔적을 남기지 않습니다.**
         --hidden 으로 세기를 조절합니다. 이것이 어떤 모델도 지울 수 없는
         잔여 편향의 하한이며, 논문에서 「추정 가능한 한계」를 수치로 제시할 수 있습니다.
4. 훈련 선택은 U·V 에 음(-)으로 의존합니다. 즉 형편이 나쁜 사람이 더 많이 훈련받습니다.
   그래서 단순 비교는 참 효과를 **과소평가**합니다(실데이터에서 단순차이의 부호가
   뒤집힌 것과 같은 방향).
5. 양성성(positivity)을 의도적으로 위반시킨 칸을 둡니다.
   c24 운전 · c26 기계생산 · c27 금속생산(용접) · c28 전기생산 = 여성 수강자 0명.
   추정 금지 규칙(양성성 게이트)이 실제로 작동하는지 시험하기 위한 장치입니다.
6. 시간 순서를 지킵니다. t 행의 상태는 t 월 **시작 시점**(행동 이전),
   t 행의 행동은 t 월에 시작한 훈련, t 행의 보상은 t 월 전이 **이후**입니다.
   따라서 훈련 시작 월의 상태에는 그 훈련의 효과가 들어 있지 않습니다(누출 0).
7. 훈련은 D개월 지속되며 그 동안 취업확률이 낮아집니다(lock-in / J-곡선).
   효과는 훈련 종료 후에 정점을 찍고 반감기 24개월로 감쇠합니다.
8. 임금은 만원 단위 로그 한 번만 취합니다. 참값 MU 와 관측 Y 의 척도가 같습니다.
9. 산업은 직종과 별개의 c00~c22 (23종) 체계이며 직종의 복사본이 아닙니다.
10. 난수 시드를 고정하고 생성기를 노트북이 아닌 이 .py 에 둡니다.

반사실 정답
-----------
사람마다 훈련 결정 시점 t* 가 하나 있습니다. t* 이전 이력은 모든 반사실 분기가 공유합니다.
t* 에서 행동 a ∈ {0, 1..36} 을 강제로 넣고 공통난수(CRN)로 R회 앞으로 굴려
    MU_Y1[i,a,H] = P(H개월 후 취업)
    MU_Y2[i,a,H] = E[로그임금 | H개월 후 취업]
    MU_Y3[i,a,H] = E[취업 × 임금(만원)]        ← 정책 판단용
를 몬테카를로로 계산합니다. IHDP 의 mu0/mu1 과 같은 「조건부 기댓값」 정의입니다.

실행
----
    cd confieee2026
    python tools/gen_empservicesynt.py                 # 기본 (N=4000, hidden=0.35)
    python tools/gen_empservicesynt.py --hidden 0.0    # 가정 1이 정확히 성립하는 판
"""
from __future__ import annotations
import os, sys, json, argparse, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CONF = os.path.dirname(HERE)
sys.path.insert(0, CONF)
from cdtlib import jobcode as jc                                    # noqa: E402

# [86] 폴더 경로는 cdtlib/config.py 한 곳에서만 정합니다
from cdtlib.config import DIR_SYNT                              # noqa: E402

# ══════════════════════════════════════════════════════════ 코드 체계
N_OCC = 36                      # c01~c36 (c36 = 미분류).  c00 = 미취업 / 훈련없음
N_IND = 22                      # c01~c22.                c00 = 미취업
OCC = [f'c{i:02d}' for i in range(N_OCC + 1)]
IND = [f'c{i:02d}' for i in range(N_IND + 1)]

# 직종 → 산업 (다대일). 실데이터의 산업 23종 체계에 맞춘 고정 사상.
OCC2IND = np.array([
    0,                                    # c00 미취업
    1, 1, 2, 3, 4, 5, 6, 6, 7,            # c01 관리 … c09 교육
    7, 8, 8, 8, 9, 10, 10, 11, 11,        # c10 사복종교 … c18 여행
    12, 13, 9, 12, 14, 15, 4, 16, 17,     # c19 음식 … c27 금속생산
    17, 18, 19, 16, 16, 16, 3, 20, 21,    # c28 전기생산 … c35 농림어업, c36 미분류
], dtype=np.int16)
assert len(OCC2IND) == N_OCC + 1 and OCC2IND.max() <= N_IND

# ══════════════════════════════════════════════════════════ 참 효과표
# τ_emp : 훈련이 취업확률(로짓)에 주는 효과.  τ_wage : 로그임금에 주는 효과.
# 남성 기준값이며 여성 배수는 FEM_* 로 곱합니다.
# 값은 Card–Kluve–Weber(2018) 메타분석의 효과크기 범위(대체로 0~0.15 표준화효과)를
# 참고해 「정책적으로 의미 있으나 과장되지 않은」 폭으로 잡았습니다.
TAU_EMP = np.zeros(N_OCC + 1)
TAU_WAGE = np.zeros(N_OCC + 1)
FEM_EMP = np.ones(N_OCC + 1)
FEM_WAGE = np.ones(N_OCC + 1)

SCALE_EMP = 3.0     # 로짓 단위 효과 배율 (교정으로 정함)
SCALE_WAGE = 1.6    # 로그임금 단위 효과 배율


def _set(code, temp, twage, femp=1.0, fwage=1.0):
    i = int(code[1:])
    TAU_EMP[i] = temp * SCALE_EMP
    TAU_WAGE[i] = twage * SCALE_WAGE
    FEM_EMP[i], FEM_WAGE[i] = femp, fwage

#      코드   τ_emp  τ_wage  여성배수(취업) 여성배수(임금)
_set('c01', 0.20, 0.030, 0.70, 0.60)   # 관리
_set('c02', 0.45, 0.045, 1.05, 0.95)   # 경영사무   — 최다 수강, 효과 중간
_set('c03', 0.55, 0.090, 1.00, 0.95)   # 금융보험
_set('c04', 0.85, 0.155, 0.95, 0.95)   # IT연       — 남녀 모두 큰 효과
_set('c05', 0.40, 0.075, 0.60, 0.45)   # 건설연
_set('c06', 0.50, 0.080, 0.75, 0.60)   # 제조연
_set('c07', 0.15, 0.020, 1.00, 1.00)   # 인문연
_set('c08', 0.20, 0.030, 1.00, 1.00)   # 자과연
_set('c09', 0.35, 0.040, 1.20, 1.05)   # 교육
_set('c10', 0.45, 0.030, 1.25, 1.10)   # 사복종교
_set('c11', 0.10, 0.020, 1.00, 1.00)   # 법률       — 희소
_set('c12', 0.10, 0.015, 0.80, 0.80)   # 경찰       — 희소
_set('c13', 0.05, 0.010, 0.50, 0.50)   # 군인       — 희소
_set('c14', 0.70, 0.085, 1.35, 1.25)   # 보건의료   — 여성에게 더 큼
_set('c15', 0.30, 0.035, 1.10, 1.00)   # 예술방송
_set('c16', 0.15, 0.020, 1.05, 1.00)   # 스포레크   — 희소
_set('c17', 0.55, 0.050, 1.40, 1.20)   # 미용반려   — 여성에게 더 큼
_set('c18', 0.20, 0.025, 1.10, 1.00)   # 여행       — 희소
_set('c19', 0.35, 0.020, 1.10, 1.00)   # 음식       — 취업은 되나 임금은 거의 안 오름
_set('c20', 0.25, 0.030, 0.70, 0.70)   # 경호경비
_set('c21', 0.65, 0.025, 1.30, 1.10)   # 돌봄       — 취업효과 크고 임금효과 작음
_set('c22', 0.20, 0.010, 1.10, 1.00)   # 청소       — 거의 무효과
_set('c23', 0.30, 0.035, 1.05, 0.95)   # 영업
_set('c24', 0.50, 0.090, 0.00, 0.00)   # 운전       ★ 여성 수강자 0명 → 추정 불가
_set('c25', 0.45, 0.085, 0.35, 0.30)   # 건설채굴
_set('c26', 0.60, 0.110, 0.00, 0.00)   # 기계생산   ★ 여성 수강자 0명
_set('c27', 0.75, 0.130, 0.00, 0.00)   # 금속생산(용접) ★ 여성 수강자 0명 — 선생님 예시
_set('c28', 0.70, 0.120, 0.00, 0.00)   # 전기생산   ★ 여성 수강자 0명
_set('c29', 0.40, 0.070, 0.80, 0.75)   # 화학생산
_set('c30', 0.35, 0.040, 1.20, 1.05)   # 섬유생산
_set('c31', 0.30, 0.030, 1.10, 1.00)   # 식품생산
_set('c32', 0.25, 0.030, 1.00, 0.95)   # 기타생산
_set('c33', 0.10, 0.005, 1.00, 1.00)   # 제조단순   — 거의 무효과
_set('c34', 0.75, 0.125, 0.90, 0.90)   # IT정비
_set('c35', 0.20, 0.020, 0.95, 0.95)   # 농림어업   — 희소
_set('c36', 0.15, 0.015, 1.00, 1.00)   # 미분류

# 훈련직종 선택 확률의 기저(로그 가중). 실데이터의 쏠림(상위 5종 60.7%)을 모사.
TAKEUP_W = np.array([
    0.0,                                                   # c00 자리표시(미사용)
    0.6, 3.2, 0.9, 2.3, 0.5, 1.4, 0.3, 0.3, 0.7,           # c01~c09
    0.6, 0.05, 0.05, 0.02, 0.9, 2.0, 0.10, 1.1, 0.15,      # c10~c18
    1.7, 0.5, 0.9, 0.6, 0.9, 0.8, 0.6, 0.9, 0.9,           # c19~c27
    0.8, 0.4, 0.5, 1.2, 0.8, 0.4, 0.7, 0.12, 1.5,          # c28~c36
])
assert len(TAKEUP_W) == N_OCC + 1
FEM_TAKEUP = np.ones(N_OCC + 1)                     # 여성의 직종별 수강 상대확률
for c, m in {'c24': 0.0, 'c26': 0.0, 'c27': 0.0, 'c28': 0.0,
             'c05': 0.15, 'c25': 0.10, 'c20': 0.35, 'c13': 0.10,
             'c14': 2.0, 'c17': 2.6, 'c21': 2.4, 'c09': 1.6, 'c10': 1.7,
             'c30': 1.8, 'c02': 1.2}.items():
    FEM_TAKEUP[int(c[1:])] = m

# 직종별 임금 수준(만원 로그 절편 가산). 실데이터 Y|취업 평균 5.445, 표준편차 0.358 에 맞춤.
RNG_JOBW = np.random.default_rng(11)
JOB_WAGE = np.zeros(N_OCC + 1)
JOB_WAGE[1:] = np.array([
    0.42, 0.10, 0.30, 0.35, 0.26, 0.24, 0.06, 0.10, 0.14,
    0.02, 0.34, 0.16, 0.04, 0.14, 0.00, -0.04, -0.10, -0.06,
    -0.14, -0.06, -0.16, -0.20, 0.02, 0.10, 0.18, 0.16, 0.20,
    0.20, 0.14, -0.08, -0.04, -0.02, -0.18, 0.22, -0.10, -0.06,
])

# ══════════════════════════════════════════════════════════ 모수
P = dict(
    T=120, D_MIN=2, D_MAX=6,
    AGE_MIN=24, AGE_MAX=54,
    # 고용 전이
    SEP0=-3.10, SEP_U=-0.55, SEP_TEN=-0.012, SEP_AGE=0.010,
    HIRE0=-4.50, HIRE_U=0.85, HIRE_GAP=-0.030, HIRE_GAPCAP=18.0, HIRE_AGE=-0.022,
    HIRE_EDU=0.10, HIRE_LOCKIN=-1.30,
    # 임금 (만원 로그)
    W0=4.79, W_U=0.240, W_EDU=0.055, W_TEN=0.0022, W_EXP=0.0016,
    W_AGE=0.0060, W_ETA=0.060, W_EPS=0.115,
    # 훈련 선택
    TK0=-6.62, TK_U=-0.42, TK_V=-0.42, TK_UNEMP=1.60, TK_GAP=0.030,
    TK_AGE=-0.020, TK_EDU=0.06,
    # 처치 이후에만 작동하는 은닉 교란 V (이력에 흔적 없음)
    V_HIRE=0.55, V_WAGE=0.105,
    # 훈련 효과
    HALFLIFE=24.0, MATCH_SAME=1.00, MATCH_GROUP=0.35,
    REDIRECT=0.70,          # 훈련 후 재취업 시 그 직종으로 갈 확률 가산
    AGEMOD_SLOPE=0.020, AGEMOD_LO=0.30, AGEMOD_HI=1.30,
    RELEXP_BONUS=0.25,
)
HORIZONS = (6, 12, 24)
GROUP_ID = np.array([{'해당없음': -1, '미분류': -2}.get(jc.group(f'c{i:02d}'), 0)
                     if jc.group(f'c{i:02d}') in ('해당없음', '미분류') else 0
                     for i in range(N_OCC + 1)])
_G = {g: k for k, g in enumerate(sorted({jc.group(f'c{i:02d}') for i in range(1, N_OCC + 1)}))}
GROUP_ID = np.array([-1] + [_G[jc.group(f'c{i:02d}')] for i in range(1, N_OCC + 1)])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def agemod(age):
    return np.clip(1.0 - P['AGEMOD_SLOPE'] * (age - 30.0), P['AGEMOD_LO'], P['AGEMOD_HI'])


# ══════════════════════════════════════════════════════════ 사람 생성
def make_people(N, rng, hidden):
    g = rng.integers(0, 2, N)                                  # 0=남 1=여
    age0 = rng.integers(P['AGE_MIN'], P['AGE_MAX'] + 1, N).astype(float)
    edu = rng.integers(1, 6, N).astype(float)
    U = rng.normal(0, 1, N)
    V = rng.normal(0, 1, N) * (1.0 if hidden > 0 else 0.0)
    eta = rng.normal(0, P['W_ETA'], N)                          # 개인 임금 고정효과
    region = rng.integers(0, 5, N)
    return dict(g=g, age0=age0, edu=edu, U=U, V=V, eta=eta, region=region, hidden=hidden)


def tau_of(pp, occ_idx, relexp):
    """occ_idx: (M,) 훈련직종 인덱스(0=훈련없음). 반환 (τ_emp, τ_wage)."""
    fem = pp['g'] == 1
    te = TAU_EMP[occ_idx] * np.where(fem, FEM_EMP[occ_idx], 1.0)
    tw = TAU_WAGE[occ_idx] * np.where(fem, FEM_WAGE[occ_idx], 1.0)
    m = agemod(pp['age0']) * (1.0 + P['RELEXP_BONUS'] * relexp)
    return te * m, tw * m


# ══════════════════════════════════════════════════════════ 전이 엔진
def advance(st, pp, noise, absmonth):
    """한 달 전진. st 는 제자리 수정. 반환 없음.
    호출 시점에 st 는 '해당 월 시작 상태'이고, 행동(st['act_now'])은 이미 정해져 있음."""
    M = st['emp'].shape[0]
    idx = st['pidx']                       # 사람 인덱스 (M,)
    U, V, g, edu = pp['U'][idx], pp['V'][idx], pp['g'][idx], pp['edu'][idx]
    age = pp['age0'][idx] + absmonth // 12
    hid = pp['hidden']

    # 1) 훈련 시작
    start = st['act_now'] > 0
    if start.any():
        st['tr_occ'][start] = st['act_now'][start]
        st['tr_end'][start] = absmonth[start] if np.ndim(absmonth) else absmonth
        st['tr_end'][start] = (absmonth[start] if np.ndim(absmonth) else absmonth) + st['dur'][start]
        st['in_tr'][start] = True
        st['ever'][start] = True

    # 2) 훈련 종료 → 잔존효과 1.0 에서 시작
    am = absmonth if np.ndim(absmonth) else np.full(M, absmonth)
    done = st['in_tr'] & (am >= st['tr_end'])
    st['in_tr'][done] = False
    st['res'] *= 0.5 ** (1.0 / P['HALFLIFE'])
    st['res'][done] = 1.0

    # 3) 훈련 효과
    te, tw = st['te'], st['tw']
    eff_e = te * st['res']

    # 4) 고용 전이
    emp = st['emp'].astype(bool)
    sep_p = sigmoid(P['SEP0'] + P['SEP_U'] * U + P['SEP_TEN'] * st['ten'] + P['SEP_AGE'] * (age - 30))
    hire_lin = (P['HIRE0'] + P['HIRE_U'] * U + hid * P['V_HIRE'] * V
                + P['HIRE_GAP'] * np.minimum(st['gap'], P['HIRE_GAPCAP']) + P['HIRE_AGE'] * (age - 30)
                + P['HIRE_EDU'] * edu + eff_e
                + P['HIRE_LOCKIN'] * st['in_tr'])
    hire_p = sigmoid(hire_lin)

    lose = emp & (noise['u_sep'] < sep_p)
    gain = (~emp) & (noise['u_hire'] < hire_p)

    st['emp'][lose] = 0
    st['job'][lose] = 0
    st['ten'][lose] = 0.0
    st['comp'][lose] = 0
    st['reg'][lose] = -1
    st['etype'][lose] = -1
    st['ctype'][lose] = -1

    if gain.any():
        # 재취업 직종: 훈련직종으로 유도(REDIRECT × 잔존효과) 아니면 사전분포
        want = st['tr_occ'].copy()
        redirect = gain & (want > 0) & (noise['u_red'] < P['REDIRECT'] * st['res'])
        newjob = st['prior_job'].copy()
        newjob[redirect] = want[redirect]
        take = gain
        st['emp'][take] = 1
        st['job'][take] = newjob[take]
        st['ten'][take] = 0.0
        st['comp'][take] = 1 + (noise['u_comp'][take] * 11).astype(np.int16)
        st['reg'][take] = pp['region'][idx][take]
        st['etype'][take] = (noise['u_et'][take] * 3).astype(np.int16)
        st['ctype'][take] = (noise['u_ct'][take] * 2).astype(np.int16)

    emp = st['emp'].astype(bool)

    # 5) 임금 (취업자만)
    job = st['job']
    same = (job == st['tr_occ']) & (job > 0)
    grp = (GROUP_ID[job] == GROUP_ID[np.maximum(st['tr_occ'], 0)]) & (job > 0) & (st['tr_occ'] > 0) & ~same
    match = np.where(same, P['MATCH_SAME'], np.where(grp, P['MATCH_GROUP'], 0.0))
    logw = (P['W0'] + pp['eta'][idx] + P['W_U'] * U + hid * P['V_WAGE'] * V
            + P['W_EDU'] * edu + P['W_AGE'] * (age - 30)
            + P['W_TEN'] * st['ten'] + P['W_EXP'] * st['totexp']
            + JOB_WAGE[job] + tw * st['res'] * match
            + noise['e_w'])
    st['wage'] = np.where(emp, logw, 0.0)

    # 6) 상태 누적
    st['ten'][emp] += 1.0
    st['totexp'][emp] += 1.0
    st['gap'][emp] = 0.0
    st['gap'][~emp] += 1.0
    st['act_now'][:] = 0


def new_state(M, pidx, pp, rng):
    st = dict(
        pidx=pidx.astype(np.int32),
        emp=np.zeros(M, np.int8), job=np.zeros(M, np.int16),
        ten=np.zeros(M), totexp=np.zeros(M), gap=np.zeros(M),
        comp=np.zeros(M, np.int16), reg=np.full(M, -1, np.int16),
        etype=np.full(M, -1, np.int16), ctype=np.full(M, -1, np.int16),
        res=np.zeros(M), tr_occ=np.zeros(M, np.int16), tr_end=np.full(M, 1 << 30, np.int64),
        in_tr=np.zeros(M, bool), ever=np.zeros(M, bool), dur=np.zeros(M, np.int64),
        act_now=np.zeros(M, np.int16), wage=np.zeros(M),
        te=np.zeros(M), tw=np.zeros(M),
        prior_job=np.zeros(M, np.int16),
    )
    return st


def step_noise(M, h, base_seed):
    """h 번째 달의 난수. 행동(반사실 분기)이 달라도 동일한 값이 나오도록
    (base_seed, h) 로 결정론적으로 생성합니다 = 공통난수(CRN)."""
    r = np.random.default_rng((base_seed, h))
    return dict(u_sep=r.random(M), u_hire=r.random(M), u_red=r.random(M),
                u_comp=r.random(M), u_et=r.random(M), u_ct=r.random(M),
                u_job=r.random(M), e_w=r.normal(0, P['W_EPS'], M))


def _job_cdf():
    w = np.stack([TAKEUP_W[1:], TAKEUP_W[1:] * np.maximum(FEM_TAKEUP[1:], 0.05)])
    w = w / w.sum(1, keepdims=True)
    return w.cumsum(1)


_JOB_CDF = _job_cdf()


def job_prior(pp, idx, u):
    """재취업 시 기본 직종 추첨(성별 가중). u 는 (M,) 균등난수."""
    fem = (pp['g'][idx] == 1).astype(np.int8)
    c = _JOB_CDF[fem]                                   # (M, 36)
    return (1 + (u[:, None] > c).sum(1)).astype(np.int16)


# ══════════════════════════════════════════════════════════ 사실 궤적
def simulate_factual(N, pp, rng, T):
    idx = np.arange(N)
    st = new_state(N, idx, pp, rng)
    st['dur'] = rng.integers(P['D_MIN'], P['D_MAX'] + 1, N)
    NOISE_SEED_F = 900001
    # 초기 상태: 절반은 취업 상태에서 시작
    st['emp'] = (rng.random(N) < 0.55).astype(np.int8)
    st['job'] = np.where(st['emp'] == 1, job_prior(pp, idx, rng.random(N)), 0).astype(np.int16)
    st['ten'] = np.where(st['emp'] == 1, rng.integers(0, 40, N), 0).astype(float)
    st['totexp'] = st['ten'] + rng.integers(0, 60, N)
    st['comp'] = np.where(st['emp'] == 1, rng.integers(1, 12, N), 0).astype(np.int16)
    st['reg'] = np.where(st['emp'] == 1, pp['region'], -1).astype(np.int16)
    st['etype'] = np.where(st['emp'] == 1, rng.integers(0, 3, N), -1).astype(np.int16)
    st['ctype'] = np.where(st['emp'] == 1, rng.integers(0, 2, N), -1).astype(np.int16)

    # 누적 경력(직종·산업) — 사실 궤적에서만 추적
    expocc = np.zeros((N, N_OCC + 1), np.float32)
    expind = np.zeros((N, N_IND + 1), np.float32)
    resocc = np.zeros((N, N_OCC + 1), np.float32)
    j0 = st['job']
    expocc[idx, j0] = st['ten']
    expind[idx, OCC2IND[j0]] = st['ten']

    rows = dict(emp=[], job=[], gap=[], comp=[], reg=[], etype=[], ctype=[],
                act=[], wage=[], expocc=[], expind=[], resocc=[], ten=[], totexp=[])
    tstar = np.full(N, -1, np.int64)
    obs_a = np.zeros(N, np.int16)
    snapshot = None

    for t in range(1, T + 1):
        # ── (1) 상태 기록 (행동 이전)
        rows['emp'].append(st['emp'].copy()); rows['job'].append(st['job'].copy())
        rows['gap'].append(st['gap'].copy()); rows['comp'].append(st['comp'].copy())
        rows['reg'].append(st['reg'].copy()); rows['etype'].append(st['etype'].copy())
        rows['ctype'].append(st['ctype'].copy()); rows['ten'].append(st['ten'].copy())
        rows['totexp'].append(st['totexp'].copy())
        rows['expocc'].append(expocc.copy()); rows['expind'].append(expind.copy())
        rows['resocc'].append(resocc.copy())

        # ── (2) 훈련 결정 (아직 훈련 안 받았고 관측창이 남은 사람만)
        age = pp['age0'] + t // 12
        cand = (~st['ever']) & (t <= T - max(HORIZONS) - P['D_MAX'])
        lin = (P['TK0'] + P['TK_U'] * pp['U'] + P['TK_V'] * pp['V'] * (pp['hidden'] > 0)
               + P['TK_UNEMP'] * (st['emp'] == 0) + P['TK_GAP'] * np.minimum(st['gap'], 24)
               + P['TK_AGE'] * (age - 30) + P['TK_EDU'] * pp['edu'])
        take = cand & (rng.random(N) < sigmoid(lin))
        act = np.zeros(N, np.int16)
        if take.any():
            ii = np.where(take)[0]
            w = np.tile(TAKEUP_W[1:], (len(ii), 1))
            fem = pp['g'][ii] == 1
            w = np.where(fem[:, None], w * FEM_TAKEUP[1:], w)
            # 관련 경력이 있으면 그 직종을 고를 확률 가산
            w = w * (1.0 + 1.5 * (expocc[ii, 1:] > 6))
            s = w.sum(1, keepdims=True)
            w = w / s
            c = w.cumsum(1)
            u = rng.random((len(ii), 1))
            act[ii] = (1 + (u > c).sum(1)).astype(np.int16)
            tstar[ii] = t
            obs_a[ii] = act[ii]
        rows['act'].append(act.copy())
        st['act_now'] = act

        # 훈련 효과 계수 확정
        newly = act > 0
        if newly.any():
            relexp = (expocc[np.arange(N), np.maximum(act, 0)] > 6).astype(float)
            te, tw = tau_of(pp, np.maximum(act, 0), relexp)
            st['te'][newly] = te[newly]
            st['tw'][newly] = tw[newly]

        # ── (2b) t* 직전 상태 스냅샷 저장
        if snapshot is None:
            snapshot = dict(saved=np.zeros(N, bool))
            for k in ('emp', 'job', 'ten', 'totexp', 'gap', 'comp', 'reg', 'etype', 'ctype', 'dur'):
                snapshot[k] = np.zeros(N, dtype=st[k].dtype)
            snapshot['t'] = np.zeros(N, np.int64)
        newsnap = take & (~snapshot['saved'])
        if newsnap.any():
            for k in ('emp', 'job', 'ten', 'totexp', 'gap', 'comp', 'reg', 'etype', 'ctype', 'dur'):
                snapshot[k][newsnap] = st[k][newsnap]
            snapshot['t'][newsnap] = t
            snapshot['saved'][newsnap] = True

        # ── (3) 전이 + 보상
        nz = step_noise(N, t, NOISE_SEED_F)
        st['prior_job'] = job_prior(pp, idx, nz['u_job'])
        advance(st, pp, nz, np.full(N, t, np.int64))
        rows['wage'].append(st['wage'].copy())

        # ── (4) 누적 갱신 (기록 이후)
        e = st['emp'].astype(bool)
        expocc[e, st['job'][e]] += 1.0 / 12.0 * 12.0   # 개월 단위
        expind[e, OCC2IND[st['job'][e]]] += 1.0
        resocc *= 0.5 ** (1.0 / P['HALFLIFE'])
        fin = st['ever'] & (~st['in_tr']) & (st['res'] > 0)
        resocc[np.arange(N), st['tr_occ']] = np.where(fin, st['res'], resocc[np.arange(N), st['tr_occ']])
        resocc[:, 0] = 0.0

    return rows, tstar, obs_a, snapshot


# ══════════════════════════════════════════════════════════ 반사실
def simulate_cf(pp, snapshot, action, R, rng, Hmax):
    """모든 사람에 대해 t* 에서 action 을 강제하고 R개 복제를 Hmax 개월 전진."""
    N = len(pp['U'])
    M = N * R
    pidx = np.repeat(np.arange(N), R)
    st = new_state(M, pidx, pp, rng)
    for k in ('emp', 'job', 'ten', 'totexp', 'gap', 'comp', 'reg', 'etype', 'ctype', 'dur'):
        st[k] = np.repeat(snapshot[k], R).astype(st[k].dtype)
    t0 = np.repeat(snapshot['t'], R)
    NOISE_SEED_CF = 900777

    a = np.full(M, action, np.int16)
    a[~np.repeat(snapshot['saved'], R)] = 0
    st['act_now'] = a.copy()
    if action > 0:
        relexp = np.zeros(M)
        te, tw = tau_of({k: np.repeat(v, R) if isinstance(v, np.ndarray) else v
                         for k, v in pp.items()}, np.maximum(a, 0), relexp)
        st['te'], st['tw'] = te, tw

    out = {}
    for h in range(1, Hmax + 1):
        nz = step_noise(M, h, NOISE_SEED_CF)
        st['prior_job'] = job_prior(pp, pidx, nz['u_job'])
        advance(st, pp, nz, t0 + h)
        if h in HORIZONS:
            out[h] = (st['emp'].copy().astype(float), st['wage'].copy())
    return out, R


# ══════════════════════════════════════════════════════════ 조립
def build(N, seed, hidden, R, outdir, tag=None):
    t_start = time.time()
    rng = np.random.default_rng(seed)
    pp = make_people(N, rng, hidden)
    T = P['T']
    rows, tstar, obs_a, snap = simulate_factual(N, pp, rng, T)
    print('  사실 궤적 완료  %.1fs' % (time.time() - t_start))

    # ── 반사실
    rng2 = np.random.default_rng(seed + 777)
    Hmax = max(HORIZONS)
    truth = []
    for a in range(0, N_OCC + 1):
        out, _ = simulate_cf(pp, snap, a, R, rng2, Hmax)
        for h, (emp, wage) in out.items():
            e = emp.reshape(N, R)
            w = wage.reshape(N, R)
            mu1 = e.mean(1)
            with np.errstate(invalid='ignore'):
                s = e.sum(1)
                mu2 = np.where(s >= 3, (w * e).sum(1) / np.maximum(s, 1), np.nan)
                mu3 = (e * np.exp(w)).mean(1)
            truth.append(pd.DataFrame(dict(
                persona=[f't{i:05d}' for i in range(N)], action=f'c{a:02d}', horizon=h,
                MU_Y1=mu1.astype(np.float32), MU_Y2=mu2.astype(np.float32),
                MU_Y3=mu3.astype(np.float32),
                SE_Y1=(np.sqrt(np.maximum(mu1 * (1 - mu1), 0) / R)).astype(np.float32))))
        if a % 9 == 0:
            print('    반사실 a=c%02d  %.1fs' % (a, time.time() - t_start))
    truth = pd.concat(truth, ignore_index=True)

    # ── 관측 데이터프레임
    stack = lambda k: np.stack(rows[k], 1)                       # (N, T, ...)
    emp = stack('emp'); job = stack('job'); gap = stack('gap')
    comp = stack('comp'); reg = stack('reg'); et = stack('etype'); ct = stack('ctype')
    act = stack('act'); wage = stack('wage')
    expocc = np.stack(rows['expocc'], 1); expind = np.stack(rows['expind'], 1)
    resocc = np.stack(rows['resocc'], 1)

    tt = np.tile(np.arange(1, T + 1), (N, 1))
    yr = (tt - 1) // 12
    REGN = ['csu', 'cky', 'cch', 'cho', 'cy']
    ETN = ['c1', 'c2', 'cy']; CTN = ['cN', 'cY']
    fmt = np.array([f'c{i:02d}' for i in range(max(N_OCC, N_IND) + 1)])

    d = {}
    d['persona'] = np.repeat([f't{i:05d}' for i in range(N)], T)
    ever = (act > 0).any(1)
    d['istreated'] = np.repeat(ever.astype(np.int64), T)
    d['메모리'] = tt.ravel()
    d['상태_년'] = (yr + 1).ravel()
    d['상태_년범주'] = np.array([f'c{2015 + y}' for y in range(10)])[yr].ravel()
    d['상태_분기범주'] = np.array(['c1', 'c2', 'c3', 'c4'])[((tt - 1) % 12) // 3].ravel()
    d['상태_성별_남'] = np.repeat((pp['g'] == 0).astype(np.int64), T)
    d['상태_성별_여'] = np.repeat((pp['g'] == 1).astype(np.int64), T)
    d['상태_나이'] = ((pp['age0'][:, None] + yr) / 100.0).ravel()
    d['상태_기업규모'] = comp.astype(np.int64).ravel()
    d['상태_지역범주'] = np.where(reg < 0, 'c00', np.array(REGN + ['c00'])[reg]).ravel()
    d['상태_고용형태구분코드'] = np.where(et < 0, 'c00', np.array(ETN + ['c00'])[et]).ravel()
    d['상태_계약근로여부'] = np.where(ct < 0, 'c00', np.array(CTN + ['c00'])[ct]).ravel()
    d['상태_현사업장직종'] = fmt[job].ravel()
    d['상태_현사업장산업'] = fmt[OCC2IND[job]].ravel()
    d['상태_현재실직기간'] = (gap / 12.0).ravel()
    d['정책_훈련직종'] = fmt[act].ravel()
    d['보상_로그임금'] = wage.ravel()
    d['보상_손실함수계산여부'] = (wage > 0).astype(np.float64).ravel()
    for j in range(N_OCC + 1):
        d[f'상태_생애_훈련직종c{j:02d}효과잔존량'] = resocc[:, :, j].ravel()
    for j in range(N_OCC + 1):
        d[f'상태_생애_직종c{j:02d}누적경력기간'] = (expocc[:, :, j] / 12.0).ravel()
    for j in range(N_IND + 1):
        d[f'상태_생애_산업c{j:02d}누적경력기간'] = (expind[:, :, j] / 12.0).ravel()
    df = pd.DataFrame(d)

    # ── 사람 단위 진실
    per = pd.DataFrame(dict(
        persona=[f't{i:05d}' for i in range(N)],
        성별=np.where(pp['g'] == 0, '남', '여'), 나이=pp['age0'].astype(int),
        학력=pp['edu'].astype(int), U=pp['U'].astype(np.float32), V=pp['V'].astype(np.float32),
        tstar=tstar, 훈련직종=fmt[obs_a], 훈련기간=snap['dur'],
    ))

    os.makedirs(outdir, exist_ok=True)
    tag = tag or ('v2' if hidden > 0 else 'v2h0')
    f1 = os.path.join(outdir, f'empservicesynt_{tag}.parquet')
    f2 = os.path.join(outdir, f'empservicesynt_{tag}_truth.parquet')
    f3 = os.path.join(outdir, f'empservicesynt_{tag}_person.parquet')
    f4 = os.path.join(outdir, f'empservicesynt_{tag}_meta.json')
    df.to_parquet(f1, index=False, compression='snappy')
    truth.to_parquet(f2, index=False, compression='snappy')
    per.to_parquet(f3, index=False, compression='snappy')
    meta = dict(version='v2', tag=tag, seed=seed, N=N, T=T, R=R, hidden=hidden,
                n_occ=N_OCC, n_ind=N_IND, horizons=list(HORIZONS),
                params=P, tau_emp=TAU_EMP.tolist(), tau_wage=TAU_WAGE.tolist(),
                fem_emp=FEM_EMP.tolist(), fem_wage=FEM_WAGE.tolist(),
                takeup_w=TAKEUP_W.tolist(), fem_takeup=FEM_TAKEUP.tolist(),
                occ2ind=OCC2IND.tolist(), job_wage=JOB_WAGE.tolist(),
                female_zero=['c24', 'c26', 'c27', 'c28'],
                생성시각='(스크립트 실행 시각은 파일 mtime 참조)')
    with open(f4, 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    print('  저장 완료  %.1fs' % (time.time() - t_start))
    print('   ', f1, df.shape)
    print('   ', f2, truth.shape)
    print('   ', f3, per.shape)
    return df, truth, per, meta


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=20260821)
    ap.add_argument('--hidden', type=float, default=0.35,
                    help='이력에 흔적을 남기지 않는 은닉 교란 V 의 세기. 0 이면 가정 1이 정확히 성립')
    ap.add_argument('--reps', type=int, default=40, help='반사실 몬테카를로 복제 수')
    ap.add_argument('--out', default=DIR_SYNT)
    a = ap.parse_args()
    print('empservicesynt v2 생성  N=%d seed=%d hidden=%.2f R=%d' % (a.n, a.seed, a.hidden, a.reps))
    build(a.n, a.seed, a.hidden, a.reps, a.out)


# ══════════════════════════════════════════════════════════ 규칙표 입출력
RULE_COLS = ['코드', '직종명', 'τ_취업', 'τ_로그임금', '여성배수_취업', '여성배수_임금',
             '수강가중', '여성수강배수', '임금수준']


def rules_to_frame():
    """현재 참 효과·수강 규칙을 표 하나로 내보냅니다. 노트북에서 고쳐 쓰기 위한 것."""
    rows = []
    for i in range(1, N_OCC + 1):
        c = f'c{i:02d}'
        rows.append([c, jc.name(c), float(TAU_EMP[i]), float(TAU_WAGE[i]),
                     float(FEM_EMP[i]), float(FEM_WAGE[i]),
                     float(TAKEUP_W[i]), float(FEM_TAKEUP[i]), float(JOB_WAGE[i])])
    return pd.DataFrame(rows, columns=RULE_COLS)


def rules_from_frame(df):
    """표를 되읽어 생성기에 적용합니다. 반드시 build() 전에 부르십시오."""
    global _JOB_CDF
    miss = [c for c in RULE_COLS if c not in df.columns]
    if miss:
        raise ValueError(f'규칙표에 없는 열: {miss}')
    for _, r in df.iterrows():
        i = int(str(r['코드'])[1:])
        if not (1 <= i <= N_OCC):
            raise ValueError(f"코드 범위를 벗어남: {r['코드']}")
        TAU_EMP[i] = float(r['τ_취업'])
        TAU_WAGE[i] = float(r['τ_로그임금'])
        FEM_EMP[i] = float(r['여성배수_취업'])
        FEM_WAGE[i] = float(r['여성배수_임금'])
        TAKEUP_W[i] = float(r['수강가중'])
        FEM_TAKEUP[i] = float(r['여성수강배수'])
        JOB_WAGE[i] = float(r['임금수준'])
    _JOB_CDF = _job_cdf()                  # 수강 가중이 바뀌면 반드시 다시 계산
    return dict(n_rows=len(df), 여성수강0인직종=[f'c{i:02d}' for i in range(1, N_OCC + 1)
                                                if FEM_TAKEUP[i] == 0.0])
