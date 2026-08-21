# -*- coding: utf-8 -*-
"""
경로·상수 중앙집중.

폴더 규약
    p_2026icml/                  ← 프로젝트 루트 (원자료가 있는 곳, 건드리지 않음)
    └─ confieee2026/             ← CONF_ROOT. 이번 논문 작업물은 전부 여기 안에만 만든다
       ├─ cdtlib/                코드
       ├─ datastd/               표준화 데이터 (산출물)
       ├─ out/                   기술분석 표·그림 (산출물)
       ├─ runs/                  실험 결과 (산출물)
       ├─ paper/                 논문 원고 · fig/ · tab/
       └─ docs/                  문서

모든 경로는 __file__ 기준 절대경로다. 따라서 노트북에서 os.chdir 을 할 필요가 없다.
"""
import os

CONF_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # confieee2026/
PROJECT_ROOT = os.path.dirname(CONF_ROOT)                                    # p_2026icml/

# ── 읽기 전용 (프로젝트 루트에 있음. 어떤 코드도 여기 쓰지 않는다)
DIR_RAW     = os.path.join(PROJECT_ROOT, 'data')
DIR_DATARAW = os.path.join(PROJECT_ROOT, 'dataraw')     # 고용보험 원자료 7.6GB
DIR_CODE    = os.path.join(PROJECT_ROOT, 'datacode')    # 코드테이블

# ── 산출물 (전부 confieee2026/ 안)
DIR_STD   = os.path.join(CONF_ROOT, 'datastd')
DIR_SYNT  = os.path.join(CONF_ROOT, 'datasynt')   # v2 합성데이터(생성물)
DIR_REAL  = os.path.join(CONF_ROOT, 'datareal')   # 실데이터 코호트(생성물)
DIR_REF   = os.path.join(CONF_ROOT, 'dataref')    # 벤치마크 정본 대조용(내려받음)
# [86] DIR_REAL 은 여기가 **유일한 정의**입니다. 16판까지 datagen 이 따로 정의하고
#   있어서, 폴더를 옮기면 한쪽만 바뀌는 사고가 날 자리였습니다. 생성기도 여기서 읽습니다.
DIR_OUT   = os.path.join(CONF_ROOT, 'out')
DIR_RUNS  = os.path.join(CONF_ROOT, 'runs')
DIR_DOCS  = os.path.join(CONF_ROOT, 'docs')
DIR_PAPER = os.path.join(CONF_ROOT, 'paper')             # 논문 집필
DIR_FIG   = os.path.join(DIR_PAPER, 'fig')               # 원고에 들어갈 그림
DIR_TAB   = os.path.join(DIR_PAPER, 'tab')               # 원고에 들어갈 표

RAW = {
    'ihdp_train'     : os.path.join(DIR_RAW, '1IHDP', 'ihdp_npci_1-100.train.npz'),
    'ihdp_test'      : os.path.join(DIR_RAW, '1IHDP', 'ihdp_npci_1-100.test.npz'),
    'ihdp1000_train' : os.path.join(DIR_RAW, '1IHDP', 'ihdp_npci_1-1000.train.npz'),
    'ihdp1000_test'  : os.path.join(DIR_RAW, '1IHDP', 'ihdp_npci_1-1000.test.npz'),
    'jobs_train'     : os.path.join(DIR_RAW, 'jobs_train.npz'),
    'jobs_test'      : os.path.join(DIR_RAW, 'jobs_test.npz'),
    'empservicesynt' : os.path.join(DIR_RAW, '4empservicesynt', 'empservicesynt.parquet'),
    'empservicereal' : os.path.join(DIR_RAW, '5empservicereal', 'empservicereal.parquet'),
    # [84] 확대 코호트. icml_datagen16 이 confieee2026/datareal/ 에 만듭니다.
    'empservicereal_r16k' : os.path.join(CONF_ROOT, 'datareal', 'empservicereal_r16k.parquet'),
    # [105] 확대 코호트 — 검정력 요구(전체 145,374명)를 충족하는 규모. icml_datagen31 이 만듭니다.
    'empservicereal_r240k': os.path.join(CONF_ROOT, 'datareal', 'empservicereal_r240k.parquet'),
    'empservicereal_r160k': os.path.join(CONF_ROOT, 'datareal', 'empservicereal_r160k.parquet'),
    'empservicesyntv2'      : os.path.join(DIR_SYNT, 'empservicesynt_v2.parquet'),
    'empservicesyntv2_truth': os.path.join(DIR_SYNT, 'empservicesynt_v2_truth.parquet'),
    'empservicesyntv2_person': os.path.join(DIR_SYNT, 'empservicesynt_v2_person.parquet'),
    'empservicesyntsyn7'        : os.path.join(DIR_SYNT, 'empservicesynt_syn7.parquet'),
    'empservicesyntsyn7_person' : os.path.join(DIR_SYNT, 'empservicesynt_syn7_person.parquet'),
    'empservicesyntsyn8'        : os.path.join(DIR_SYNT, 'empservicesynt_syn8.parquet'),
    'empservicesyntsyn8_person' : os.path.join(DIR_SYNT, 'empservicesynt_syn8_person.parquet'),
}

# 분할 비율 — 파일에 고정 저장되므로 실행마다 달라지지 않는다.
# ── 상태 지연 (단계 2) ───────────────────────────────────────
# 행 t 의 상태가 그 달의 행동 결과를 이미 담고 있으면 「훈련을 안 받았다면」을 만들 수 없다.
# 실데이터는 훈련 당월 상태에 그 훈련의 효과잔존량이 100% 들어 있으므로 1시점 지연시킨다.
# 합성 v2·syn7·syn8 은 생성 순서를 「상태 → 행동 → 보상」으로 잡아 이미 누출이 0 이라 지연 없음.
STATE_LAG = {
    'empservicereal'    : 1,
    'empservicesynt'    : 1,   # v1 (폐기 대상). 같은 누출이 있어 동일 처리
    'empservicesyntv2'  : 0,
    'empservicesyntsyn7': 0,
    'empservicesyntsyn8': 0,
}

SPLIT_SEED  = 20260821
VAL_FRAC    = 0.20
SPLIT_RATIO = (0.70, 0.15, 0.15)
TRAJ_LEN    = 120
SPLIT_NAMES = {0: 'train', 1: 'valid', 2: 'test'}


def ensure_dirs():
    for d in (DIR_STD, DIR_SYNT, DIR_REAL, DIR_REF, DIR_OUT, DIR_RUNS, DIR_DOCS, DIR_FIG, DIR_TAB):
        os.makedirs(d, exist_ok=True)


def show():
    """경로가 맞는지 한눈에 확인."""
    rows = [('CONF_ROOT', CONF_ROOT), ('PROJECT_ROOT', PROJECT_ROOT),
            ('원자료 DIR_RAW', DIR_RAW), ('표준 DIR_STD', DIR_STD),
            ('산출 DIR_OUT', DIR_OUT), ('실험 DIR_RUNS', DIR_RUNS),
            ('논문 DIR_PAPER', DIR_PAPER), ('그림 DIR_FIG', DIR_FIG), ('표 DIR_TAB', DIR_TAB)]
    w = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f'{k:<{w}}  {v}  {"" if os.path.exists(v) else "  ← 없음"}')


# ══════════════════════════════════════ [84] 실데이터 코호트 전환
#
# 왜 스위치로 두는가.  코호트를 2,000명 → 16,000명으로 키우면서 **변수 정의까지 바뀌면**
# 「n 이 늘어 달라진 것」인지 「변수가 바뀌어 달라진 것」인지 구별할 수 없습니다.
# 새 생성기는 상태열이 111개(현재실직기간 추가)인데 기존 분석은 70개를 씁니다.
# 그래서 새 코호트에서도 **기존과 똑같은 70개 열만** 골라 쓰도록 화이트리스트를 둡니다.
# 이러면 바뀌는 것은 사람 수 하나뿐이라, 그림 3 의 예측을 그대로 시험할 수 있습니다.
#
#   REAL_COHORT = None      → 기존 2,000명 (기본)
#   REAL_COHORT = 'r16k'    → 확대 16,000명, 열은 기존 70개로 맞춤
REAL_COHORT = os.environ.get('REAL_COHORT') or None

# 기존 분석이 쓰는 상태열 70개. 새 코호트에서 이 목록만 골라 씁니다.
# (없는 열이 있으면 dataio 가 이름을 찍고 멈춥니다 — 조용히 다른 변수로 도는 것을 막습니다)
STATE_KEEP = {}


def use_cohort(tag=None, verbose=True):
    """실데이터 코호트를 고릅니다. 노트북 설정 셀에서 한 줄로 부릅니다."""
    global REAL_COHORT
    REAL_COHORT = tag or None
    if verbose:
        print('실데이터 코호트: %s' % (tag or '기존 2,000명'))
        if tag:
            print('  파일 %s' % RAW.get('empservicereal_%s' % tag, '(등록 안 됨)'))
    return REAL_COHORT


def real_path():
    """지금 고른 코호트의 파일 경로."""
    if REAL_COHORT:
        k = 'empservicereal_%s' % REAL_COHORT
        if k not in RAW:
            raise KeyError('RAW 에 %s 가 없습니다. config.py 에 등록하십시오.' % k)
        return RAW[k]
    return RAW['empservicereal']


# ══════════════════════════════════════ [85] 자료는 어디에 있는가
#
# **규칙은 한 줄입니다.**
#     p_2026icml/  = 원자료. 읽기만 합니다. 우리가 만들지 않습니다.
#     confieee2026/ = 우리가 만든 것 전부. 지워도 다시 만들어집니다.
#
# `data*` 로 시작하는 폴더가 여섯 개나 되어 헷갈리므로, 모든 노트북 §0 에서
# 이 표를 찍습니다. 「이 파일이 원자료인가 산출물인가」를 매번 눈으로 확인하게 합니다.

_DATA_MAP = [
    # (구분, 경로, 무엇, 만드는 것)
    ('원자료', DIR_DATARAW,                                  '고용보험 원본 CSV 5개', ''),
    ('원자료', os.path.join(DIR_RAW, '1IHDP'),               'IHDP 정본', ''),
    ('원자료', os.path.join(DIR_RAW, '5empservicereal'),     '기존 2,000명 코호트', ''),
    # [20] 이것은 원자료가 아닙니다. 옛 실행이 만든 산출물이 여기 놓여 있을 뿐입니다.
    #   원자료 행세를 하게 두면 「CSV 에서 분석파일까지」를 재현할 수 없습니다.
    ('옛산출', os.path.join(DIR_RAW, '_pkl_source'),         'pkl(2026-01 옛 실행) — 지금은 안 씁니다', ''),
    ('산출물', DIR_REF,                                      '벤치마크 정본(내려받음)', 'run §1.7'),
    ('산출물', DIR_REAL,                                     '원자료 pkl · 코호트', 'datagen'),
    ('산출물', DIR_SYNT,                                     '합성 3종', 'datagen'),
    ('산출물', DIR_STD,                                      '표준 npz  ← 학습이 읽는 것', 'run §1'),
    ('산출물', DIR_RUNS,                                     '실행 하나가 폴더 하나', 'run §7'),
    ('산출물', DIR_OUT,                                      '표·요약 csv', 'run'),
    ('산출물', DIR_PAPER,                                    '그림·표', 'run §8'),
]


RAW5_MARK = 's01QT_Neteza__VW_TB_EW_IPP_1022DW_01__201501_202412.csv'


def raw5_dir():
    """
    고용보험 원본 CSV 5개가 있는 폴더.

    16판까지 생성기가 스스로 `dataraw/` 와 `data/5empservicereal/` 을 뒤졌습니다.
    경로를 아는 곳이 둘이 되면 한쪽만 고쳐지므로 여기로 옮겼습니다.
    """
    for d in (DIR_DATARAW, os.path.join(DIR_RAW, '5empservicereal')):
        if os.path.exists(os.path.join(d, RAW5_MARK)):
            return d + os.sep
    return DIR_DATARAW + os.sep          # 없으면 안내용으로 이 값


def _dirsize(path):
    if not os.path.isdir(path):
        return None
    n = tot = 0
    for root, _, fs in os.walk(path):
        for f in fs:
            try:
                tot += os.path.getsize(os.path.join(root, f)); n += 1
            except OSError:
                pass
    return n, tot


def show_data(verbose=True):
    """자료가 어디에 있는지 한 장으로. 노트북 §0 에서 부릅니다."""
    rows = []
    for kind, path, what, made in _DATA_MAP:
        sz = _dirsize(path)
        if sz is None:
            nfile, size = '-', '없음'
        else:
            nfile = sz[0]
            size = ('%.1f GB' % (sz[1] / 1e9)) if sz[1] > 1e9 else ('%.0f MB' % (sz[1] / 1e6))
        rows.append({'구분': kind, '폴더': os.path.relpath(path, PROJECT_ROOT),
                     '내용': what, '만드는곳': made, '파일': nfile, '크기': size})
    if verbose:
        print('자료 위치  —  %s' % PROJECT_ROOT)
        print('  원자료 = 읽기만 합니다  ·  산출물 = 지워도 다시 만들어집니다  ·  옛산출 = 안 씁니다')
        print()
        print('  %-4s %-26s %-9s %6s  %s' % ('구분', '폴더', '크기', '파일', '내용'))
        for r in rows:
            made = ('  (%s 가 만듦)' % r['만드는곳']) if r['만드는곳'] else ''
            print('  %-4s %-26s %-9s %6s  %s%s'
                  % (r['구분'], r['폴더'], r['크기'], r['파일'], r['내용'], made))
        print()
        print('  실데이터 코호트: %s' % (REAL_COHORT or '기존 2,000명'))
        print('    → %s' % real_path())
    return rows
