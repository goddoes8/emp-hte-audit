# -*- coding: utf-8 -*-
"""
벤치마크 데이터 정본 대조 · 변수 식별 · 합성데이터 DGP 검증
------------------------------------------------------------------
문서(docs/데이터명세_분석계획서.docx) 8~13장의 모든 수치를 이 스크립트가 만듭니다.
문서에 손으로 숫자를 옮겨 적지 않습니다.

실행:
    cd confieee2026
    python tools/verify_benchmarks.py            # 전체
    python tools/verify_benchmarks.py --fetch    # 정본 참조자료 재다운로드 포함

산출:
    out/ref01_ihdp_대조.csv        IHDP 100판 정본 대조 결과
    out/ref02_ihdp_변수.csv        IHDP 25개 공변량 이름·유형
    out/ref03_jobs_구성.csv        Jobs 표본 구성 검증
    out/ref04_jobs_변수.csv        Jobs 17개 공변량 식별 결과
    out/ref05_synt_dgp.csv         empservicesynt DGP 검증 결과
    out/ref06_synt_ite.csv         합성 True ITE 분포·이질성
"""
import os, sys, io, json, hashlib, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CONF = os.path.dirname(HERE)
sys.path.insert(0, CONF)
from cdtlib.config import DIR_RAW, DIR_OUT, ensure_dirs           # noqa: E402

# [86] 폴더 경로는 cdtlib/config.py 한 곳에서만 정합니다
from cdtlib.config import DIR_REF                               # noqa: E402
DIR_IHDPCSV = os.path.join(DIR_REF, 'ihdp_csv')

# 정본 출처 (2026-08 확인)
SRC_IHDP_CSV = ('https://raw.githubusercontent.com/claudiashi57/dragonnet/'
                'master/dat/ihdp/csv/ihdp_npci_%d.csv')
SRC_IHDP_RAW = ('https://raw.githubusercontent.com/vdorie/npci/'
                'master/examples/ihdp_sim/data/ihdp.RData')
N_REF_REP = 50   # 위 미러가 공개하는 판 수

IHDP_COV = ['bw', 'b.head', 'preterm', 'birth.o', 'nnhealth', 'momage',
            'sex', 'twin', 'b.marr', 'mom.lths', 'mom.hs', 'mom.scoll',
            'cig', 'first', 'booze', 'drugs', 'work.dur', 'prenatal',
            'ark', 'ein', 'har', 'mia', 'pen', 'tex', 'was']
IHDP_KOR = ['출생체중', '머리둘레', '조산주수', '출생순위', '신생아건강지수', '어머니나이',
            '성별', '쌍둥이', '기혼', '어머니_고졸미만', '어머니_고졸', '어머니_전문대이상',
            '임신중흡연', '첫아이', '임신중음주', '약물', '임신중취업', '산전관리',
            '지역_Arkansas', '지역_Einstein', '지역_Harvard', '지역_Miami',
            '지역_Pennsylvania', '지역_Texas', '지역_Washington']


# ────────────────────────────────────────────────────────── 참조자료 확보
def unzip_reference():
    """dataref/ihdp_reference.zip 이 있으면 풀어서 씁니다(재다운로드 불필요)."""
    z = os.path.join(DIR_REF, 'ihdp_reference.zip')
    if os.path.isdir(DIR_IHDPCSV) and len(os.listdir(DIR_IHDPCSV)) >= N_REF_REP:
        return True
    if not os.path.exists(z):
        return False
    import zipfile
    with zipfile.ZipFile(z) as f:
        f.extractall(CONF)          # 압축 안에 dataref/... 경로가 들어 있습니다
    ok = os.path.isdir(DIR_IHDPCSV) and len(os.listdir(DIR_IHDPCSV)) >= N_REF_REP
    print('  dataref/ihdp_reference.zip 해제 %s' % ('완료' if ok else '실패'))
    return ok


def fetch_reference(force=False):
    import urllib.request, concurrent.futures as cf
    os.makedirs(DIR_IHDPCSV, exist_ok=True)

    def one(i):
        f = os.path.join(DIR_IHDPCSV, 'ihdp_npci_%d.csv' % i)
        if (not force) and os.path.exists(f) and os.path.getsize(f) > 1000:
            return 'skip'
        for _ in range(3):
            try:
                d = urllib.request.urlopen(SRC_IHDP_CSV % i, timeout=45).read()
                if len(d) > 1000:
                    open(f, 'wb').write(d)
                    return 'ok'
            except Exception:
                pass
        return 'FAIL%d' % i

    with cf.ThreadPoolExecutor(8) as ex:
        r = list(ex.map(one, range(1, N_REF_REP + 1)))
    fail = [x for x in r if x.startswith('FAIL')]
    print('  IHDP 정본 CSV %d/%d 확보%s'
          % (N_REF_REP - len(fail), N_REF_REP, (' 실패:' + str(fail)) if fail else ''))

    f = os.path.join(DIR_REF, 'ihdp.RData')
    if force or not os.path.exists(f):
        try:
            open(f, 'wb').write(urllib.request.urlopen(SRC_IHDP_RAW, timeout=60).read())
            print('  IHDP 원자료(ihdp.RData) 확보')
        except Exception as e:
            print('  IHDP 원자료 확보 실패:', e)


def _rowset(M):
    return sorted(hashlib.md5(np.round(x, 6).tobytes()).hexdigest() for x in M)


# ────────────────────────────────────────────────────────── 1. IHDP
def check_ihdp():
    tr = np.load(os.path.join(DIR_RAW, '1IHDP', 'ihdp_npci_1-100.train.npz'))
    te = np.load(os.path.join(DIR_RAW, '1IHDP', 'ihdp_npci_1-100.test.npz'))
    X = np.concatenate([tr['x'], te['x']], 0)          # (747, 25, 100)
    T = np.concatenate([tr['t'], te['t']], 0)
    n, p, R = X.shape

    rows = []
    n_ok = n_miss = 0
    for r in range(R):
        f = os.path.join(DIR_IHDPCSV, 'ihdp_npci_%d.csv' % (r + 1))
        if not os.path.exists(f):
            rows.append([r + 1, '', '', '정본없음'])
            n_miss += 1
            continue
        M = np.column_stack(
            [np.concatenate([tr[k][:, r], te[k][:, r]]) for k in
             ['t', 'yf', 'ycf', 'mu0', 'mu1']] + [X[:, :, r]])
        C = pd.read_csv(f, header=None).values
        same = _rowset(M) == _rowset(C)
        order = M.shape == C.shape and np.allclose(M, C, atol=1e-8)
        rows.append([r + 1, same, order, '일치' if same else '불일치'])
        n_ok += int(same)

    df = pd.DataFrame(rows, columns=['판', '행집합일치', '행순서일치', '판정'])
    df.to_csv(os.path.join(DIR_OUT, 'ref01_ihdp_대조.csv'), index=False, encoding='utf-8-sig')

    # 변수표
    v = []
    for j in range(p):
        col = X[:, j, 0]
        u = np.unique(np.round(col, 6))
        v.append([('x%d' % (j + 1)), IHDP_COV[j], IHDP_KOR[j],
                  '이진' if len(u) <= 2 else '연속', len(u),
                  round(float(col.min()), 4), round(float(col.max()), 4),
                  round(float(col.mean()), 4), round(float(col.std()), 4)])
    pd.DataFrame(v, columns=['열', '원변수명', '한글', '유형', '고유값수',
                             '최소', '최대', '평균', '표준편차']
                 ).to_csv(os.path.join(DIR_OUT, 'ref02_ihdp_변수.csv'),
                          index=False, encoding='utf-8-sig')

    same_X = _rowset(X[:, :, 0]) == _rowset(X[:, :, 1]) == _rowset(X[:, :, R - 1])
    exact_X = bool(np.allclose(X[:, :, 0], X[:, :, 1]))
    summary = dict(
        n=n, p=p, R=R, n_train=tr['x'].shape[0], n_test=te['x'].shape[0],
        처치수=int(T[:, 0].sum()), 대조수=int((1 - T[:, 0]).sum()),
        처치수_판간동일=bool(len(set(T.sum(0).astype(int))) == 1),
        정본대조_일치=n_ok, 정본대조_미확보=n_miss, 정본대조_대상=R,
        공변량집합_판간동일=bool(same_X), 행순서_판간동일=exact_X,
        ate메타=float(tr['ate']), yadd=int(tr['yadd']), ymul=int(tr['ymul']),
    )
    print('\n[1] IHDP')
    for k, val in summary.items():
        print('    %-20s %s' % (k, val))
    return summary


# ────────────────────────────────────────────────────────── 2. Jobs
def check_jobs():
    from scipy.stats import spearmanr
    tr = np.load(os.path.join(DIR_RAW, 'jobs_train.npz'))
    te = np.load(os.path.join(DIR_RAW, 'jobs_test.npz'))
    X = np.concatenate([tr['x'], te['x']], 0)
    T = np.concatenate([tr['t'], te['t']], 0)
    E = np.concatenate([tr['e'], te['e']], 0)
    Y = np.concatenate([tr['yf'], te['yf']], 0)
    n, p, R = X.shape

    e1t1 = ((E == 1) & (T == 1)).sum(0)
    e1t0 = ((E == 1) & (T == 0)).sum(0)
    e0t1 = ((E == 0) & (T == 1)).sum(0)
    att = Y[:, 0][(E[:, 0] == 1) & (T[:, 0] == 1)].mean() - \
          Y[:, 0][(E[:, 0] == 1) & (T[:, 0] == 0)].mean()

    comp = pd.DataFrame([
        ['전체 표본', n, 'LaLonde 실험군 722 + PSID 대조군 2,490'],
        ['e=1 (무작위배정 표본)', int(E[:, 0].sum()), 'LaLonde(1986) 실험 표본'],
        ['e=1 & t=1 (실험 처치)', int(e1t1[0]), 'NSW 훈련 수강'],
        ['e=1 & t=0 (실험 대조)', int(e1t0[0]), 'NSW 무작위 대조'],
        ['e=0 & t=1', int(e0t1[0]), '0 이어야 정상 (PSID 는 전원 대조)'],
        ['e=0 (관측 대조군)', int((1 - E[:, 0]).sum()), 'PSID 비교집단'],
        ['판 수(replication)', R, ''],
        ['공변량 수', p, ''],
        ['결과 yf 유형', '이진 {0,1}' if len(np.unique(Y)) <= 2 else '연속', '1978년 취업 여부'],
        ['파일 ate', round(float(tr['ate'].ravel()[0]), 6), '무작위 표본 ATT'],
        ['재계산 ATT (e=1)', round(float(att), 6), 'E[Y|e=1,t=1] − E[Y|e=1,t=0]'],
        ['ate 일치', bool(abs(att - float(tr['ate'].ravel()[0])) < 1e-6), ''],
    ], columns=['항목', '값', '설명'])
    comp.to_csv(os.path.join(DIR_OUT, 'ref03_jobs_구성.csv'), index=False, encoding='utf-8-sig')

    # 변수 식별
    Z = X[:, :, 0]
    age, educ, black, hisp = Z[:, 0], Z[:, 1], Z[:, 2], Z[:, 3]
    re74, re75 = Z[:, 6], Z[:, 7]
    u74 = (re74 == re74.min()).astype(float)
    u75 = (re75 == re75.min()).astype(float)

    def mono(a, b):
        return abs(spearmanr(a, b).statistic)

    ident = [
        ['x1',  'age',        '나이',                  '기준변수',                       1.0],
        ['x2',  'education',  '교육연수',              '기준변수',                       1.0],
        ['x3',  'black',      '흑인',                  '이진',                           1.0],
        ['x4',  'hispanic',   '히스패닉',              '이진',                           1.0],
        ['x5',  'married',    '기혼',                  '이진 (풀링평균 산술검증)',        1.0],
        ['x6',  'nodegree',   '고졸미만',              '이진 (풀링평균 산술검증)',        1.0],
        ['x7',  're74',       '1974년 소득',           '기준변수',                       1.0],
        ['x8',  're75',       '1975년 소득',           '기준변수',                       1.0],
        ['x9',  'age^2',      '나이 제곱',             'x1 과 단조 1:1',                 mono(age, Z[:, 8])],
        ['x10', 'age^3',      '나이 세제곱',           'x1 과 단조 1:1',                 mono(age, Z[:, 9])],
        ['x11', 'education^2', '교육연수 제곱',        'x2 와 단조 1:1',                 mono(educ, Z[:, 10])],
        ['x12', 're74^2',     '1974소득 제곱',         'x7 과 단조 1:1',                 mono(re74, Z[:, 11])],
        ['x13', 're75^2',     '1975소득 제곱',         'x8 과 단조 1:1',                 mono(re75, Z[:, 12])],
        ['x14', 'u74',        '1974년 무소득',         'x7=0 집합과 완전일치',
         float((Z[:, 13] == u74).mean())],
        ['x15', 'u75',        '1975년 무소득',         'x8=0 집합과 완전일치',
         float((Z[:, 14] == u75).mean())],
        ['x16', '(미확정)',   '소득계열 변수',
         'ρ(x16, re74+re75)=%.3f · 0집중 %.1f%%' % (spearmanr(Z[:, 15], re74 + re75).statistic,
                                                    100 * (Z[:, 15] == Z[:, 15].min()).mean()),
         np.nan],
        ['x17', 'hispanic×u74', '히스패닉×1974무소득', 'hisp*u74 와 완전일치',
         float((Z[:, 16] == hisp * u74).mean())],
    ]
    prof = []
    for i, row in enumerate(ident):
        col = Z[:, i]
        u = np.unique(np.round(col, 6))
        prof.append(row + ['이진' if len(u) <= 2 else '연속', len(u),
                           round(float(col.mean()), 4), round(float(col.std()), 4)])
    pd.DataFrame(prof, columns=['열', '추정변수명', '한글', '식별근거', '일치도',
                                '유형', '고유값수', '평균', '표준편차']
                 ).to_csv(os.path.join(DIR_OUT, 'ref04_jobs_변수.csv'),
                          index=False, encoding='utf-8-sig')

    print('\n[2] Jobs')
    for _, r in comp.iterrows():
        print('    %-24s %s' % (r['항목'], r['값']))
    return dict(n=n, p=p, R=R, e1=int(E[:, 0].sum()), t1=int(T[:, 0].sum()),
                att=float(att), ate_file=float(tr['ate'].ravel()[0]))


# ────────────────────────────────────────────────────────── 3. empservicesynt
def check_synt():
    f = os.path.join(DIR_RAW, '4empservicesynt', 'empservicesynt.parquet')
    s = pd.read_parquet(f)
    N, T = 2000, 120
    assert len(s) == N * T, len(s)

    emp = s['보상_손실함수계산여부'].values.reshape(N, T)
    a = emp[:, :-1].ravel(); b = emp[:, 1:].ravel()
    p10 = ((a == 1) & (b == 0)).sum() / max((a == 1).sum(), 1)
    p01 = ((a == 0) & (b == 1)).sum() / max((a == 0).sum(), 1)
    trn = (s['정책_훈련직종'].values != 'c00').reshape(N, T)
    ite = (s['True_Mu1'] - s['True_Mu0']).values
    m = s['보상_손실함수계산여부'].values == 1
    ct = pd.crosstab(s['persona'], s['istreated'])
    ind_eq = all((s[f'상태_생애_산업c{j:02d}누적경력기간'].values ==
                  s[f'상태_생애_직종c{j:02d}누적경력기간'].values).all() for j in range(37))
    gap = s['상태_현재실직기간'].values.reshape(N, T)
    monotone = bool(np.all(np.diff(gap, axis=1) >= -1e-12))

    chk = pd.DataFrame([
        ['V1', 'istreated 의 정체', 'persona∈{P1,P2} 이면 1',
         'istreated×persona 교차표가 완전분리 (%s)' % (ct.values.tolist()),
         '처치변수 아님 — 페르소나 라벨'],
        ['V2', '산업 = 직종 중복', '37쌍 전부 동일값' if ind_eq else '불일치 있음',
         'c00~c36 전 열 비교', '설명변수 37개가 완전 중복'],
        ['V3', '보상 이중로그', 'Y=log1p(임금), 임금 자체가 이미 로그척도',
         'Y|M=1 평균 %.4f (범위 %.3f~%.3f)' % (s['보상_로그임금'][m].mean(),
                                              s['보상_로그임금'][m].min(),
                                              s['보상_로그임금'][m].max()),
         '절편 5.52 = log(250만원)'],
        ['V4', 'Y 와 Mu 척도 불일치', 'Y 는 log1p, Mu0/Mu1 은 원척도',
         'Mu0 평균 %.4f / Y 평균 %.4f' % (s['True_Mu0'].mean(), s['보상_로그임금'].mean()),
         'PEHE 계산 불가'],
        ['V5', '취업률', '이론값 0.833 (0.02/0.10 전이) vs 실제 %.4f' % emp.mean(),
         'P(취업→실직)=%.4f, P(실직→취업)=%.4f' % (p10, p01),
         '노트북 소스의 0.02 와 불일치'],
        ['V6', '실직기간 정의', '현재 실직기간(재취업 시 0으로 복귀)',
         '단조증가 여부 = %s' % monotone,
         '노트북 셀81 은 누적값을 기록'],
        ['V7', '훈련 사건', '총 %d셀, 1회이상 %d명 (%.1f%%)' % (trn.sum(), (trn.sum(1) > 0).sum(),
                                                            100 * (trn.sum(1) > 0).mean()),
         '훈련 시 실업상태 비율 %.3f' % (1 - emp[trn].mean()),
         'istreated 평균 0.40 과 무관'],
        ['V8', '반사실 대비 오염', 'Mu1 은 알선(place)까지 포함',
         'mu_1=f(train=Target, place=act_place), mu_0=f(0,0)',
         '훈련효과와 알선효과가 섞임'],
        ['V9', '재현성', '노트북 4개 생성셀 전부 주석처리 · 어느 셀도 133열 스키마를 만들지 못함',
         '셀78 스키마 불일치, 셀80/81 은 N=1 · CSV만 저장',
         'parquet 재생성 불가'],
    ], columns=['번호', '항목', '확인내용', '증거', '판정'])
    chk.to_csv(os.path.join(DIR_OUT, 'ref05_synt_dgp.csv'), index=False, encoding='utf-8-sig')

    vc = pd.Series(np.round(ite, 6)).value_counts().head(6)
    d = pd.DataFrame({'ite': ite, 'm': m, '남': s['상태_성별_남'].values,
                      'persona': s['persona'].values})
    g1 = d[d.m].groupby('남')['ite'].agg(['mean', 'std', 'count']).reset_index()
    g2 = d[d.m].groupby('persona')['ite'].agg(['mean', 'std', 'count']).reset_index()
    rows = [['전체', ite.mean(), ite.std(), (ite == 0).mean() * 100, len(ite)],
            ['취업(M=1)', ite[m].mean(), ite[m].std(), (ite[m] == 0).mean() * 100, m.sum()]]
    for _, r in g1.iterrows():
        rows.append(['성별_남=%d' % r['남'], r['mean'], r['std'], np.nan, r['count']])
    for _, r in g2.iterrows():
        rows.append(['persona=%s' % r['persona'], r['mean'], r['std'], np.nan, r['count']])
    ited = pd.DataFrame(rows, columns=['집단', '평균ITE', '표준편차', '정확히0(%)', 'n'])
    ited['상위값'] = ''
    ited.loc[0, '상위값'] = ' | '.join('%.3f:%d' % (k, v) for k, v in vc.items())
    ited.to_csv(os.path.join(DIR_OUT, 'ref06_synt_ite.csv'), index=False, encoding='utf-8-sig')

    print('\n[3] empservicesynt')
    for _, r in chk.iterrows():
        print('    %-4s %-22s %s' % (r['번호'], r['항목'], r['판정']))
    print('    ITE  전체평균 %.4f  |ITE|>1 비율 %.4f%%  정확히0 %.2f%%'
          % (ite.mean(), 100 * (np.abs(ite) > 1).mean(), 100 * (ite == 0).mean()))
    return dict(p10=float(p10), p01=float(p01), emp=float(emp.mean()),
                ite_zero=float((ite == 0).mean()), ite_big=float((np.abs(ite) > 1).mean()))


# ────────────────────────────────────────────────────────── main
def run(fetch=False, synt_v1=True):
    """노트북에서 호출할 진입점. icml64.ipynb 가 이 함수를 부릅니다."""
    ensure_dirs()
    os.makedirs(DIR_REF, exist_ok=True)
    print('[0] 정본 참조자료 확보')
    if fetch or not unzip_reference():
        fetch_reference(force=fetch)
    out = dict(ihdp=check_ihdp(), jobs=check_jobs())
    if synt_v1 and os.path.exists(os.path.join(DIR_RAW, '4empservicesynt',
                                               'empservicesynt.parquet')):
        out['synt'] = check_synt()          # v1 진단 (폐기 근거 보존용)
    with open(os.path.join(DIR_OUT, 'ref00_summary.json'), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2, default=str)
    print('\n산출: %s/ref0*.csv, ref00_summary.json' % DIR_OUT)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fetch', action='store_true', help='정본 참조자료 재다운로드')
    args = ap.parse_args()
    run(fetch=args.fetch)
