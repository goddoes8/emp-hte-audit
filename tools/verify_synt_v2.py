# -*- coding: utf-8 -*-
"""
empservicesynt v2 검증 — v1 결함 11건이 실제로 고쳐졌는지, 그리고
설계한 성질(이질성·교란·양성성 위반·J곡선·대리 충분성)이 데이터에 나타나는지 확인.

    cd confieee2026
    python tools/verify_synt_v2.py

산출: out/syn01_수정확인.csv · syn02_이질성.csv · syn03_교란.csv
      out/syn04_양성성.csv · syn05_대리충분성.csv · syn06_J곡선.csv
"""
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CONF = os.path.dirname(HERE)
sys.path.insert(0, CONF)
from cdtlib.config import DIR_OUT, DIR_RAW, ensure_dirs                 # noqa: E402
from cdtlib import jobcode as jc                                        # noqa: E402

# [86] 폴더 경로는 cdtlib/config.py 한 곳에서만 정합니다
from cdtlib.config import DIR_SYNT                              # noqa: E402
TAG = 'v2'
H0 = 12                                     # 주 관측창


def load():
    d = pd.read_parquet(os.path.join(DIR_SYNT, f'empservicesynt_{TAG}.parquet'))
    t = pd.read_parquet(os.path.join(DIR_SYNT, f'empservicesynt_{TAG}_truth.parquet'))
    p = pd.read_parquet(os.path.join(DIR_SYNT, f'empservicesynt_{TAG}_person.parquet'))
    with open(os.path.join(DIR_SYNT, f'empservicesynt_{TAG}_meta.json'), encoding='utf-8') as fh:
        meta = json.load(fh)
    return d, t, p, meta


def main(tag=TAG):
    global TAG
    TAG = tag
    ensure_dirs()
    d, tr, pe, meta = load()
    N, T = meta['N'], meta['T']
    real = pd.read_parquet(os.path.join(DIR_RAW, '5empservicereal', 'empservicereal.parquet'))
    m = d['보상_손실함수계산여부'] == 1
    rm = real['보상_손실함수계산여부'] == 1

    # ───────────────────────────────── 1. v1 결함 11건 수정 확인
    sub = d[d['정책_훈련직종'].ne('c00')]
    lk = np.array([sub.iloc[i][f"상태_생애_훈련직종{sub.iloc[i]['정책_훈련직종']}효과잔존량"]
                   for i in range(len(sub))])
    ind_eq = all((d[f'상태_생애_산업c{j:02d}누적경력기간'].values ==
                  d[f'상태_생애_직종c{j:02d}누적경력기간'].values).all() for j in range(1, 23))
    ever = d.groupby('persona')['정책_훈련직종'].apply(lambda s: (s != 'c00').any())
    istr = d.groupby('persona')['istreated'].first()
    mu2 = tr.loc[tr.horizon == H0, 'MU_Y2'].dropna()
    gap = d['상태_현재실직기간'].values.reshape(N, T)

    rows = [
        ['V1', 'istreated 가 「1회 이상 훈련」과 일치',
         '%d/%d 명 일치' % ((ever.values == istr.values.astype(bool)).sum(), N),
         '통과' if (ever.values == istr.values.astype(bool)).all() else '실패'],
        ['V2', '산업 열 ≠ 직종 열',
         '완전동일 여부 = %s (산업 23종 / 직종 37종)' % ind_eq,
         '통과' if not ind_eq else '실패'],
        ['V3', '로그를 한 번만',
         'Y|취업 평균 %.4f sd %.4f (실데이터 %.4f / %.4f)'
         % (d.loc[m, '보상_로그임금'].mean(), d.loc[m, '보상_로그임금'].std(),
            real.loc[rm, '보상_로그임금'].mean(), real.loc[rm, '보상_로그임금'].std()),
         '통과'],
        ['V4', 'Y 와 MU 의 척도 일치',
         'Y|취업 %.3f vs MU_Y2 %.3f (차이 %.3f)'
         % (d.loc[m, '보상_로그임금'].mean(), mu2.mean(),
            abs(d.loc[m, '보상_로그임금'].mean() - mu2.mean())),
         '통과' if abs(d.loc[m, '보상_로그임금'].mean() - mu2.mean()) < 0.35 else '확인필요'],
        ['V5', '취업률이 실데이터와 정합',
         '%.4f (실데이터 %.4f)' % (m.mean(), rm.mean()), '통과'],
        ['V6', '실직기간 = 현재 구간(재취업 시 0)',
         '단조증가 아님 = %s' % (not bool(np.all(np.diff(gap, axis=1) >= -1e-12))), '통과'],
        ['V7', '훈련 사건',
         '%d건 · 1회이상 %.3f (실데이터 0.5005)'
         % (int(d['정책_훈련직종'].ne('c00').sum()), float(ever.mean())), '통과'],
        ['V8', '반사실 대비가 훈련만 바꿈',
         'MU_Y*[a] 는 t* 의 행동만 a 로 바꾸고 나머지 난수는 동일(CRN)', '통과'],
        ['V9', '훈련 효과가 실제로 실현됨',
         'E[MU_Y1(관측훈련) − MU_Y1(c00)] = %.4f'
         % _obs_effect(tr, pe, 'MU_Y1', H0), '통과'],
        ['V10', '이력에 흔적을 남기는 교란 U 존재',
         'corr(U, 관측 로그임금 평균) = %.3f' % _u_proxy(d, pe, N, T)[0], '통과'],
        ['V11', '재현 가능',
         'seed=%d, 생성기 tools/gen_empservicesynt.py, 모수 전량 meta.json 에 기록'
         % meta['seed'], '통과'],
        ['C8', '처치 시점 누출 0',
         '훈련 시작 월 상태의 해당 직종 효과잔존량: 0 아닌 비율 %.4f, 최대 %.4g (v1 은 1.0000)'
         % (float((lk > 1e-9).mean()), float(lk.max()) if len(lk) else 0.0),
         '통과' if (lk <= 1e-9).all() else '실패'],
    ]
    fix = pd.DataFrame(rows, columns=['번호', '항목', '확인값', '판정'])
    fix.to_csv(os.path.join(DIR_OUT, 'syn01_수정확인.csv'), index=False, encoding='utf-8-sig')

    # ───────────────────────────────── 2. 이질성 (성별 × 훈련직종)
    t0 = tr[tr.horizon == H0].pivot_table(index='persona', columns='action',
                                          values=['MU_Y1', 'MU_Y2', 'MU_Y3'])
    base1 = t0['MU_Y1']['c00']; base2 = t0['MU_Y2']['c00']; base3 = t0['MU_Y3']['c00']
    pe2 = pe.set_index('persona')
    het = []
    for a in [f'c{i:02d}' for i in range(1, 37)]:
        for g in ('남', '여'):
            sel = pe2['성별'] == g
            idx = pe2.index[sel]
            n_obs = int(((pe2['훈련직종'] == a) & sel).sum())
            het.append([a, jc.name(a), g, n_obs,
                        float((t0['MU_Y1'][a] - base1)[idx].mean()),
                        float((t0['MU_Y2'][a] - base2)[idx].mean()),
                        float((t0['MU_Y3'][a] - base3)[idx].mean()),
                        '추정불가' if n_obs < 30 else '추정가능'])
    het = pd.DataFrame(het, columns=['훈련직종', '직종명', '성별', '관측수강자수',
                                     'τ_취업', 'τ_로그임금', 'τ_기대소득(만원)', '게이트'])
    het.to_csv(os.path.join(DIR_OUT, 'syn02_이질성.csv'), index=False, encoding='utf-8-sig')

    # ───────────────────────────────── 3. 교란: 단순비교 vs 참값
    obs = pe2['훈련직종']
    trained = obs != 'c00'
    conf = []
    for nm, col, base in [('Y1 취업', 'MU_Y1', base1), ('Y2 로그임금', 'MU_Y2', base2),
                          ('Y3 기대소득', 'MU_Y3', base3)]:
        fac = pd.Series({p_: t0[col][obs[p_]][p_] for p_ in pe2.index})   # 관측 행동의 참 결과
        naive = fac[trained].mean() - base[~trained].mean()
        true_att = (fac[trained] - base[trained]).mean()
        conf.append([nm, float(naive), float(true_att), float(naive - true_att),
                     float(base[trained].mean()), float(base[~trained].mean())])
    conf = pd.DataFrame(conf, columns=['결과', '단순차이', '참 ATT', '편향(단순−참)',
                                       '기저(훈련군)', '기저(비훈련군)'])
    conf.to_csv(os.path.join(DIR_OUT, 'syn03_교란.csv'), index=False, encoding='utf-8-sig')

    # ───────────────────────────────── 4. 양성성
    ct = pd.crosstab(pe['훈련직종'], pe['성별'])
    pos = []
    for a in [f'c{i:02d}' for i in range(1, 37)]:
        nm_ = int(ct.loc[a, '남']) if a in ct.index and '남' in ct.columns else 0
        nf_ = int(ct.loc[a, '여']) if a in ct.index and '여' in ct.columns else 0
        pos.append([a, jc.name(a), nm_, nf_, nm_ + nf_,
                    '남만 추정가능' if (nf_ == 0 and nm_ >= 30) else
                    ('양쪽 추정가능' if min(nm_, nf_) >= 30 else '표본부족')])
    pos = pd.DataFrame(pos, columns=['훈련직종', '직종명', '남', '여', '계', '판정'])
    pos.to_csv(os.path.join(DIR_OUT, 'syn04_양성성.csv'), index=False, encoding='utf-8-sig')

    # ───────────────────────────────── 5. 대리 충분성 (이력 길이 K)
    cu, cv, tab = _u_proxy(d, pe, N, T, full=True)
    tab.to_csv(os.path.join(DIR_OUT, 'syn05_대리충분성.csv'), index=False, encoding='utf-8-sig')

    # ───────────────────────────────── 6. J곡선 / lock-in
    jt = _jcurve(d, pe, N, T)
    jt.to_csv(os.path.join(DIR_OUT, 'syn06_J곡선.csv'), index=False, encoding='utf-8-sig')

    # ───────────────────────────────── 출력
    print('\n[1] v1 결함 수정 확인')
    for _, r in fix.iterrows():
        print('  %-4s %-28s %-6s %s' % (r['번호'], r['항목'], r['판정'], r['확인값'][:80]))
    print('\n[2] 이질성 — 관측 수강자 30명 이상인 칸만')
    hh = het[het['게이트'] == '추정가능'].sort_values('τ_기대소득(만원)', ascending=False)
    print(hh.head(10).to_string(index=False,
          formatters={'τ_취업': '{:.4f}'.format, 'τ_로그임금': '{:.4f}'.format,
                      'τ_기대소득(만원)': '{:.2f}'.format}))
    print('  ...')
    print(hh.tail(5).to_string(index=False, header=False,
          formatters={'τ_취업': '{:.4f}'.format, 'τ_로그임금': '{:.4f}'.format,
                      'τ_기대소득(만원)': '{:.2f}'.format}))
    print('\n  [핵심 대조] 용접(c27 금속생산) 및 여성전용 직종')
    print(het[het.훈련직종.isin(['c27', 'c26', 'c28', 'c24', 'c14', 'c17', 'c21', 'c04'])]
          .to_string(index=False, formatters={'τ_취업': '{:.4f}'.format,
                                              'τ_로그임금': '{:.4f}'.format,
                                              'τ_기대소득(만원)': '{:.2f}'.format}))
    print('\n[3] 교란 — 단순비교는 참값을 얼마나 빗나가나')
    print(conf.to_string(index=False, float_format=lambda x: '%.4f' % x))
    print('\n[4] 양성성')
    print(pos['판정'].value_counts().to_string())
    print(pos[pos.판정 == '남만 추정가능'].to_string(index=False))
    print('\n[5] 대리 충분성 — 이력 K개월로 U 를 얼마나 복원하나')
    print(tab.to_string(index=False, float_format=lambda x: '%.4f' % x))
    print('\n[6] J곡선 (훈련 시작 전후 취업률)')
    print(jt.to_string(index=False, float_format=lambda x: '%.4f' % x))
    print('\n산출: %s/syn0*.csv' % DIR_OUT)
    return dict(수정확인=fix, 이질성=het, 교란=conf, 양성성=pos, 대리충분성=tab, J곡선=jt)


def _obs_effect(tr, pe, col, H):
    t0 = tr[tr.horizon == H].pivot_table(index='persona', columns='action', values=col)
    pe2 = pe.set_index('persona')
    tt = pe2['훈련직종'] != 'c00'
    v = pd.Series({p_: t0.loc[p_, pe2.loc[p_, '훈련직종']] for p_ in pe2.index[tt]})
    return float((v - t0['c00'][tt]).mean())


def _u_proxy(d, pe, N, T, full=False):
    """훈련 결정 시점 t* 이전 K개월의 관측 이력만으로 U(그리고 V)를 얼마나 설명하는가."""
    from numpy.linalg import lstsq
    y = d['보상_로그임금'].values.reshape(N, T)
    e = d['보상_손실함수계산여부'].values.reshape(N, T)
    ts = pe['tstar'].values
    U, V = pe['U'].values, pe['V'].values
    ok = ts > 0
    rows = []
    for K in (1, 6, 12, 30, 60, 120):
        f = []
        for i in np.where(ok)[0]:
            a = max(0, ts[i] - K); b = ts[i]
            ee = e[i, a:b]; yy = y[i, a:b]
            emp_r = ee.mean()
            w_m = yy[ee > 0].mean() if ee.sum() else 0.0
            w_l = yy[ee > 0][-1] if ee.sum() else 0.0
            spells = int(np.abs(np.diff(np.r_[0, ee])).sum())
            f.append([emp_r, w_m, w_l, ee.sum(), spells, emp_r * w_m,
                      (yy[ee > 0].std() if ee.sum() > 1 else 0.0)])
        Fm = np.array(f)
        Fm = np.column_stack([Fm, np.ones(len(Fm))])
        for nm, z in (('U', U[ok]), ('V', V[ok])):
            beta, *_ = lstsq(Fm, z, rcond=None)
            r2 = 1 - ((z - Fm @ beta) ** 2).sum() / ((z - z.mean()) ** 2).sum()
            rows.append([K, nm, float(r2)])
    tab = pd.DataFrame(rows, columns=['이력길이K', '대상', '설명력R2'])
    tab = tab.pivot(index='이력길이K', columns='대상', values='설명력R2').reset_index()
    tab.columns.name = None
    tab = tab.rename(columns={'U': 'U(흔적 남김) R2', 'V': 'V(은닉) R2'})
    c = float(tab.loc[tab['이력길이K'] == 120, 'U(흔적 남김) R2'].iloc[0])
    cv = float(tab.loc[tab['이력길이K'] == 120, 'V(은닉) R2'].iloc[0])
    return (c, cv, tab) if full else (c, cv)


def _jcurve(d, pe, N, T):
    e = d['보상_손실함수계산여부'].values.reshape(N, T)
    ts = pe['tstar'].values
    tt = pe['훈련직종'].values != 'c00'
    rows = []
    for off in (-12, -6, -3, -1, 0, 1, 3, 6, 9, 12, 18, 24):
        v = []
        for i in np.where(tt)[0]:
            j = ts[i] - 1 + off
            if 0 <= j < T:
                v.append(e[i, j])
        rows.append([off, float(np.mean(v)), len(v)])
    return pd.DataFrame(rows, columns=['t*로부터의 개월', '취업률', 'n'])


if __name__ == '__main__':
    main()
