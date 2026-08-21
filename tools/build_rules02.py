# -*- coding: utf-8 -*-
"""
synthetic_데이타생성규칙02.xlsx 생성기
=====================================
규칙표 01 을 **확대**합니다. P1~P5 의 기존 12개 행은 값이 그대로이고,
새 행 4개는 P1~P5 에서 전부 0 이므로 **01 과 완전히 같은 데이터가 나옵니다**(하위호환).

새로 넣은 것
------------
① 변수 4개 (모든 페르소나 공통)
   · Gender_Female     여성 임금격차 (수준)
   · Age_Year          나이 1세당 임금 (수준)
   · Act_Train_Female  여성일 때 **훈련효과**에 더하는 값      ← 성별 이질성
   · Act_Train_Age     나이 1세당 **훈련효과** 변화            ← 연령 이질성

② 페르소나 3종
   · P6 성별 민감형 (금속·용접)   타겟 Job27. 남성 +60만원, **여성 0만원**. 여성 비율 2%
   · P7 연령 민감형 (IT·플랫폼)   타겟 Job04. 30세 +70만원, 1세당 −5만원 → 44세면 소멸
   · P8 성별 민감형 (보건·돌봄)   타겟 Job14. 남성 +15만원, **여성 +55만원**. 여성 비율 65%

   P6 은 「참값은 0 인데 표본이 없어 추정할 수 없는 칸」,
   P8 은 「성별로 효과가 갈리는데 표본이 있어 추정할 수 있는 칸」입니다.
   둘이 짝을 이루어야 양성성 게이트가 제대로 작동하는지 채점할 수 있습니다.

③ Sheet2 — 모집단과 공정
   페르소나별 인구비중·여성비율·연령범위, 그리고 고용 전이·훈련 수강 확률.
   01 에서는 이 값들이 코드에 숨어 있었습니다. 수강 확률은 **실데이터에 맞춰 다시 잡았습니다**.

계수 규약 (01 과 동일)
----------------------
    계수 = log( (기저만원 + 증가만원) / 기저만원 ),   Intercept 계수 = log(기저만원)

교호항은 「그 조건에서의 훈련 임금증가」를 기준으로 로그 차이를 씁니다.
    Act_Train_Female 계수 = log((기저+여성증가)/기저) − log((기저+남성증가)/기저)
    Act_Train_Age    계수 = log((기저+남성증가+1세당)/기저) − log((기저+남성증가)/기저)

    python tools/build_rules02.py
"""
import os, math
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

HERE = os.path.dirname(os.path.abspath(__file__))
CONF = os.path.dirname(HERE)
OUT = os.path.join(CONF, 'docs', 'synthetic_데이타생성규칙02.xlsx')

# ── 페르소나 정의: (이름, 기저만원, 타겟Job, {변수: (증가만원, 증가만원sd)})
#    Act_Train_Female = 여성일 때의 훈련 임금증가(만원)
#    Act_Train_Age    = 훈련효과의 1세당 변화(만원)
P = {
 'P1': ('고효율 훈련형 (AI/개발)', 250, 10, dict(
        Residual=(20, 20), Act_Train_Code=(50, 5), Act_Train_Female=(50, 5), Act_Train_Age=(0, 0),
        Act_Placement=(0, 0), Time_Year=(3, 2), Exp_Matched=(30, 10), NotExp=(0, 0),
        Exp_General=(0, 0), Decay=(-2, 2), Company_Size=(20, 10), Contract_Type=(40, 10),
        Gap_Duration=(-5, 2), Gender_Female=(0, 0), Age_Year=(0, 0))),
 'P2': ('알선 민감형 (플랫폼/서비스)', 200, 20, dict(
        Residual=(20, 20), Act_Train_Code=(20, 3), Act_Train_Female=(20, 3), Act_Train_Age=(0, 0),
        Act_Placement=(10, 3), Time_Year=(3, 2), Exp_Matched=(10, 10), NotExp=(2, 2),
        Exp_General=(0, 0), Decay=(-2, 2), Company_Size=(20, 10), Contract_Type=(40, 10),
        Gap_Duration=(-5, 2), Gender_Female=(0, 0), Age_Year=(0, 0))),
 'P3': ('저효율 훈련형 (일반사무)', 200, 11, dict(
        Residual=(20, 20), Act_Train_Code=(10, 5), Act_Train_Female=(10, 5), Act_Train_Age=(0, 0),
        Act_Placement=(0, 0), Time_Year=(3, 2), Exp_Matched=(10, 5), NotExp=(10, 10),
        Exp_General=(0, 0), Decay=(-2, 2), Company_Size=(20, 10), Contract_Type=(40, 10),
        Gap_Duration=(-5, 2), Gender_Female=(0, 0), Age_Year=(0, 0))),
 'P4': ('경력 중시형 (제조업 숙련공)', 220, 5, dict(
        Residual=(20, 20), Act_Train_Code=(5, 5), Act_Train_Female=(5, 5), Act_Train_Age=(0, 0),
        Act_Placement=(3, 3), Time_Year=(3, 2), Exp_Matched=(40, 10), NotExp=(0, 0),
        Exp_General=(0, 0), Decay=(-2, 2), Company_Size=(20, 10), Contract_Type=(40, 10),
        Gap_Duration=(-20, 10), Gender_Female=(0, 0), Age_Year=(0, 0))),
 'P5': ('낙인 취약형 (경력단절 위험군)', 180, 15, dict(
        Residual=(30, 30), Act_Train_Code=(5, 5), Act_Train_Female=(5, 5), Act_Train_Age=(0, 0),
        Act_Placement=(0, 0), Time_Year=(3, 2), Exp_Matched=(20, 10), NotExp=(0, 0),
        Exp_General=(0, 0), Decay=(-2, 2), Company_Size=(10, 5), Contract_Type=(20, 10),
        Gap_Duration=(-50, 20), Gender_Female=(0, 0), Age_Year=(0, 0))),
 # ── 신설 ─────────────────────────────────────────────────────────────
 'P6': ('성별 민감형 (금속·용접)', 240, 27, dict(
        Residual=(25, 25), Act_Train_Code=(60, 8), Act_Train_Female=(0, 0), Act_Train_Age=(-1, 0.5),
        Act_Placement=(5, 3), Time_Year=(3, 2), Exp_Matched=(35, 10), NotExp=(0, 0),
        Exp_General=(0, 0), Decay=(-2, 2), Company_Size=(25, 10), Contract_Type=(45, 10),
        Gap_Duration=(-15, 8), Gender_Female=(-40, 10), Age_Year=(2, 1))),
 'P7': ('연령 민감형 (IT·플랫폼)', 260, 4, dict(
        Residual=(25, 25), Act_Train_Code=(70, 8), Act_Train_Female=(65, 8), Act_Train_Age=(-5, 1),
        Act_Placement=(8, 3), Time_Year=(4, 2), Exp_Matched=(30, 10), NotExp=(5, 5),
        Exp_General=(0, 0), Decay=(-3, 2), Company_Size=(25, 10), Contract_Type=(40, 10),
        Gap_Duration=(-12, 6), Gender_Female=(-25, 8), Age_Year=(-1, 1))),
 'P8': ('성별 민감형 (보건·돌봄)', 210, 14, dict(
        Residual=(20, 20), Act_Train_Code=(15, 5), Act_Train_Female=(55, 8), Act_Train_Age=(-0.5, 0.5),
        Act_Placement=(5, 3), Time_Year=(3, 2), Exp_Matched=(25, 10), NotExp=(5, 5),
        Exp_General=(0, 0), Decay=(-2, 2), Company_Size=(15, 8), Contract_Type=(30, 10),
        Gap_Duration=(-10, 5), Gender_Female=(-10, 5), Age_Year=(1, 1))),
}

# ── Sheet2: 모집단과 공정
POP = {   # 페르소나: (인구비중, 여성비율, 연령min, 연령max)
 'P1': (0.12, 0.50, 24, 54), 'P2': (0.12, 0.50, 24, 54), 'P3': (0.12, 0.50, 24, 54),
 'P4': (0.12, 0.50, 24, 54), 'P5': (0.12, 0.50, 24, 54),
 'P6': (0.13, 0.02, 25, 55), 'P7': (0.13, 0.45, 24, 40), 'P8': (0.14, 0.65, 26, 55),
}
PROC = [
 ('P_SEP', 0.20, '취업 → 실직 월 확률'),
 ('P_HIRE', 0.10, '실직 → 취업 월 확률'),
 ('P_TRAIN_UNEMP', 0.0140, '실직 중 훈련 수강 월 확률  ← 실데이터에 맞춰 재조정 (01 은 0.05)'),
 ('P_TRAIN_EMP', 0.00140, '취업 중 훈련 수강 월 확률  ← 실직 중의 1/10'),
 ('TRAIN_GATE_MONTHS', 9, '직전 훈련 후 재수강까지 최소 개월  (01 은 12)'),
 ('TRAIN_FRAILTY_SD', 1.1, '사람별 수강성향 배수의 로그정규 표준편차 ← 반복수강 분포를 실데이터에 맞춤'),
 ('P_PLACE', 0.30, '실직 중 알선 받을 월 확률'),
 ('P_TARGET_JOB', 0.80, '훈련 시 타겟 직종을 고를 확률'),
 ('T', 120, '관측 개월 수'),
 ('DECAY_PERIOD', 60.0, '상태_생애_훈련직종 효과잔존량의 반감 기준(개월)'),
]
CAL = [   # N=2,000 · seed 20260821 로 실제 생성해 확인한 값
 ('1회 이상 훈련', 0.4995, 0.5005), ('총 훈련사건(2,000명)', 1897, 1837),
 ('훈련자 평균 횟수', 1.90, 1.84), ('취업률', 0.3422, 0.3205),
 ('Y|취업 평균', 5.65, 5.4453),
]

ORDER = [('Intercept', None, ''), ('Residual', '상태', ''),
         ('Act_Train_Code', '정책', 'JOB'), ('Act_Train_Female', '정책', '여성'),
         ('Act_Train_Age', '정책', '1세당'), ('Act_Placement', '정책', ''),
         ('Time_Year', '상태', ''), ('Exp_Matched', '상태', 'JOB'),
         ('Not Exp_Matched', '', ''), ('Exp_General', '상태', ''),
         ('Last_Train (Decay)', '상태', ''), ('Company_Size', '상태', ''),
         ('Contract_Type', '상태', ''), ('Gap_Duration', '상태', ''),
         ('Gender_Female', '상태', '여성'), ('Age_Year', '상태', '1세당')]
KEY = {'Not Exp_Matched': 'NotExp', 'Last_Train (Decay)': 'Decay'}

HDR = PatternFill('solid', fgColor='D9E2F3'); NEW = PatternFill('solid', fgColor='E2EFDA')
THIN = Border(*[Side('thin', color='BFBFBF')] * 4)


def coef(base, won, var, tm):
    """계수 = log((기저+증가)/기저). 교호항은 훈련효과 기준의 로그 차이."""
    if var == 'Intercept':
        return math.log(base)
    if var == 'Act_Train_Female':
        return math.log((base + won) / base) - math.log((base + tm) / base)
    if var == 'Act_Train_Age':
        return math.log((base + tm + won) / base) - math.log((base + tm) / base)
    return math.log((base + won) / base)


def main():
    wb = openpyxl.Workbook()
    ws = wb.active; ws.title = 'Sheet1'
    ws['A1'] = 'synthetic data 생성 규칙  02  (01 의 확대판 — P1~P5 는 01 과 값이 같습니다)'
    ws['A1'].font = Font(bold=True, size=12)
    head1 = ['페르소나ID', '페르소나명', '구분', '변수명', '변수값', '임금증가', '임금증가', '임금증가', '임금증가']
    head2 = ['', '', '', '', '', '평균', '표준편차', '평균', '표준편차']
    for j, (a, b) in enumerate(zip(head1, head2), 1):
        ws.cell(2, j, a).fill = HDR; ws.cell(2, j, a).font = Font(bold=True)
        ws.cell(3, j, b).fill = HDR; ws.cell(3, j, b).font = Font(bold=True)
    ws.cell(2, 6, '임금증가(로그계수)'); ws.cell(2, 8, '임금증가(만원)')
    for j in (6, 7, 8, 9):
        ws.cell(2, j).fill = HDR; ws.cell(2, j).font = Font(bold=True)

    r = 4
    for p, (nm, base, tgt, d) in P.items():
        newp = p in ('P6', 'P7', 'P8')
        tm = d['Act_Train_Code'][0]
        for i, (var, gu, mark) in enumerate(ORDER):
            k = KEY.get(var, var)
            won, wsd = (base, 0) if var == 'Intercept' else d[k]
            c = coef(base, won, var, tm)
            csd = (0.0 if wsd == 0 else
                   abs(math.log((base + tm + wsd) / (base + tm)) if var in
                       ('Act_Train_Female', 'Act_Train_Age') else math.log((base + wsd) / base)))
            val = (f'Job{tgt:02d}' if mark == 'JOB' else mark)
            row = [p if i == 0 else '', nm if i == 1 else '', gu, var, val,
                   round(c, 4), round(csd, 4), won, wsd]
            for j, v in enumerate(row, 1):
                cell = ws.cell(r, j, v); cell.border = THIN
                if newp: cell.fill = NEW
                if var in ('Act_Train_Female', 'Act_Train_Age', 'Gender_Female', 'Age_Year'):
                    cell.font = Font(bold=True)
            r += 1
        r += 1
    for col, w in zip('ABCDEFGHI', (11, 26, 7, 20, 9, 12, 12, 11, 11)):
        ws.column_dimensions[col].width = w

    # ── Sheet2
    w2 = wb.create_sheet('Sheet2')
    w2['A1'] = '모집단과 공정  (01 에서는 코드에 숨어 있던 값들)'
    w2['A1'].font = Font(bold=True, size=12)
    w2.append([]); w2.append(['[모집단]'])
    w2.append(['페르소나', '페르소나명', '타겟직종', '인구비중', '여성비율', '연령min', '연령max'])
    for j in range(1, 8):
        w2.cell(w2.max_row, j).fill = HDR; w2.cell(w2.max_row, j).font = Font(bold=True)
    for p, (share, fem, a0, a1) in POP.items():
        w2.append([p, P[p][0], f'Job{P[p][2]:02d}', share, fem, a0, a1])
    w2.append(['합계', '', '', round(sum(v[0] for v in POP.values()), 4), '', '', ''])
    w2.append([]); w2.append(['[공정]'])
    w2.append(['항목', '값', '설명'])
    for j in range(1, 4):
        w2.cell(w2.max_row, j).fill = HDR; w2.cell(w2.max_row, j).font = Font(bold=True)
    for k, v, note in PROC:
        w2.append([k, v, note])
    w2.append([]); w2.append(['[실데이터 대조]'])
    w2.append(['항목', '규칙02 예상', '실데이터'])
    for j in range(1, 4):
        w2.cell(w2.max_row, j).fill = HDR; w2.cell(w2.max_row, j).font = Font(bold=True)
    for k, a, b in CAL:
        w2.append([k, a, b])
    for col, wd in zip('ABCDEFG', (22, 28, 12, 12, 12, 10, 10)):
        w2.column_dimensions[col].width = wd

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    wb.save(OUT)
    print('생성:', OUT)
    print('  페르소나 %d종 × 변수 %d개 = 계수 %d개' % (len(P), len(ORDER), len(P) * len(ORDER)))


if __name__ == '__main__':
    main()
