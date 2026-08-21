# -*- coding: utf-8 -*-
"""
어느 기계에서 계산했는지 · 어떤 장치를 썼는지를 결과에 박아 두는 모듈.

컴퓨터가 둘(윈도우 CPU · 우분투 GPU)이라 이것이 없으면 나중에
「이 표의 이 칸은 어디서 나온 숫자인가」를 되짚을 수 없습니다.
CPU 와 GPU 는 같은 시드로도 부동소수점 연산 순서가 달라 소수점 아래가 다릅니다.
그래서 **표 하나 안의 숫자는 반드시 한 기계에서** 뽑아야 합니다.
"""
import os
import platform
import random
import warnings

import numpy as np


def pick_device(prefer='auto'):
    """'auto' | 'cuda' | 'cpu'. torch 가 없으면 'cpu'."""
    try:
        import torch
    except ImportError:
        return 'cpu'
    if prefer == 'cpu':
        return 'cpu'
    if prefer == 'cuda' or prefer == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        if prefer == 'cuda':
            raise RuntimeError('cuda 를 요청했는데 torch.cuda.is_available() 이 False 입니다.')
    return 'cpu'


def stamp(device=None):
    """이 실행의 환경 도장. config.json · 결과표에 그대로 넣습니다."""
    d = dict(호스트=platform.node(), os='%s %s' % (platform.system(), platform.release()),
             python=platform.python_version(), numpy=np.__version__)
    try:
        import torch
        d['torch'] = torch.__version__
        dev = device or pick_device()
        d['장치'] = dev
        if dev == 'cuda' and torch.cuda.is_available():
            d['gpu'] = torch.cuda.get_device_name(0)
            d['vram_gb'] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            d['cuda'] = torch.version.cuda
    except ImportError:
        d['torch'] = '(없음)'
        d['장치'] = 'cpu'
    try:
        import psutil
        d['ram_gb'] = round(psutil.virtual_memory().total / 1e9, 1)
        d['cpu코어'] = psutil.cpu_count(logical=False)
    except Exception:
        pass
    return d


def show(device=None):
    s = stamp(device)
    w = max(len(k) for k in s)
    for k, v in s.items():
        print('  %-*s  %s' % (w, k, v))
    return s


DETERMINISTIC_NOTE = [None]      # 한 번만 알리려고 담아 둡니다
STRICT = [True]                  # 결정적 연산을 쓸 수 있는가 (check_deterministic 이 정합니다)


def check_deterministic(device=None, verbose=True):
    """
    **미리 확인합니다.** `use_deterministic_algorithms(True)` 는 켤 때가 아니라
    결정적 구현이 없는 연산을 만나는 순간 예외를 던집니다. 그것이 학습 도중에
    터지면 몇 시간짜리 실행이 통째로 날아갑니다. 그래서 여기서 실제 부품
    (임베딩·어텐션·LayerNorm)으로 순전파·역전파를 한 번 해 보고 정합니다.

    반환: True 면 결정적 연산을 켠 채로 돕니다.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        STRICT[0] = False
        return False
    dev = device or pick_device()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        # **실제 모형으로** 한 걸음 돌려 봅니다. 작은 대용품으로는 못 잡는 연산이 있습니다.
        from .models import build_cdt_modules
        net = build_cdt_modules()(state_dim=5, n_action=2, hidden=16, K=4,
                                  n_layer=2, n_head=2).to(dev)
        B, K = 3, 4
        s = torch.randn(B, K, 5, device=dev)
        a = torch.randint(0, 2, (B, K), device=dev)
        ts = torch.arange(K, device=dev).expand(B, K).contiguous()
        m = torch.ones(B, K, device=dev)
        m[0, 0] = 0                                  # 가려진 자리도 함께 확인
        logit, rhat = net(s, a, torch.zeros(B, K, 1, device=dev), ts, mask=m)
        loss = (nn.CrossEntropyLoss()(logit.reshape(-1, 2), a.reshape(-1))
                + (rhat[:, -1, 0] ** 2).mean())
        loss.backward()
        del net, s, a, ts, m, logit, rhat, loss
        STRICT[0] = True
        DETERMINISTIC_NOTE[0] = '결정적 연산 켜짐 — 같은 시드면 같은 숫자가 나옵니다'
    except Exception as e:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        STRICT[0] = False
        DETERMINISTIC_NOTE[0] = '결정적 연산 불가 (%s) — 시드를 늘려 평균±표준편차로 보고하십시오' % str(e)[:70]
    if verbose:
        print('  %s' % DETERMINISTIC_NOTE[0])
    return STRICT[0]


def set_seed(seed, deterministic=True, strict=True):
    """
    재현성.

    **`cudnn.deterministic` 만으로는 부족합니다.** 그것은 합성곱 알고리즘 선택에만
    영향을 줍니다. 트랜스포머가 쓰는 LayerNorm·GELU·임베딩 역전파는 원자적 덧셈으로
    누적해서 더하는 순서가 실행마다 달라지고, 200에폭을 도는 동안 그 차이가 커집니다.

    v75 에서 **완전히 같은 설정을 세 번 돌려 √PEHE 가 29.85 / 30.88 / 34.68** 이
    나왔습니다(합성 v2). 재려던 효과보다 이 흔들림이 더 컸습니다.
    `strict=True` 면 `torch.use_deterministic_algorithms(True)` 까지 켜서
    **같은 기계·같은 시드면 같은 숫자**가 나오게 합니다.

    그래도 CPU 와 GPU 의 숫자는 같아지지 않습니다(연산 순서 자체가 다릅니다).
    표 하나 안의 숫자는 반드시 한 기계에서 뽑아야 합니다.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        if strict and STRICT[0]:
            # check_deterministic() 이 미리 통과시킨 경우에만 켭니다.
            try:
                torch.use_deterministic_algorithms(True, warn_only=False)
            except Exception:
                pass
    except ImportError:
        pass


def autocast_ctx(device, enabled=True):
    """GPU 면 bf16/fp16 자동혼합정밀, CPU 면 아무것도 안 하는 문맥."""
    import contextlib
    try:
        import torch
    except ImportError:
        return contextlib.nullcontext()
    if device != 'cuda' or not enabled:
        return contextlib.nullcontext()
    dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast('cuda', dtype=dt)


def fit_batch(fn, batch, min_batch=4):
    """
    VRAM 부족 대비. `fn(batch)` 가 메모리 부족으로 죽으면 배치를 반으로 줄여 다시 잡습니다.
    성공한 배치 크기를 함께 돌려주므로 그 뒤로는 그 값을 쓰면 됩니다.
    """
    try:
        import torch
    except ImportError:
        return fn(batch), batch
    b = int(batch)
    while b >= min_batch:
        try:
            return fn(b), b
        except RuntimeError as e:
            if 'out of memory' not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            b //= 2
            print('  VRAM 부족 — 배치를 %d 로 줄여 다시 시도합니다' % b, flush=True)
    raise RuntimeError('배치를 %d 까지 줄여도 VRAM 이 부족합니다.' % min_batch)

# ══════════════════════════════════════ 라이브러리 지문
def lib_fingerprint(root='.', subdirs=('cdtlib', 'tools'), exts=('.py',)):
    """
    `cdtlib/*.py` · `tools/*.py` 의 내용을 해싱합니다.

    노트북 지문만으로는 부족합니다. 컴퓨터가 둘이면 노트북은 같은데 라이브러리 한 파일만
    옛것인 상황이 생기고, 그러면 **지문은 같은데 결과가 다릅니다.**
    반환: (합친 지문, {파일: 지문})
    """
    import hashlib
    import os as _os
    per = {}
    h = hashlib.md5()
    for sub in subdirs:
        d = _os.path.join(root, sub)
        if not _os.path.isdir(d):
            continue
        for fn in sorted(_os.listdir(d)):
            if not fn.endswith(exts) or fn.startswith('__'):
                continue
            fp = _os.path.join(d, fn)
            b = open(fp, 'rb').read()
            g = hashlib.md5(b).hexdigest()[:8]
            per['%s/%s' % (sub, fn)] = g
            h.update(('%s/%s:%s|' % (sub, fn, g)).encode())
    return h.hexdigest()[:10], per


REQUIRED_API = {
    'CDT.fit': ('configs', 'head', 'select', 'state_drop', 'action', 'dropout', 'K'),
    'TARNet.fit': ('select', 'configs', 'lr'),
}


def check_lib(expect=None, root='.', hard=True):
    """
    **노트북과 라이브러리가 같은 판인지 1초 만에 확인합니다.**

    v78 에서 이런 일이 있었습니다 — 노트북만 우분투로 옮기고 `cdtlib/models.py` 는
    옛 판이 남아, `CDT.fit() got an unexpected keyword argument 'configs'` 로
    **CDT 125개 실행이 전부 실패**했습니다. 그런데 표는 나머지 다섯 모형으로
    멀쩡히 채워져서, 요약 맨 아래 한 줄을 보기 전까지는 아무 문제가 없어 보였습니다.
    한 시간을 버린 뒤에 알게 되는 종류의 사고입니다.

    지문 비교(어긋났나)와 **기능 검사**(그 인자를 정말 받는가)를 함께 합니다.
    지문은 「다르다」만 알려 주지만, 기능 검사는 「무엇이 없다」를 짚어 줍니다.
    """
    import inspect
    from . import models as _m
    now, _ = lib_fingerprint(root)
    miss = []
    for path, args in REQUIRED_API.items():
        cls, fn = path.split('.')
        try:
            sig = inspect.signature(getattr(getattr(_m, cls), fn)).parameters
        except Exception as e:
            miss.append('%s 를 못 찾음 (%s)' % (path, e))
            continue
        for a in args:
            if a not in sig:
                miss.append('%s(%s=) 가 없음' % (path, a))
    ok = not miss
    if expect and now != expect:
        print('  라이브러리 지문이 다릅니다: 지금 %s · 기대 %s' % (now, expect))
    if miss:
        print()
        print('  ' + '=' * 72)
        print('  ⚠ cdtlib 이 이 노트북보다 옛 판입니다. 지금 멈추십시오.')
        for m in miss:
            print('     · %s' % m)
        print('  윈도우 폴더의 confieee2026/cdtlib/ 전체를 이 기계로 다시 복사하십시오.')
        print('  (노트북만 옮기면 이 사고가 납니다 — v78 에서 CDT 125개가 전부 죽었습니다)')
        print('  ' + '=' * 72)
        if hard:
            raise RuntimeError('cdtlib 이 옛 판입니다: ' + ' / '.join(miss[:3]))
    elif expect and now == expect:
        print('  라이브러리 검사 통과 (지문 %s · 필요한 인자 전부 있음)' % now)
    else:
        print('  라이브러리 기능 검사 통과 (지문 %s)' % now)
    return ok


def show_lib(root='.'):
    """두 기계를 견줄 때 이 표를 나란히 놓으면 어느 파일이 옛것인지 바로 보입니다."""
    g, per = lib_fingerprint(root)
    print('라이브러리 지문 %s   (%d개 파일)' % (g, len(per)))
    w = max((len(k) for k in per), default=10)
    for k, v in per.items():
        print('   %-*s  %s' % (w, k, v))
    return g, per

# ══════════════════════════════════════ 그림 · 한글 폰트
KOREAN_FONTS = ('Malgun Gothic', 'NanumGothic', 'Nanum Gothic', 'NanumBarunGothic',
                'Noto Sans CJK KR', 'Noto Sans KR', 'Source Han Sans KR',
                'AppleGothic', 'UnDotum', 'Baekmuk Gulim')


def setup_matplotlib(prefer=KOREAN_FONTS, quiet=True, verbose=True):
    """
    **설치된 폰트 목록에서 골라야 합니다.**
    `plt.rc('font', family='없는폰트')` 는 예외를 안 냅니다. 그래서
    try/except 로 대체 폰트를 고르던 옛 코드는 언제나 첫 후보에서 멈췄고,
    리눅스에서 `findfont: Font family 'Malgun Gothic' not found` 가 수백 줄 쏟아졌습니다.

    반환: 고른 폰트 이름(한글 폰트가 없으면 None)
    """
    import matplotlib as mpl
    from matplotlib import font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    pick = next((f for f in prefer if f in have), None)
    mpl.rcParams['font.family'] = pick or 'DejaVu Sans'
    mpl.rcParams['axes.unicode_minus'] = False
    if quiet:
        import logging
        logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
        for pat in (r'.*findfont.*', r'.*[Mm]issing from (current )?font.*',
                    r'.*Glyph \d+ .*'):
            warnings.filterwarnings('ignore', message=pat)
    if verbose:
        if pick:
            print('그림 폰트: %s' % pick)
        else:
            print('그림 폰트: 한글 폰트가 없습니다 → DejaVu Sans (한글은 네모로 나옵니다)')
            print('  논문 그림은 영문 라벨이라 문제 없습니다. 탐색용 그림의 한글을 보시려면:')
            print('    sudo apt install -y fonts-nanum && rm -rf ~/.cache/matplotlib')
    return pick

def nb_fingerprint(path):
    """
    노트북 코드 셀의 지문. **셀별 지문도 함께** 돌려줍니다.

    두 기계의 지문이 다를 때 「어디가 다른가」를 알 수 없으면 경고가 소용없습니다.
    셀별 값을 `out/_지문.json` 에 남겨 두면 두 파일을 나란히 놓고 바로 짚을 수 있습니다.
    """
    import hashlib
    import json as _js
    try:
        nb = _js.load(open(path, encoding='utf-8'))
    except Exception:
        return None, []
    per, h = [], hashlib.md5()
    for i, c in enumerate(nb.get('cells', [])):
        if c.get('cell_type') != 'code':
            continue
        src = c['source']
        src = ''.join(src) if isinstance(src, list) else str(src)
        g = hashlib.md5(src.encode('utf-8')).hexdigest()[:8]
        head = next((l.strip() for l in src.split('\n') if l.strip()), '')
        per.append({'셀': i, '지문': g, '첫줄': head[:60]})
        h.update((g + '|').encode())
    return h.hexdigest()[:10], per
