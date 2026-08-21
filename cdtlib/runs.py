# -*- coding: utf-8 -*-
"""
단계 4 — 실행 관리.

절대 규칙 다섯 (README 규칙 3·4·5 의 집행부)
    1. run_id = 날짜_데이터_모델_시드.  실행 하나가 폴더 하나.
    2. 조기종료·모델선택은 **valid 로만** 한다. test 는 최종 1회만 연다.
    3. **옛 체크포인트를 이어받지 않는다.** 같은 run_id 가 이미 있으면 거부한다.
       (`../_archive/ckpt/` 를 다시 불러 재현이 깨졌던 사고의 재발 방지)
    4. 모든 실행은 config.json · metrics.csv · ckpt_best.pt 를 남긴다.
    5. config.json 에 환경 도장(어느 기계·어느 장치)을 넣는다.
"""
import json
import os
import time

import numpy as np

from .config import DIR_RUNS
from . import env as _env


def run_id(dataset, model, seed, realization=None, tag=None, ver=None):
    """
    실행 이름. **노트북 판 번호**로 묶습니다(`v73_ihdp_cdt_s0`).
    날짜로 묶으면 날이 바뀔 때마다 같은 코드가 새 실행이 되어 쓸데없이 다시 돕니다.
    코드를 고치면 판이 올라가고, 그때 자동으로 새 실행이 됩니다 — 꼬리표가 필요 없습니다.
    """
    d = ('v%s' % ver) if ver is not None else time.strftime('%Y%m%d')
    parts = [d, dataset, model, 's%d' % seed]
    if realization is not None:
        parts.append('r%03d' % realization)
    if tag:
        parts.append(str(tag))
    return '_'.join(parts)


class Run:
    """
    with Run('empservicereal', 'cdt', seed=0, config={...}) as r:
        r.log(epoch=1, train=0.5, valid=0.6)
        r.save_ckpt(model.state_dict(), score=0.6)    # 좋아졌을 때만 저장됩니다
    """

    def __init__(self, dataset, model, seed, realization=None, tag=None,
                 config=None, device=None, overwrite=False, root=None, ver=None):
        self.id = run_id(dataset, model, seed, realization, tag, ver)
        self.dir = os.path.join(root or DIR_RUNS, self.id)
        if os.path.exists(self.dir) and not overwrite:
            raise FileExistsError(
                '이 run_id 가 이미 있습니다: %s\n'
                '  옛 결과를 덮거나 이어받지 않습니다(규칙 5).\n'
                '  다시 돌리시려면 tag= 를 주어 이름을 바꾸거나 overwrite=True 를 쓰십시오.'
                % self.dir)
        os.makedirs(self.dir, exist_ok=True)
        self.t0 = time.time()
        self.device = device or _env.pick_device()
        self.best = None
        self.rows = []
        self.cfg = dict(run_id=self.id, dataset=dataset, model=model, seed=seed,
                        realization=realization, tag=tag, device=self.device,
                        시작=time.strftime('%Y-%m-%d %H:%M:%S'),
                        환경=_env.stamp(self.device))
        if config:
            self.cfg['설정'] = config
        self._write_cfg()

    # ── 기록
    def _write_cfg(self):
        with open(os.path.join(self.dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(self.cfg, f, ensure_ascii=False, indent=1)

    def log(self, **kw):
        kw.setdefault('경과초', round(time.time() - self.t0, 1))
        self.rows.append(kw)
        import pandas as pd
        pd.DataFrame(self.rows).to_csv(os.path.join(self.dir, 'metrics.csv'),
                                       index=False, encoding='utf-8-sig')
        return kw

    def save_ckpt(self, state, score, lower_better=True):
        """valid 점수가 좋아졌을 때만 덮어씁니다. 좋아졌으면 True."""
        better = (self.best is None or
                  (score < self.best if lower_better else score > self.best))
        if not better:
            return False
        self.best = float(score)
        try:
            import torch
            torch.save(state, os.path.join(self.dir, 'ckpt_best.pt'))
        except ImportError:
            np.savez_compressed(os.path.join(self.dir, 'ckpt_best.npz'), **state)
        self.cfg['best_valid'] = self.best
        self._write_cfg()
        return True

    def result(self, **kw):
        """최종 결과(테스트 포함)를 result.json 으로."""
        kw['run_id'] = self.id
        with open(os.path.join(self.dir, 'result.json'), 'w', encoding='utf-8') as f:
            json.dump(kw, f, ensure_ascii=False, indent=1, default=float)
        return kw

    # ── with 문
    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        self.cfg['끝'] = time.strftime('%Y-%m-%d %H:%M:%S')
        self.cfg['총초'] = round(time.time() - self.t0, 1)
        self.cfg['상태'] = '정상' if et is None else '%s: %s' % (et.__name__, ev)
        self._write_cfg()
        return False


class EarlyStop:
    """valid 점수가 patience 번 연속 안 좋아지면 멈춥니다. test 는 절대 안 봅니다."""

    def __init__(self, patience=10, min_delta=0.0, lower_better=True):
        self.p, self.d, self.lb = patience, min_delta, lower_better
        self.best, self.bad, self.best_epoch = None, 0, -1

    def step(self, score, epoch=None):
        good = (self.best is None or
                (score < self.best - self.d if self.lb else score > self.best + self.d))
        if good:
            self.best, self.bad = float(score), 0
            self.best_epoch = epoch
        else:
            self.bad += 1
        return self.bad >= self.p          # True 면 멈춥니다


def list_runs(root=None):
    """runs/ 안의 실행을 표로. 어떤 것이 끝났고 어떤 것이 죽었는지 한눈에."""
    import pandas as pd
    root = root or DIR_RUNS
    rows = []
    if not os.path.isdir(root):
        return pd.DataFrame(rows)
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d, 'config.json')
        if not os.path.exists(p):
            continue
        try:
            c = json.load(open(p, encoding='utf-8'))
        except Exception:
            continue
        rows.append({'run_id': d, '데이터': c.get('dataset'), '모델': c.get('model'),
                     '시드': c.get('seed'), '장치': c.get('device'),
                     '호스트': c.get('환경', {}).get('호스트'),
                     'best_valid': c.get('best_valid'),
                     '총초': c.get('총초'), '상태': c.get('상태', '진행중')})
    return pd.DataFrame(rows)
