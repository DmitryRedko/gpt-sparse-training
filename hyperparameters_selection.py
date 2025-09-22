# -*- coding: utf-8 -*-
"""
@author: Влад
Динамический Подбор гиперпараметров в процессе обучения

"""
import os
import sys
import random
from types import SimpleNamespace
from datetime import datetime
from pathlib import Path
import shutil
from time import time
import subprocess
from contextlib import redirect_stderr, redirect_stdout
import io
from teeoutput import TeeOutput
import torch

os.environ['WANDB_MODE'] = 'offline'
CONFIG="config/train_shakespeare_char.py"
RESUME = "" # Не None чтобы конфигуратор знал исходный тип
CYCLES_ITER = 250
MAX_ITER = 50000
LOG_PATH = Path(os.environ.get('LOG_PATH', f"log/series.{datetime.now().strftime('%Y%m%d_%H%M')}"))


exec(open('configurator.py').read())  # may overwrite globals

class ExponentialChanger:
    def __init__(self, name:str, scaler:float, min_value:float=None, max_value:float=None):
        self.name = name
        self.scaler,self.min_value,self.max_value = scaler,min_value,max_value
        self.epsilon=0.01
    def up(self, cfg:dict):
        value = cfg[self.name]
        if (self.max_value is not None) and (abs(self.max_value-value) < abs(self.epsilon*self.max_value)):
            return None
        value = value*self.scaler
        if self.max_value is not None:
            value = min(value, self.max_value)
            if abs(self.max_value-value) < abs(self.epsilon*self.max_value): value = self.max_value
        cfg = cfg.copy()
        cfg[self.name] = value
        return cfg
    def down(self, cfg:dict):
        value = cfg[self.name]
        if (self.min_value is not None) and (abs(self.min_value-value) < abs(self.epsilon*self.min_value)):
            return None
        value = value/self.scaler
        if self.min_value is not None:
            value = max(value, self.min_value)
            if abs(self.min_value-value) < abs(self.epsilon*self.min_value): value = self.min_value
        cfg = cfg.copy()
        cfg[self.name] = value
        return cfg
class IncrementialChanger:
    def __init__(self, name:str, step:float, min_value:float=None, max_value:float=None):
        self.name = name
        self.step,self.min_value,self.max_value = step,min_value,max_value
        self.epsilon=0.01
    def up(self, cfg:dict):
        value = cfg[self.name]
        if (self.max_value is not None) and (abs(self.max_value-value) <= abs(self.epsilon*self.max_value)):
            return None
        value = value + self.step
        if self.max_value is not None:
            value = min(value, self.max_value)
            if abs(self.max_value-value) <= abs(self.epsilon*self.max_value): value = self.max_value
        cfg = cfg.copy()
        cfg[self.name] = value
        return cfg
    def down(self, cfg:dict):
        value = cfg[self.name]
        if (self.min_value is not None) and (abs(self.min_value-value) <= abs(self.epsilon*self.min_value)):
            return None
        value = value - self.step
        if self.min_value is not None:
            value = max(value, self.min_value)
            if abs(self.min_value-value) <= abs(self.epsilon*self.min_value): value = self.min_value
        cfg = cfg.copy()
        cfg[self.name] = value
        return cfg
class CompositeChanger:
    def __init__(self, *chaners):
        self.chaners = chaners
    @property
    def name(self):
        result = []
        for c in self.chaners:
            children = c.name if isinstance(c.name, list) else [c.name]
            result.extend(children)
        return result
    def up(self, cfg:dict):
        for changer in self.chaners:
            cfg = changer.up(cfg)
            if cfg is None:
                return None
        return cfg
    def down(self, cfg:dict):
        for changer in self.chaners:
            cfg = changer.up(cfg)
            if cfg is None:
                return None
        return cfg
class InverseChanger:
    def __init__(self, changer):
        self.changer = changer
    @property
    def name(self):
        return self.changer.name
    def up(self, cfg:dict):
        return self.changer.down(cfg)
    def down(self, cfg:dict):
        return self.changer.up(cfg)

### vvvvv ТУТ НАСТРАИВАЕМЫЕ ПАРАМЕТРЫ vvvvv ###
BASIC = {'sparsity_type': 'masked_weights', # "masked_weights",
         'weight_decay': 0.8, 'learning_rate': 0.002, 'min_lr': 2e-05, 'lr_decay_iters': 2000,
         'dropout':0.0, "weight_decay":1,
         'early_stop_mode':False, 'max_iters':100000,
         # 'always_save_checkpoint':True
         "save_last_checkpoint":False}
CHANGERS = [ExponentialChanger('learning_rate',2),
            ExponentialChanger('min_lr',2),
            ExponentialChanger('weight_decay', 1.25),
            IncrementialChanger("dropout", .05, 0, .95)
            ]
### ^^^^^ ТУТ НАСТРАИВАЕМЫЕ ПАРАМЕТРЫ ^^^^^ ###

all_affected = []
for c in CHANGERS:
    all_affected.extend(c.name if isinstance(c.name, list) else [c.name])
current_params = dict([(name, BASIC[name]) for name in all_affected])
BASIC = dict([item for item in BASIC.items() if item[0] not in current_params])
LOG_PATH.mkdir(exist_ok=True, parents=True)
LOGFILE=LOG_PATH/"results.log"

class RunResult: # Задоблало нетипизированное болото
    def __init__(self, loss_train:torch.Tensor,
                       loss_val:torch.Tensor,
                       out:Path,
                       cfg:dict,
                       iters_num:int,
                       loss_best_val:torch.Tensor):
        self.loss_train = loss_train
        self.loss_val = loss_val
        self.cfg = cfg
        self.out = out
        self.iters_num = iters_num
        self.loss_best_val = loss_best_val
    def __str__(self) -> str:
        return f"RunResult(train={self.loss_train.item():.4f}, val={self.loss_val.item():.4f}, out={self.out}, cfg={self.cfg}, iters={self.iters_num}, best_val:{self.loss_best_val.item():4f})"
    def __repr__(self) -> str:
        return str(self)

def write_global_log(output:str):
    with open(LOGFILE, 'a', encoding='utf-8') as file:
        file.write(output)
        file.write("\n")

current_iter = 0
phase = 0

def run_once (source:Path=None, **kargs)->RunResult:
    run_name = "-".join([f"{k}_{v:.2e}".replace(".",",") if isinstance(v, float) else f"{k}_{v}" for k,v in kargs.items()])
    run_dir = LOG_PATH/f"phase{phase:05d}"/run_name
    out_dir = run_dir/"out"
    params = dict(BASIC)
    params.update({"out_dir":out_dir, "wandb_run_name":run_name, "eval_stable":True, "max_iters":CYCLES_ITER+current_iter})
    params.update(kargs)
    if source is not None:
        source = Path(source) if not isinstance(source, Path) else source
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in source.glob(f'last_*.pt'):
            shutil.copy2(f, out_dir/(f.name))
        params.update({'init_from':'resume', 'save_last_model':True, 'eval_ckpt_name':'last_model.pt', 'eval_history_name':'last_history.pt'})
    else:
        params.update({"always_save_checkpoint":True})
    
    print(f"==========================\n\tRun: {run_name}\n\tOut dir: {out_dir}\n==========================")
    
    # TRAIN
    train_global = {'CONFIG':CONFIG, 'PARAMS':params}
    stdout_buffer, stderr_buffer = io.StringIO(), io.StringIO()
    original_stdout,original_stderr = sys.stdout,sys.stderr
    sys.stdout, sys.stderr = TeeOutput(original_stdout, stdout_buffer),TeeOutput(original_stderr, stderr_buffer)
    start_time = time()
    #with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
    exec(open('train.py').read(), train_global)
    sys.stdout,sys.stderr = original_stdout, original_stderr
    print('spended: ', time()-start_time)
    with open(run_dir/"train.log", 'a', encoding='utf-8') as file:
        file.write(f"spended: {time()-start_time}\n")
        file.write("\nSTDOUT:\n")
        file.write(stdout_buffer.getvalue())
        file.write("\n\nSTDERR:\n")
        file.write(stderr_buffer.getvalue())

    # EVAL (resume)
    params = dict(BASIC)
    params.update({"out_dir":out_dir, "wandb_run_name":run_name, "eval_only":True, "init_from":'resume', 'eval_ckpt_name':'last_model.pt', 'eval_history_name':'last_history.pt'})
    params.update(kargs)
    val_global = {'CONFIG':CONFIG, 'PARAMS':params}
    stdout_buffer, stderr_buffer = io.StringIO(), io.StringIO()
    original_stdout,original_stderr = sys.stdout,sys.stderr
    sys.stdout, sys.stderr = TeeOutput(original_stdout, stdout_buffer),TeeOutput(original_stderr, stderr_buffer)
    start_time = time()
    #with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
    exec(open('train.py').read(), val_global)
    sys.stdout,sys.stderr = original_stdout, original_stderr
    print('spended: ', time()-start_time)
    with open(run_dir/"train.log", 'a', encoding='utf-8') as file:
        file.write(f"spended: {time()-start_time}\n")
        file.write("\nSTDOUT:\n")
        file.write(stdout_buffer.getvalue())
        file.write("\n\nSTDERR:\n")
        file.write(stderr_buffer.getvalue())
    
    # Пишу глобальный лог
    write_global_log(f"{run_name}:\n"+
                     f"\tstep: {train_global['iter_num']}\n"+
                     f"\ttrain loss: {val_global['losses']['train']}\n"+
                     f"\tval loss: {val_global['best_val_loss']}\n"+
                     f"\tppl: {val_global['ppl_val']}\n")
    return RunResult(val_global['losses']['train'], val_global['losses']['val'], out_dir, kargs, train_global['iter_num'], val_global['best_val_loss'])

        

best_attempt = run_once(None, **current_params)
SOURCE_PATH = LOG_PATH/"initial"
SOURCE_PATH.mkdir(parents=True, exist_ok=True)
for f in best_attempt.out.glob(f'ckpt_{0:05d}*.pt'):
    shutil.copy2(f, SOURCE_PATH/('last'+f.name[len(f'ckpt_{0:05d}'):]))

def config_diff(one:dict[str,float], other:dict[str,float]):
    return [(n,(one[n],other[n])) for n in one if one[n] != other[n]]

def dense_cycle(cycles:int=3):
    '''Плотный цикл, все параметры перебираются пока не происходит выбор оптимального.'''
    global best_attempt
    global SOURCE_PATH
    global current_iter
    global phase
    for cycle in range(cycles):
        optimal = True
        for changer in CHANGERS:
            new_params = changer.down(best_attempt.cfg)
            if new_params is not None:
                attempt = run_once(SOURCE_PATH, **new_params)
                if attempt.loss_val < best_attempt.loss_val:
                    while attempt.loss_val < best_attempt.loss_val:
                        optimal = False
                        write_global_log(f"Hyperparameter improvement: {config_diff(best_attempt.cfg,attempt.cfg)}; loss:{attempt.loss_val-best_attempt.loss_val:.5f}\n")
                        best_attempt = attempt
                        new_params = changer.down(best_attempt.cfg)
                        if new_params is None:
                            break
                        attempt = run_once(SOURCE_PATH, **new_params)
                    continue
            new_params = changer.up(best_attempt.cfg)
            if new_params is not None:
                attempt = run_once(SOURCE_PATH, **new_params)
                if attempt.loss_val < best_attempt.loss_val:
                    while attempt.loss_val < best_attempt.loss_val:
                        optimal = False
                        write_global_log(f"Hyperparameter improvement: {config_diff(best_attempt.cfg,attempt.cfg)}; loss:{attempt.loss_val-best_attempt.loss_val:.5f}\n")
                        best_attempt = attempt
                        new_params = changer.up(best_attempt.cfg)
                        if new_params is None:
                            break
                        attempt = run_once(SOURCE_PATH, **new_params)
                    continue
            # Этот параметр в оптимуме, ничего менять не нужно
        if optimal:
            break
    phase = phase + 1
    current_iter = best_attempt.iters_num
    SOURCE_PATH = best_attempt.out
    best_attempt = run_once(SOURCE_PATH, **best_attempt.cfg)

def sparse_cycle(changer):
    '''Неплотный цикл, проверяется только один очередной параметр.'''
    global best_attempt
    global SOURCE_PATH
    global current_iter
    global phase

    def end():
        global phase
        global current_iter
        global SOURCE_PATH
        global best_attempt
        phase = phase + 1
        current_iter = best_attempt.iters_num
        SOURCE_PATH = best_attempt.out
        best_attempt = run_once(SOURCE_PATH, **best_attempt.cfg)
    global best_attempt
    optimal = True
    new_params = changer.up(best_attempt.cfg)
    if new_params is not None:
        attempt = run_once(SOURCE_PATH, **new_params)
        if attempt.loss_val < best_attempt.loss_val:
            while attempt.loss_val < best_attempt.loss_val:
                optimal = False
                write_global_log(f"Hyperparameter improvement: {config_diff(best_attempt.cfg,attempt.cfg)}; loss:{attempt.loss_val-best_attempt.loss_val:.5f}\n")
                best_attempt = attempt
                new_params = changer.up(best_attempt.cfg)
                if new_params is None:
                    end();return
                attempt = run_once(SOURCE_PATH, **new_params)
            end();return
    new_params = changer.down(best_attempt.cfg)
    if new_params is not None:
        attempt = run_once(SOURCE_PATH, **new_params)
        if attempt.loss_val < best_attempt.loss_val:
            while attempt.loss_val < best_attempt.loss_val:
                optimal = False
                write_global_log(f"Hyperparameter improvement:{config_diff(best_attempt.cfg,attempt.cfg)}; loss:{attempt.loss_val-best_attempt.loss_val:.5f}\n")
                best_attempt = attempt
                new_params = changer.down(best_attempt.cfg)
                if new_params is None:
                    end();return
                attempt = run_once(SOURCE_PATH, **new_params)
            end();return
    # Этот параметр в оптимуме, ничего менять не нужно
    end();
    
dense_cycle()
for i in range(3):
    dense_cycle(1)
for i in range(100):
    changers_shuffled = CHANGERS.copy()
    random.shuffle(changers_shuffled)
    for changer in changers_shuffled:
        sparse_cycle(changer)


























