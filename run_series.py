#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Запуск серии из питона, чтобы упростить пошаговую отладку """
import os
import sys
from datetime import datetime
from pathlib import Path
from time import time
import subprocess
from contextlib import redirect_stderr, redirect_stdout
import io
from teeoutput import TeeOutput

os.environ['WANDB_MODE'] = 'offline'
#os.environ["WANDB_DISABLED"] = "true"
CONFIG="config/train_shakespeare_char.py" # "config/train_wikitext.py"

# Эти значения будут добавляться ко всем параметрам
# ("max_iters", 100), ("eval_interval", 100),
BASIC = {'sparsity_type': 'orig',
         'weight_decay': 0.8, 'learning_rate': 0.001, 'min_lr': 2e-05, 'lr_decay_iters': 2000,
         'dropout':0.0, "weight_decay":0.5, 'n_embd':24*16, "n_head":12, "sparsity_type":"masked_weights",
         'early_stop_mode':False, 'max_iters':100000,
         'wdw_pruning_positive_loss_reduction':3e-6,
         'wdw_pruning_negative_loss_reduction':3e-7,
         'wdw_pruning_positive_max_count':1000,
         "wdw_pruning_negative_max_count":0,
         'always_save_checkpoint':True
         }
# Списки значений для перебора:
SERIES = [
          ("wdw_pruning_replanish_ratio", [0.5, .75, 0.9])]

# Тут можно прямо указать дату эксперимента, и тогда можно будет резюмить, на сколько я понимаю
LOG_PATH = Path(os.environ.get('LOG_PATH', f"log/series.{datetime.now().strftime('%Y%m%d_%H%M')}"))
LOG_PATH.mkdir(exist_ok=True, parents=True)
LOGFILE=LOG_PATH/"results.log"

# Обходим дерево вглубину
PATH, EXPERIMENTS = [], []
while True:
    if len(PATH) < len(SERIES): # Если нужно углубиться
        itr = iter(SERIES[len(PATH)][1])
        PATH.append((next(itr),itr))
    elif len(SERIES) == 0:
        break
    else:
        EXPERIMENTS.append([(key,value) for (key,series),(value,iterator) in zip(SERIES,PATH)])
        while len(PATH) > 0:
            try:
                PATH[-1] = (next(PATH[-1][1]), PATH[-1][1])
                break
            except StopIteration:
                del PATH[-1]
        if len(PATH) == 0:
            break
EXPERIMENTS.append([])
print("Expected experiments:\nBASIC:",BASIC, *[f"\n\t{params}" for params in EXPERIMENTS])

def run_once (**kargs):
    run_name = "-".join([f"{k}_{kargs[k]}".replace(".",",") for k in kargs]) if len(kargs)>0 else "basic"
    out_dir = LOG_PATH/run_name/"out"
    grads_dir = LOG_PATH/run_name/"grad"
    print(f"==========================\n\tRun: {run_name}\n\tOut dir: {out_dir}\n==========================")
    # TRAIN
    params = dict(BASIC)
    params.update({"grads_dir":grads_dir, "out_dir":out_dir, "wandb_run_name":run_name})
    params.update(kargs)
    train_global = {'CONFIG':CONFIG, 'PARAMS':params}
    stdout_buffer, stderr_buffer = io.StringIO(), io.StringIO()
    original_stdout,original_stderr = sys.stdout,sys.stderr
    sys.stdout, sys.stderr = TeeOutput(original_stdout, stdout_buffer),TeeOutput(original_stderr, stderr_buffer)
    start_time = time()
    #with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
    exec(open('train.py').read(), train_global)
    sys.stdout,sys.stderr = original_stdout, original_stderr
    print('spended: ', time()-start_time)
    with open(LOG_PATH/run_name/"train.log", 'a', encoding='utf-8') as file:
        file.write(f"spended: {time()-start_time}\n")
        file.write("\nSTDOUT:\n")
        file.write(stdout_buffer.getvalue())
        file.write("\n\nSTDERR:\n")
        file.write(stderr_buffer.getvalue())

    # EVAL (resume)
    params = dict(BASIC)
    params.update({"grads_dir":grads_dir, "out_dir":out_dir, "wandb_run_name":run_name,
              "eval_only":True, "init_from":'resume'})
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
    with open(LOG_PATH/run_name/"train.log", 'a', encoding='utf-8') as file:
        file.write(f"spended: {time()-start_time}\n")
        file.write("\nSTDOUT:\n")
        file.write(stdout_buffer.getvalue())
        file.write("\n\nSTDERR:\n")
        file.write(stderr_buffer.getvalue())
    
    # Пишу глобальный лог
    with open(LOGFILE, 'a', encoding='utf-8') as file:
        file.write(f"{run_name}:\n"+
                   f"\tstep: {train_global['iter_num']}\n"+
                   f"\ttrain loss: {train_global['losses']['train']}\n"+
                   f"\tval loss: {val_global['best_val_loss']}\n"+
                   f"\tppl: {val_global['ppl_val']}\n")
        

with open(LOGFILE, 'a', encoding='utf-8') as file:
    file.write(f"BASIC:{BASIC}\n")

start_time = time()
for params in EXPERIMENTS:
    run_once(**dict(params))
print(f"Все эксперименты завершены. Итоги см. в {LOGFILE}, {(time()-start_time):.1f}")

