#--------------------Common params----------------------#

mode = "custom" # Mode of train (how many and which one of layers should be pruned). Avalibe: "all" or "custom"
custom_slice=slice(4, -5) # Slice of layers, that should be pruned (used if mode =='custom')

dataset = 'wikitext'
batch_size = 32

#-------------------------------------------------------#
#---------------Train and validation params-------------#

wandb_log = True # override via command line if you like
wandb_project = dataset + '_partically_gpt_earlystopped' # change it if need
wandb_run_name= 'name' #overwrited in run_all_config.sh


sparsity_mode = "static"
sparsity_ratio = 0.3 # overwrited in run_all_config.sh 
# Avalible: 
#    float: 0.3 if sparsity_mode = "static"
#    dict - borders: {'start': 0.0, 'end': 0.5, 'total_steps': 100_000} if sparsity_mode = "uniform"
#    dict - grid: {0: 0.0, 10_000: 0.1, 30_000: 0.3, 50_000: 0.5} format {step: sparsity_ratio, ...} if sparsity_mode = "grid"

sparsity_type = "masked-activations-layer" # overwrited in run_all_config.sh  # Avalible: "masked-activations-layer" "masked-weights-layer"

log_interval = 50 # don't print too too often
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200

gradient_accumulation_steps = 1

save_gradients = False
gradient_save_interval = 250 

save_best_model = True # Should be saved best model to best_model.pt
always_save_checkpoint = True # Should be saved all checkpoints like: ckpt_{iter_num}.pt


learning_rate = 5e-4  # или даже 3e-4
max_iters = 500000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small


early_stop_mode = True
early_stop_patience = 15 # number of consecutive evals with rising val loss

compile = False # do not torch compile the model

#-------------------------------------------------------#
#------------------Evaluate params----------------------#
# To start evaluation you need to set eval_only=True and init_from='resume'. Params already configurated in run_all_config.sh.

eval_ckpt_name = "best_model.pt" # Which one of checkpoints from out_dir should be used to get ppl

