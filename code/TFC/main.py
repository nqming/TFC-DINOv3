"""Updated implementation for TF-C -- Xiang Zhang, Jan 16, 2023"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config_files.extension import configure
from dinov3_encoder import build_model
from checkpoint import load_checkpoint
import logging
import numpy as np
from datetime import datetime
import argparse
from utils import _logger
from model import *
from dataloader import data_generator
from trainer import Trainer




# Args selections
start_time = datetime.now()
parser = argparse.ArgumentParser()
######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=42, type=int, help='seed value')

# 1. self_supervised pre_train; 2. finetune (itself contains finetune and test)
parser.add_argument('--training_mode', default='fine_tune_test', type=str, choices=['pre_train', 'fine_tune_test'],
                    help='pre_train, fine_tune_test')

parser.add_argument('--pretrain_dataset', default='SleepEEG', type=str,
                    help='Dataset of choice: SleepEEG, FD_A, HAR, ECG')
parser.add_argument('--target_dataset', default='Epilepsy', type=str,
                    help='Dataset of choice: Epilepsy, FD_B, Gesture, EMG')

parser.add_argument('--logs_save_dir', default='../experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
parser.add_argument('--config', help='Extension JSON configuration (paths relative to current directory)')
parser.add_argument('--resume', help='Pretraining checkpoint including optimizer state')
parser.add_argument('--checkpoint', help='Explicit weights checkpoint for fine-tuning')
parser.add_argument('--data_root', default=str(Path(__file__).resolve().parents[2] / 'datasets'))
parser.add_argument('--subset', action='store_true', help='Use original debug subset')
args = parser.parse_args()
if args.resume and args.training_mode != 'pre_train':
    parser.error('--resume is supported for pre_train only')

device = torch.device(args.device if args.device != 'cuda' or torch.cuda.is_available() else 'cpu')
print('We are using %s now.' %device)

pretrain_dataset = args.pretrain_dataset
targetdata = args.target_dataset
experiment_description = str(pretrain_dataset) + '_2_' + str(targetdata)


method = 'TF-C'
training_mode = args.training_mode
run_description = args.run_description
logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)
exec(f'from config_files.{pretrain_dataset}_Configs import Config as Configs')
configs = configure(Configs(), args.config)
encoder_tag = "2layertransformer" if configs.encoder == "original" else "dinov3"

# # ##### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################

experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, training_mode + f"_seed_{SEED}_{encoder_tag}")
# 'experiments_logs/Exp1/run1/train_linear_seed_0'
os.makedirs(experiment_log_dir, exist_ok=True)

# loop through domains
counter = 0
src_counter = 0


# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
# 'experiments_logs/Exp1/run1/train_linear_seed_0/logs_14_04_2022_15_13_12.log'
logger = _logger(log_file_name)
loss_logger = logging.getLogger("tfc.loss")
loss_logger.setLevel(logging.INFO)
loss_logger.handlers = logger.handlers[:]
loss_logger.propagate = False
logger.debug("=" * 45)
logger.debug(f'Pre-training Dataset: {pretrain_dataset}')
logger.debug(f'Target (fine-tuning) Dataset: {targetdata}')
logger.debug(f'Method:  {method}')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)

# Load datasets
sourcedata_path = os.path.join(args.data_root, pretrain_dataset)
targetdata_path = os.path.join(args.data_root, targetdata)
subset = args.subset
train_dl, valid_dl, test_dl = data_generator(sourcedata_path, targetdata_path, configs, training_mode, subset = subset)
logger.debug("Data loaded ...")

# Load Model
"""Here are two models, one basemodel, another is temporal contrastive model"""
TFC_model = build_model(configs).to(device)
classifier = target_classifier(configs).to(device)
temporal_contr_model = None


if training_mode == "fine_tune_test":
    # load saved model of this experiment
    load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description,
    f"pre_train_seed_{SEED}_{encoder_tag}", "saved_models"))
    print("The loading file path", load_from)
    load_checkpoint(args.checkpoint or os.path.join(load_from, "ckp_last.pt"), TFC_model, device, configs)

model_optimizer = torch.optim.Adam(TFC_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

if args.resume:
    configs.start_epoch = load_checkpoint(args.resume, TFC_model, device, configs, model_optimizer)

# Trainer
Trainer(TFC_model, model_optimizer, classifier, classifier_optimizer, train_dl, valid_dl, test_dl, device,
        logger, configs, experiment_log_dir, training_mode)

logger.debug(f"Training time is : {datetime.now()-start_time}")
