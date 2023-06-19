import os

import optuna
from optuna.trial import TrialState

from dataset.dataset import split_dataset, XRayDataset
from dataset.transforms import get_train_transform
from optim.losses import create_criterion
from utils import *
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from optim.losses import dice_coef
import torch.nn as nn
import torch.nn.functional as F
from importlib import import_module
import wandb
import torch.cuda.amp as amp
import matplotlib.pyplot as plt
import torch.cuda.amp as amp

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 50)')
    

    # data
    parser.add_argument("--resize", nargs="+", type=int, default=[512,512], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training (default: 8)')
    parser.add_argument('--valid_batch_size', type=int, default=2, help='input batch size for validing (default: 2)')

    parser.add_argument('--model', type=str, default='FcnResnet50', help='model name (default: FcnResnet50)')
    parser.add_argument('--loss', type=str, default='combine', help='[bce, focal, dice, iou, combine: (default: bce)')

    args = parser.parse_args()
    
    return args

# 시드 실험
# def get_seed(trial):
#     the_seed = trial.suggest_int("seed",1,1000)

#     return the_seed

def get_dataset():
    args = get_args()

    seed_everything(args.seed)
    
    train_filenames, train_labelnames, val_filenames, val_labelnames = split_dataset()
    
    train_transform = get_train_transform(img_size=args.resize)
    val_transform = get_train_transform(train=False,img_size=args.resize)
    
    train_dataset = XRayDataset(
                                filenames = train_filenames,
                                labelnames = train_labelnames,
                                transforms= train_transform
                                )
    val_dataset = XRayDataset(
                            filenames = val_filenames,
                            labelnames = val_labelnames,
                            is_train = False,
                            transforms= val_transform
                            )

    num_workers = min(args.batch_size, 8)

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    valid_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=args.valid_batch_size,
        shuffle=False,
        drop_last=False
    )
    return train_loader, valid_loader

def define_model():
    args = get_args()

    # model_type = trial.suggest_categorical("model_type", ["FcnResnet50","HRNet48OCR","NestedUNet","Unet","PSPNet","DeepLabV3Plus"])
    # model_type = "PSPNet"

    model_module = getattr(import_module("models.my_model"), args.model)  # default: BaseModel
    model = model_module(
        num_classes=29
    )
    
    return model


def objective(trial):
    args = get_args()

    seed_everything(args.seed)

    # Generate the model.
    model = define_model().cuda()

    # Loss function 정의
    # loss_type = trial.suggest_categorical("loss_type",["bce","focal","dice","iou","combine"])
    loss_type = args.loss
    criterion = create_criterion(loss_type)

    # optimizer 설정 default = Adam
    # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD", "AdamW"])
    optimizer_name = "Adam"
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(import_module("torch.optim"), optimizer_name)(model.parameters(), lr=lr)

    # threshold 실험하기 default = 0.5
    # thres = trial.suggest_categorical("threshold", [0.3, 0.4, 0.5, 0.6, 0.7])
    thres = 0.5

    # Scheduler 정의
    sched_module = getattr(import_module("torch.optim.lr_scheduler"), "StepLR") # default: steplr
    scheduler = sched_module(optimizer, step_size=10, gamma=0.1)
    

     # Get the FashionMNIST dataset.
    train_loader, valid_loader = get_dataset()

    print(f'Start training..')
    
    n_class = len(XRayDataset.CLASSES)
    best_dice = 0.

    # AMP : loss scale을 위한 GradScaler 생성
    scaler = amp.GradScaler()
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        with tqdm(total=len(train_loader)) as pbar:
            for images, masks in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))
                # gpu 연산을 위해 device 할당
                images, masks = images.cuda(), masks.cuda()
                model = model.cuda()
                
                # inference
                with amp.autocast():
                    outputs = model(images)
                    
                    # restore original size for hrnet_ocr
                    output_h, output_w = outputs.size(-2), outputs.size(-1)
                    mask_h, mask_w = masks.size(-2), masks.size(-1)
                    if output_h != mask_h or output_w != mask_w:
                        outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")

                    loss = criterion(outputs, masks)
                train_loss += loss.item()
                optimizer.zero_grad()
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # loss.backward()
                # optimizer.step()
                pbar.update(1)
                train_dict ={
                    'train loss': loss.item()
                }
                pbar.set_postfix(train_dict)

            scheduler.step()

        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % 1 == 0:
            print(f'Start validation #{(epoch+1):2d}')
            model.eval()

            dices = []
            with tqdm(total=len(valid_loader)) as pbar:
                with torch.no_grad():
                    n_class = len(XRayDataset.CLASSES)
                    val_loss = 0
                    cnt = 0
                    for images, masks in valid_loader:
                        pbar.set_description('[Epoch {}]'.format(epoch + 1))
                        images, masks = images.cuda(), masks.cuda()         
                        model = model.cuda()
                        
                        outputs = model(images)
                        
                        output_h, output_w = outputs.size(-2), outputs.size(-1)
                        mask_h, mask_w = masks.size(-2), masks.size(-1)
                        
                        # restore original size
                        if output_h != mask_h or output_w != mask_w:
                            outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
                        
                        loss = criterion(outputs, masks)
                        val_loss += loss.item()
                        cnt += 1
                        outputs = torch.sigmoid(outputs)
                        outputs = (outputs > thres).detach().cpu()
                        masks = masks.detach().cpu()
                        
                        dice = dice_coef(outputs, masks)
                        dices.append(dice)
                        
                        pbar.update(1)
                        val_dict ={
                            'val loss': loss.item()
                        }
                        pbar.set_postfix(val_dict)
                        #break
                            
                    dices = torch.cat(dices, dim=0)
                    dices_per_class = torch.mean(dices, dim=0)
                    dice_str = [
                        f"{c:<12}: {d.item():.4f}"
                        for c, d in zip(XRayDataset.CLASSES, dices_per_class)
                    ]
                    dice_str = "\n".join(dice_str)
                    print(dice_str)
                
                    avg_dice = torch.mean(dices_per_class).item()

                    print(avg_dice)
            
                    trial.report(avg_dice, epoch)

                    # Handle pruning based on the intermediate value.
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned() 

    return avg_dice

# Sampler : hyper-parameter를 찾는 방법

# Option 1 (bayesian optimization 방법)
# 목적함수의 대략적은 형태에 대한 확률추정 모델로써 Gaussian Process 확률 모델을 사용
# from optuna.integration import SkoptSampler
# sampler = SkoptSampler(
#     skopt_kwargs={'n_random_starts':5,
#                   'acq_func':'EI',
#                   'acq_func_kwargs': {'xi':0.02}})

# Option 2 (TPE 방법)
# Sampler using TPE (Tree-structured Parzen Estimator) algorithm.

from optuna.samplers import TPESampler
sampler = TPESampler()


study = optuna.create_study(direction="maximize", sampler=sampler)

# n_trials 지정없으면 무한 반복
study.optimize(objective, n_trials=10)
pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])


print(f"Study statistics: ")
print(f"Number of finished trials: {len(study.trials)}")
print(f"Number of pruned trials: {len(pruned_trials)}")
print(f"Number of complete trials: {len(complete_trials)}")


# SAVE
import joblib
joblib.dump(study, "optuna_tuning_model.pkl")

print("Best trial until now:")
print(" Value: ", study.best_trial.value)
print(" Params: ")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")


# 파라미터 중요도 확인 그래프
optuna.visualization.plot_param_importances(study)
