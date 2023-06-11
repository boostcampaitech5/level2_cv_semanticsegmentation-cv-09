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
    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 50)')
    
    # data
    parser.add_argument("--resize", nargs="+", type=int, default=[512, 512], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training (default: 8)')
    parser.add_argument('--valid_batch_size', type=int, default=2, help='input batch size for validing (default: 2)')
    
    # model
    parser.add_argument('--model', type=str, default='FcnResnet50', help='model name (default: FcnResnet50)')
    parser.add_argument('--early_stopping', type=int, default = 5, help='input early stopping patience, It does not work if you input -1, default : 5')

    # optimizer
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--lr_decay_step', type=int, default=5, help='learning rate scheduler deacy step (default: 5)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer such as SGD, Momentum, Adam, Adagrad (default: adam)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay (default: 1e-6)')
    parser.add_argument('--loss', type=str, default='bce', help='[bce, focal, dice, iou, combine: (default: bce)')

    # scheduler
    parser.add_argument('--scheduler', type=str, default='steplr', help='scheduler such as steplr, lambdalr, exponentiallr, cycliclr, reducelronplateau etc. (default: steplr)')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate scheduler gamma (default: 0.5)')
    parser.add_argument('--tmax', type=int, default=5, help='tmax used in CyclicLR and CosineAnnealingLR (default: 5)')
    parser.add_argument('--maxlr', type=float, default=0.1, help='maxlr used in CyclicLR (default: 0.1)')
    parser.add_argument('--mode', type=str, default='triangular', help='mode used in CyclicLR such as triangular, triangular2, exp_range (default: triangular)')
    parser.add_argument('--factor', type=float, default=0.5, help='mode used in ReduceLROnPlateau (default: 0.5)')
    parser.add_argument('--patience', type=int, default=4, help='mode used in ReduceLROnPlateau (default: 4)')
    parser.add_argument('--threshold', type=float, default=1e-4, help='mode used in ReduceLROnPlateau (default: 1e-4)')

    # # loss
    # parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--dice_thr', type=float, default=0.5, help='dice loss threshold (default: 0.5)')
    
    # log
    parser.add_argument('--wandb', action='store_true', help='wandb logging')
    parser.add_argument('--project',type=str, default='seg_baseline')
    parser.add_argument('--name',type=str, default='base')
    parser.add_argument('--val_interval', type=int, default=1, help='evaluate interval (default: 1)')
    parser.add_argument('--viz_img_path', type=str, default='train/DCM/ID001/image1661130828152_R.png')

    # Container environment
    parser.add_argument('--data_dir', type=str, default='/opt/ml')
    parser.add_argument('--save_dir', type=str, default='./checkpoint')
    parser.add_argument('--save_name', type=str, default='best.pt')

    args = parser.parse_args()
    
    return args

if __name__=="__main__":
    
    args = get_args()
    
    if args.wandb:
        wandb.init(
            entity = 'boost_cv_09',
            project = args.project,
            name = args.name,
            config = args
        )
        
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
    
    model_module = getattr(import_module("models.my_model"), args.model)  # default: BaseModel
    model = model_module(
        num_classes=len(train_dataset.CLASSES)
    )

    
    # Loss function 정의
    criterion = create_criterion(args.loss)
    
    # Optimizer 정의
    optim_module = getattr(import_module("torch.optim"), args.optimizer)  # default: adam
    optimizer = optim_module(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
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

        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % args.val_interval == 0:
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
                        outputs = (outputs > args.dice_thr).detach().cpu()
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
                
            if best_dice < avg_dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {avg_dice:.4f}")
                best_dice = avg_dice
                save_model(model,args.save_dir, args.save_name)
                
        if args.wandb:
            if (epoch+1) < args.val_interval:
                val_loss = 0
                avg_dice = 0

            metric_info = {
                'lr/lr' : optimizer.param_groups[0]['lr'],
                'train/loss' : train_loss/len(train_loader),
                'val/loss' : val_loss/len(valid_loader),
                'val/dice' : avg_dice,
            }
            
            class_data = [value.item() for value in dices_per_class]
            plt.bar([i for i in range(len(XRayDataset.CLASSES))], class_data)
            
            metric_info['dice_hist'] = wandb.Image(plt)
            # logging visualize output - by kyungbong 
            viz_image, viz_preds = viz_img(os.path.join(args.data_dir, args.viz_img_path), model, args.dice_thr, args.resize)
            fig, ax = plt.subplots(1, 2, figsize=(24, 12))
            ax[0].imshow(viz_image)    # remove channel dimension
            ax[1].imshow(viz_preds)
            metric_info['viz_img'] = wandb.Image(plt)

            wandb.log(metric_info, step=epoch)
            plt.clf()
            plt.close('all')
