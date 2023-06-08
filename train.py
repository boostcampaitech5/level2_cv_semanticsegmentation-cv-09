from dataset.dataset import split_dataset, XRayDataset
from dataset.transforms import get_train_transform
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

def get_args():
    parser = argparse.ArgumentParser()
    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train (default: 30)')
    
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
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer such as sgd, momentum, adam, adagrad (default: sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay (default: 1e-6)')
    
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
    parser.add_argument('--log_interval', type=int, default=25, help='train log interval (default: 25)')
    parser.add_argument('--viz_img_path', type=str, default='train/DCM/ID001/image1661130828152_R.png')

    # Container environment
    parser.add_argument('--data_dir', type=str, default='/opt/ml')
    parser.add_argument('--save_dir', type=str, default='/opt/ml/level2_cv_semanticsegmentation-cv-09/checkpoint')
    parser.add_argument('--save_name', type=str, default='fcn_resnet50_best.pt')

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
    
    train_transform = get_train_transform(train=True)

    val_transform = get_train_transform(train=False)

    
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

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
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
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer 정의
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    print(f'Start training..')
    
    n_class = len(XRayDataset.CLASSES)
    best_dice = 0.
    
    # AMP : loss scale을 위한 GradScaler 생성
    scaler = amp.GradScaler()
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for step, (images, masks) in tqdm(enumerate(train_loader),total=len(train_loader)):
            # gpu 연산을 위해 device 할당
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()
            
            with amp.autocast():
                # inference
                outputs = model(images)
                # loss 계산
                loss = criterion(outputs, masks)
            
            train_loss += loss.item()
            optimizer.zero_grad()
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss.backward()
            # optimizer.step()
            
            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print(
                    f'Epoch [{epoch+1}/{args.epochs}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )
                
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % args.val_interval == 0:
            print(f'Start validation #{(epoch+1):2d}')
            model.eval()

            dices = []
            with torch.no_grad():
                n_class = len(XRayDataset.CLASSES)
                val_loss = 0
                cnt = 0

                for step, (images, masks) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                    images, masks = images.cuda(), masks.cuda()         
                    model = model.cuda()
                    
                    outputs = model(images)
                    
                    output_h, output_w = outputs.size(-2), outputs.size(-1)
                    mask_h, mask_w = masks.size(-2), masks.size(-1)
                    
                    # restore original size
                    if output_h != mask_h or output_w != mask_w:
                        outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
                    
                    loss = criterion(outputs, masks)
                    val_loss += loss
                    cnt += 1
                    
                    outputs = torch.sigmoid(outputs)
                    outputs = (outputs > args.dice_thr).detach().cpu()
                    masks = masks.detach().cpu()
                    
                    dice = dice_coef(outputs, masks)
                    dices.append(dice)
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
            viz_image, viz_preds = viz_img(os.path.join(args.data_dir, args.viz_img_path), model, args.dice_thr)
            fig, ax = plt.subplots(1, 2, figsize=(24, 12))
            ax[0].imshow(viz_image)    # remove channel dimension
            ax[1].imshow(viz_preds)
            metric_info['viz_img'] = wandb.Image(plt)

            wandb.log(metric_info, step=epoch)
            plt.clf()
            plt.close('all')