import os

import torch
from nets.unet_training import CE_Loss, Dice_loss, Focal_Loss
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score
import cv2
import numpy as np
from dss import SW_datasets_utils as SW


def fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, save_period, save_dir):
    ss = SW.SW('datas')

    total_loss      = 0
    total_f_score   = 0
    train_sPA = 0
    train_siou = 0

    val_loss        = 0
    val_f_score     = 0
    val_sPA = 0
    val_siou = 0

    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            imgs, pngs, labels = batch

            with torch.no_grad():
                imgs    = torch.from_numpy(imgs).type(torch.FloatTensor)
                pngs    = torch.from_numpy(pngs).squeeze(1).long()
                labels  = torch.from_numpy(labels).type(torch.FloatTensor)
                weights = torch.from_numpy(cls_weights)
                if cuda:
                    imgs    = imgs.cuda()
                    pngs    = pngs.cuda()
                    labels  = labels.cuda()
                    weights = weights.cuda()

            optimizer.zero_grad()

            outputs = model_train(imgs)
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss      = loss + main_dice

            with torch.no_grad():
                _f_score = f_score(outputs, labels)

                n, c, h, w = outputs.cpu().size()
                miou = 0
                mPA  = 0
                for i in range(n):
                    t_png = np.asarray(pngs[i].cpu())
                    t_output = torch.nn.functional.softmax(outputs[i].permute(1, 2, 0), dim=-1).cpu().numpy()
                    t_output = np.argmax(t_output,-1)
                    accuracy_count= ss.accuracy_count(t_png, t_output)#output , iou , accuracy , precision , recall ,(TP,TN,FP,FN)
                    mPA  += accuracy_count["accuracy"]
                    miou += accuracy_count["iou"]

                train_sPA  = train_sPA +  mPA / n
                train_siou = train_siou + miou / n

            loss.backward()
            optimizer.step()

            total_loss      += loss.item()
            total_f_score   += _f_score.item()
            
            pbar.set_postfix(**{'miou'      : train_siou / (iteration + 1),
                                'mPA'       : train_sPA  / (iteration + 1),
                                'total_loss': total_loss / (iteration + 1),
                                'f_score'   : total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')

    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            imgs, pngs, labels = batch
            with torch.no_grad():
                imgs    = torch.from_numpy(imgs).type(torch.FloatTensor)
                pngs    = torch.from_numpy(pngs).squeeze(1).long()
                labels  = torch.from_numpy(labels).type(torch.FloatTensor)
                weights = torch.from_numpy(cls_weights)
                if cuda:
                    imgs    = imgs.cuda()
                    pngs    = pngs.cuda()
                    labels  = labels.cuda()
                    weights = weights.cuda()

                outputs     = model_train(imgs)
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss  = loss + main_dice

                _f_score    = f_score(outputs, labels)

                miou = 0
                mPA  = 0
                n, c, h, w = outputs.cpu().size()
                for i in range(n):
                    t_png = np.asarray(pngs[i].cpu())
                    t_output = torch.nn.functional.softmax(outputs[i].permute(1, 2, 0), dim=-1).cpu().numpy()
                    t_output = np.argmax(t_output,-1)
                    accuracy_count = ss.accuracy_count(t_png,t_output,False)
                    mPA += accuracy_count["accuracy"]
                    miou += accuracy_count["iou"]

                val_sPA  = val_sPA +  mPA / n
                val_siou = val_siou + miou / n

                val_loss    += loss.item()
                val_f_score += _f_score.item()
                
            
            pbar.set_postfix(**{'miou'      : val_siou / (iteration + 1),
                                'mPA'       : val_sPA  / (iteration + 1),
                                'total_loss': val_loss / (iteration + 1),
                                'f_score'   : val_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    loss_history.append_loss(total_loss / (epoch_step + 1), val_loss / (epoch_step_val + 1),
                             train_sPA / (epoch_step + 1), val_sPA / (epoch_step_val + 1),
                             train_siou / (epoch_step + 1), val_siou / (epoch_step_val + 1))

    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Train mIOU: {:.2%} || Val mIOU: {:.2%}'.format(train_siou / (epoch_step + 1), val_siou / (epoch_step_val + 1)))
    print('Train mPA : {:.2%} || Val mPA:  {:.2%}'.format(train_sPA  / (epoch_step + 1), val_sPA  / (epoch_step_val + 1)))
    print('Total Loss: {:.3}  || Val Loss: {:.3} '.format(total_loss / (epoch_step + 1), val_loss / (epoch_step_val + 1)))
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(model.state_dict(),
                   'logs/ep{}-tIOU{:.2%}-vIOU{:2.2%}-tloss{:.3}-vloss{:.3}-tPA{:.2%}-vPA{:.2%}.pth'.format(
                       epoch + 1, train_siou / (epoch_step + 1), val_siou / (epoch_step_val + 1),
                       total_loss / (epoch_step + 1), val_loss / (epoch_step_val + 1),
                       train_sPA / (epoch_step + 1), val_sPA / (epoch_step_val + 1)
                   ))

