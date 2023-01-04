import json
import time
import torch
from data_loader_beta import MVTecDRAEMTrainDataset, MVTecDRAEMTestDataset
from torch.utils.data import DataLoader
from torch import optim
from tensorboard_visualizer import TensorboardVisualizer
from model import ReconstructiveSubNetwork_CBAM, DiscriminativeSubNetwork2, DiscriminativeSubNetwork
from loss import FocalLoss, SSIM
import os
import sys
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from config import obj_transform_dict

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train_on_device(obj_names, args):

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    for obj_name in obj_names:
        run_name = 'DRAEM_test_'+str(args.lr)+'_'+str(args.epochs)+'_bs'+str(args.bs)+"_"+obj_name+'_'+obj_transform_dict[obj_name]
        print(run_name)
        checkpoint_path_obj = os.path.join(args.checkpoint_path, run_name)
        if not os.path.exists(checkpoint_path_obj):
            os.makedirs(checkpoint_path_obj)
        visualizer = TensorboardVisualizer(log_dir=os.path.join(args.log_path, run_name+"/"))

        model = ReconstructiveSubNetwork_CBAM(in_channels=3, out_channels=3)
        model.cuda()
        # model.apply(weights_init)

        model_seg = DiscriminativeSubNetwork2(in_channels=6, out_channels=2)
        model_seg.cuda()
        # model_seg.apply(weights_init)

        optimizer = torch.optim.AdamW([
                                      {"params": model.parameters(), "lr": args.lr},
                                      {"params": model_seg.parameters(), "lr": args.lr}])
        # optimizer = torch.optim.Adam([
        #                               {"params": model.parameters(), "lr": args.lr},
        #                               {"params": model_seg.parameters(), "lr": args.lr}])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[args.epochs*0.7,args.epochs*0.9],gamma=0.2, last_epoch=-1)

        # 从上次的暂停的模型中训练
        pause_point = os.path.join(checkpoint_path_obj, "pause_point.pckl")
        if os.path.exists(pause_point):
            state = torch.load(pause_point)
            start_epoch = state["epoch"]
            model.load_state_dict(state["model"])
            model_seg.load_state_dict(state["model_seg"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
        else:
            start_epoch = 0
        loss_l2 = torch.nn.modules.loss.MSELoss()
        loss_ssim = SSIM()
        loss_focal = FocalLoss()

        dataset = MVTecDRAEMTrainDataset(args.data_path + obj_name + "/train/good/", args.anomaly_source_path, obj_transform_dict[obj_name], resize_shape=[256, 256])

        dataloader = DataLoader(dataset, batch_size=args.bs,
                                shuffle=True, num_workers=2, pin_memory=True)

        val_dataset = MVTecDRAEMTestDataset(args.data_path + obj_name + "/test/", resize_shape=[256, 256])
        val_dataloader = DataLoader(val_dataset, batch_size=1,
                                shuffle=False, num_workers=2, pin_memory=True)
        n_iter = 0
        best_score = 0
        best_auroc = 0
        best_auroc_pixel = 0
        cnt_display = 0
        display_indices = np.random.choice(len(dataloader), 15, replace=False)

        display_anomaly_images = torch.zeros((16 ,3 ,256 ,256)).cuda()
        display_images = torch.zeros((16 ,3 ,256 ,256)).cuda()
        display_gt_images = torch.zeros((16 ,3 ,256 ,256)).cuda()
        display_out_masks = torch.zeros((16 ,1 ,256 ,256)).cuda()
        display_in_masks = torch.zeros((16 ,1 ,256 ,256)).cuda()
        for epoch in range(1+start_epoch, 1+args.epochs): 
            print("Epoch: "+str(epoch))
            model.train()
            model_seg.train()
            cnt_display = 0
            start = time.time()
            with tqdm(dataloader, desc=obj_name+'_'+obj_transform_dict[obj_name], file=sys.stdout) as iterator:
                for i_batch, sample_batched in enumerate(iterator):
                    gray_batch = sample_batched["image"].cuda()
                    aug_gray_batch = sample_batched["augmented_image"].cuda()
                    anomaly_mask = sample_batched["anomaly_mask"].cuda()

                    # 生成器与重建
                    gray_rec = model(aug_gray_batch)   

                    joined_in = torch.cat((gray_rec, aug_gray_batch), dim=1)

                    out_mask = model_seg(joined_in)
                    out_mask_sm = torch.softmax(out_mask, dim=1)

                    l2_loss = loss_l2(gray_rec,gray_batch)
                    ssim_loss = loss_ssim(gray_rec, gray_batch)

                    segment_loss = loss_focal(out_mask_sm, anomaly_mask)
                    loss = l2_loss + ssim_loss + segment_loss

                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()

                    if args.visualize and n_iter % 200 == 0:
                        visualizer.plot_loss(l2_loss, n_iter, loss_name='l2_loss')
                        visualizer.plot_loss(ssim_loss, n_iter, loss_name='ssim_loss')
                        visualizer.plot_loss(segment_loss, n_iter, loss_name='segment_loss')

                    if i_batch in display_indices:
                        t_mask = out_mask_sm[:, 1:, :, :]
                        display_anomaly_images[cnt_display] = aug_gray_batch[0]
                        display_images[cnt_display] = gray_rec[0]
                        display_gt_images[cnt_display] = gray_batch[0]
                        display_out_masks[cnt_display] = t_mask[0]
                        display_in_masks[cnt_display] = anomaly_mask[0]
                        cnt_display += 1
                    iterator.set_postfix_str(f"l2_loss: {l2_loss:.5f}, ssim_loss: {ssim_loss:.5f}, segment_loss: {segment_loss:.5f}")
                    n_iter +=1
                    del gray_rec
                    del out_mask
            scheduler.step()
            cur_lr = optimizer.param_groups[-1]['lr']
            visualizer.plot_loss(cur_lr, epoch, loss_name='lr')
            if epoch % 50 == 0:         
                visualizer.visualize_image_batch(display_anomaly_images, epoch, image_name='train/in_images')
                visualizer.visualize_image_batch(display_gt_images, epoch, image_name='train/gt_images')
                visualizer.visualize_image_batch(display_images, epoch, image_name='train/out_images')
                visualizer.visualize_image_batch(display_out_masks, epoch, image_name='train/out_masks')
                visualizer.visualize_image_batch(display_in_masks, epoch, image_name='train/gt_masks')

            eval_metrics = test_on_device(model, model_seg, val_dataloader, visualizer, epoch)
            end = time.time()
            print("the epoch cost time: ",end-start, "s")
            eval_log = dict([(f'eval_{k}', v) for k, v in eval_metrics.items()])
            eval_log.update({"best_epoch": epoch})
            # total_score = np.mean(list(eval_metrics.values()))
            total_score = (eval_metrics["AUROC-image"] + eval_metrics["AUROC-pixel"])/2
            if best_score < total_score:
                torch.save(model.state_dict(), os.path.join(checkpoint_path_obj, "best_total_model_rec.pckl"))
                torch.save(model_seg.state_dict(), os.path.join(checkpoint_path_obj, "best_total_model_seg.pckl"))
                with open(os.path.join(checkpoint_path_obj, "best_total_socre.json"),"w") as f:
                    json.dump(eval_log, f, indent="\t")
                print(f"best_total_score, {best_score:.5f} to {total_score:.5f}")
                best_score = total_score
            if best_auroc < eval_metrics["AUROC-image"]:
                torch.save(model.state_dict(), os.path.join(checkpoint_path_obj, "best_auroc_model_rec.pckl"))
                torch.save(model_seg.state_dict(), os.path.join(checkpoint_path_obj, "best_auroc_model_seg.pckl"))
                with open(os.path.join(checkpoint_path_obj, "best_auroc_socre.json"),"w") as f:
                    json.dump(eval_log, f, indent="\t")
                print(f"best_AUROC-image_score, {best_auroc:.5f} to {eval_metrics['AUROC-image']:.5f}")
                best_auroc = eval_metrics["AUROC-image"]
            if best_auroc_pixel < eval_metrics["AUROC-pixel"]:
                torch.save(model.state_dict(), os.path.join(checkpoint_path_obj, "best_auroc-pixel_model_rec.pckl"))
                torch.save(model_seg.state_dict(), os.path.join(checkpoint_path_obj, "best_auroc-pixel_model_seg.pckl"))
                with open(os.path.join(checkpoint_path_obj, "best_auroc-pixel_socre.json"),"w") as f:
                    json.dump(eval_log, f, indent="\t")
                print(f"best_AUROC-pixel_score, {best_auroc_pixel:.5f} to {eval_metrics['AUROC-pixel']:.5f}")
                best_auroc_pixel = eval_metrics["AUROC-pixel"]
            if epoch % 200 == 0:
                torch.save(model.state_dict(), os.path.join(checkpoint_path_obj, "epoch_"+str(epoch)+"_model_rec.pckl"))
                torch.save(model_seg.state_dict(), os.path.join(checkpoint_path_obj, "epoch_"+str(epoch)+"_model_seg.pckl"))
            if epoch == args.pause_epoch:
                state = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "model_seg": model_seg.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(state, pause_point)
                break

def test_on_device(model, model_seg, val_dataloader, visualizer, epoch, img_dim=256):
    print("============test==============")
    model.eval()
    model_seg.eval()
    with torch.no_grad():
        total_pixel_scores = np.zeros((img_dim * img_dim * len(val_dataloader)))
        total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(val_dataloader)))
        mask_cnt = 0

        anomaly_score_gt = []
        anomaly_score_prediction = []

        cnt_display = 0
        display_indices = np.random.choice(len(val_dataloader), 16, replace=False)

        display_images = torch.zeros((16 ,3 ,256 ,256)).cuda()
        display_gt_images = torch.zeros((16 ,3 ,256 ,256)).cuda()
        display_out_masks = torch.zeros((16 ,1 ,256 ,256)).cuda()
        display_in_masks = torch.zeros((16 ,1 ,256 ,256)).cuda()
        for i_batch, sample_batched in enumerate(tqdm(val_dataloader)):

            gray_batch = sample_batched["image"].cuda()

            is_normal = sample_batched["has_anomaly"].detach().numpy()[0 ,0]
            anomaly_score_gt.append(is_normal)
            true_mask = sample_batched["mask"]
            true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))

            gray_rec = model(gray_batch)
            joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

            out_mask = model_seg(joined_in)
            out_mask_sm = torch.softmax(out_mask, dim=1)

            out_mask_cv = out_mask_sm[0 ,1 ,: ,:].detach().cpu().numpy()

            out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[: ,1: ,: ,:], 21, stride=1,
                                                               padding=21 // 2).cpu().detach().numpy()
            image_score = np.max(out_mask_averaged)

            anomaly_score_prediction.append(image_score)

            flat_true_mask = true_mask_cv.flatten()
            flat_out_mask = out_mask_cv.flatten()
            total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
            total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
            mask_cnt += 1

            if i_batch in display_indices:
                t_mask = out_mask_sm[:, 1:, :, :]
                display_images[cnt_display] = gray_rec[0]
                display_gt_images[cnt_display] = gray_batch[0]
                display_out_masks[cnt_display] = t_mask[0]
                display_in_masks[cnt_display] = true_mask[0]
                cnt_display += 1
        if epoch % 50 == 0:         
            visualizer.visualize_image_batch(display_gt_images, epoch, image_name='test/in_images')
            visualizer.visualize_image_batch(display_images, epoch, image_name='test/out_images')
            visualizer.visualize_image_batch(display_out_masks, epoch, image_name='test/out_masks')
            visualizer.visualize_image_batch(display_in_masks, epoch, image_name='test/gt_masks')

        anomaly_score_prediction = np.array(anomaly_score_prediction)
        anomaly_score_gt = np.array(anomaly_score_gt)
        auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
        ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)

        total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
        total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
        total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
        auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
        ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)
        print("AUC Image:  " +str(auroc))
        print("AP Image:  " +str(ap))
        print("AUC Pixel:  " +str(auroc_pixel))
        print("AP Pixel:  " +str(ap_pixel))
        print("==============================")
        visualizer.plot_loss(auroc, epoch, loss_name='auroc')
        visualizer.plot_loss(ap, epoch, loss_name='ap')
        visualizer.plot_loss(auroc_pixel, epoch, loss_name='auroc_pixel')
        visualizer.plot_loss(ap_pixel, epoch, loss_name='ap_pixel')
        metrics = {
            "AUROC-image": auroc,
            "AUROC-pixel": auroc_pixel,
            "AP-image": ap,
            "AP-pixel": ap_pixel,
        }
        return metrics

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', action='store', type=int, required=True)
    parser.add_argument('--bs', action='store', type=int, required=True)
    parser.add_argument('--lr', action='store', type=float, required=True)
    parser.add_argument('--epochs', action='store', type=int, required=True)
    parser.add_argument('--gpu_id', action='store', type=int, default=0, required=False)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--anomaly_source_path', action='store', type=str, required=True)
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
    parser.add_argument('--log_path', action='store', type=str, required=True)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--pause_epoch',  action='store', type=int, default=0, required=False)

    args = parser.parse_args()

    obj_batch = [['capsule'],
                 ['bottle'],
                 ['carpet'],
                 ['leather'],
                 ['pill'],
                 ['transistor'],
                 ['tile'],
                 ['cable'],
                 ['zipper'],
                 ['toothbrush'],
                 ['metal_nut'],
                 ['hazelnut'],
                 ['screw'],
                 ['grid'],
                 ['wood']
                 ]

    if int(args.obj_id) == -1:
        obj_list = [
                    'capsule',
                     'bottle',
                     'carpet',
                     'leather',
                     'pill',
                     'transistor',
                     'tile',
                     'cable',
                     'zipper',
                     'toothbrush',
                     'metal_nut',
                     'hazelnut',
                     'screw',
                     'grid',
                     'wood'
                     ]
        picked_classes = obj_list
    else:
        picked_classes = obj_batch[int(args.obj_id)]

    with torch.cuda.device(args.gpu_id):
        train_on_device(picked_classes, args)

