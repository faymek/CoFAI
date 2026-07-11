import logging
import os
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from .metrics import R1_mAP_eval
# from torch.cuda import amp
import torch.amp as amp

import torch.distributed as dist
import random
import contextlib

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             val_loader_multi,
             query_loader_multi,
             gallery_loader_multi,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query_single,
             num_query_multi,
             local_rank):


    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD


    device = 'cuda'
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)


    loss_meter = AverageMeter()
    loss_main_meter = AverageMeter()
    loss_prompt_meter = AverageMeter()
    acc_meter = AverageMeter()


    scaler = amp.GradScaler()
    # scaler = amp.GradScaler(enabled=False)

    # train
    for epoch in range(1, epochs + 1):

        start_time = time.time()

        loss_meter.reset()
        loss_main_meter.reset()
        loss_prompt_meter.reset()

        acc_meter.reset()


        scheduler.step(epoch)
        model.train()


        total_batch_time = 0.0
        batch_count = 0


        for n_iter, (img, vid, target_cam, target_view, target_scenario) in enumerate(train_loader):

            batch_start = time.time()

            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            target_scenario = target_scenario.to(device)

            with amp.autocast(device_type='cuda', enabled=True):
            # with amp.autocast(device_type='cuda', enabled=False):

                if cfg.descrip == 'single_view': # trans-reid
                    score, feat, _ = model(img, target, cam_label=target_cam, view_label=target_view )
                    loss = loss_fn(score, feat, target, target_cam)

                elif 'transreid_baseline' in cfg.descrip:
                    # control gradient synchronization.
                    context = model.no_sync() if cfg.MODEL.DIST_TRAIN else contextlib.suppress()
                    with context:
                        score, feat, _, _, _ = model(img, target, cam_label=target_cam, view_label=target_view)
                        if cfg.MODEL.DIST_TRAIN:
                            # Create a dummy loss from the first forward pass to retain the graph
                            dummy_loss = sum(s.sum() for s in score) * 0
                            scaler.scale(dummy_loss).backward(retain_graph=True)

                    score2, feat2, lsort, flops, _ = model(img, target, cam_label=target_cam, view_label=target_view, multi_view=True)

                    # Now calculate the real, final loss
                    loss = loss_fn([torch.cat([s,s2], dim=0) for s,s2 in zip(score, score2)], [torch.cat([f,f2], dim=0) for f,f2 in zip(feat, feat2)], torch.cat([target,target[lsort]],dim=0),  \
                                   torch.cat([target_cam,target_cam[lsort]],dim=0), cross=False, onlytriplet=False, length=None)

                    loss_main = loss
                    # The final backward call will trigger the sync for all parameters
                    scaler.scale(loss).backward()


                elif 'gps_multi_view' in cfg.descrip:
                    context = model.no_sync() if cfg.MODEL.DIST_TRAIN else contextlib.suppress()
                    with context:
                        score, feat, _, _, aux_logits_single = model(img, target, cam_label=target_cam, view_label=target_view)
                        loss1 = loss_fn(score, feat, target, target_cam)
                        scaler.scale(loss1).backward(retain_graph=True)

                    score2, feat2, lsort, flops, aux_logits_multi = model(img, target, cam_label=target_cam, view_label=target_view, multi_view=True)
                    loss2 = loss_fn(score2, feat2, target[lsort], target_cam[lsort])
                    loss_cross = loss_fn([torch.cat([s,s2], dim=0) for s,s2 in zip(score, score2)], [torch.cat([f,f2], dim=0) for f,f2 in zip(feat, feat2)], torch.cat([target,target[lsort]],dim=0),  \
                                        torch.cat([target_cam,target_cam[lsort]],dim=0), cross=True, onlytriplet=True, length=len(score[0]))

                    loss_remaining = (loss2 / 2.) + loss_cross
                    scaler.scale(loss_remaining).backward()

                    loss_main = (loss1.detach() + loss2) / 2. + loss_cross
                    loss = loss_main


            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            loss_main_meter.update(loss_main.item(), img.shape[0])

            acc_meter.update(acc, 1)

            torch.cuda.synchronize()

            batch_time = time.time() - batch_start
            total_batch_time += batch_time
            batch_count += 1
            avg_batch_time = total_batch_time / batch_count

            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Loss_main: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}, AvgBatchTime: {:.3f}s"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, loss_main_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0], avg_batch_time))


        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            torch.cuda.empty_cache()
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        torch.cuda.empty_cache()

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    # do_inference(cfg,
                    #     model,
                    #     val_loader,
                    #     False,
                    #     num_query)
                    # torch.cuda.empty_cache()
                    print("================================== Multi-View ==================================")
                    do_inference(cfg,
                            model,
                            val_loader_multi,
                            query_loader_multi,
                            gallery_loader_multi,
                            multi_view = True,
                            num_query = num_query_multi,
                            flip_view=False)
                    torch.cuda.empty_cache()
            else:
                # print("================================== Single View ==================================")
                # do_inference(cfg,
                #         model,
                #         val_loader,
                #         None,
                #         None,
                #         multi_view = False,
                #         num_query = num_query_single)
                # torch.cuda.empty_cache()

                print("================================== Multi-View ==================================")
                do_inference(cfg,
                        model,
                        val_loader_multi,
                        query_loader_multi,
                        gallery_loader_multi,
                        multi_view = True,
                        num_query = num_query_multi,
                        flip_view=False)
                torch.cuda.empty_cache()



def do_inference(cfg,
                 model,
                 val_loader,
                 query_loader,
                 gallery_loader,
                 multi_view,
                 num_query,
                 flip_view=False):


    if cfg.MODEL.DIST_TRAIN and dist.get_rank() != 0:
        return

    if cfg.MODEL.DIST_TRAIN:
        model_for_eval = model.module
    else:
        model_for_eval = model

    device = "cuda"

    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing on rank 0")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()

    model_for_eval.to(device)
    model_for_eval.eval()


    img_path_list = []
    with torch.no_grad():
        if multi_view:

            for my_loader in [query_loader, gallery_loader]:
                if my_loader == gallery_loader:
                    dataset_name = 'gallery'
                if my_loader == query_loader:
                    dataset_name = 'query'

                for n_iter, (img, pid, camid, camids, target_view, sceneid, imgpath) in enumerate(tqdm(my_loader)):
                    img = img.to(device)

                    camids = camids.to(device)
                    target = torch.tensor(pid, dtype=torch.int64).to(device)
                    target_view = target_view.to(device)
                    prof_cm = nullcontext()


                    feat, lsort = model_for_eval(img, cam_label=camids, view_label=target_view, label=target,
                                                 multi_view=multi_view,flip_view=flip_view, extra_token=False, dataset_name=dataset_name)


                    lsort = lsort.detach().cpu().numpy()
                    pid, camid = np.array(pid)[lsort], np.array(camid)[lsort]
                    evaluator.update((feat, pid, camid))
                    img_path_list.extend(imgpath)
        else:
            for n_iter, (img, pid, camid, camids, target_view, sceneid, imgpath) in enumerate(tqdm(val_loader)):
                img = img.to(device)
                camids = camids.to(device)
                target = torch.tensor(pid, dtype=torch.int64).to(device)
                target_view = target_view.to(device)

                feat, lsort = model_for_eval(img, cam_label=camids, view_label=target_view, label=target, multi_view=multi_view,flip_view=flip_view)

                evaluator.update((feat, pid, camid))
                img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.3%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.3%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


def evaluate_model(cfg, model, query_loader, gallery_loader, num_query):
    """Run the multi-view retrieval path and return machine-readable metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()
    start = time.perf_counter()
    with torch.inference_mode():
        for loader, dataset_name in ((query_loader, "query"), (gallery_loader, "gallery")):
            for img, pid, camid, camids, view, sceneid, _ in loader:
                img = img.to(device, non_blocking=True)
                camids = camids.to(device, non_blocking=True)
                target_view = view.to(device, non_blocking=True)
                target = torch.as_tensor(pid, dtype=torch.long, device=device)
                feat, order = model(
                    img,
                    cam_label=camids,
                    view_label=target_view,
                    label=target,
                    multi_view=True,
                    flip_view=False,
                    extra_token=False,
                    dataset_name=dataset_name,
                )
                order = order.detach().cpu().numpy()
                evaluator.update((feat, np.asarray(pid)[order], np.asarray(camid)[order]))
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    return {
        "mAP": float(mAP),
        "rank1": float(cmc[0]),
        "rank5": float(cmc[4]),
        "rank10": float(cmc[9]),
        "num_query": int(num_query),
        "elapsed_seconds": float(elapsed),
    }
