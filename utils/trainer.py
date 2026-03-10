import torch
import time
from tqdm import tqdm
from torch.cuda.amp import autocast
import torch.nn.functional as F
import numpy as np
import gc
from utils.util import AverageMeter


def format_loss_display(loss_value):
    """Dynamically format loss display to preserve precision for small values."""
    if loss_value < 1e-6:
        return "{:.9f}".format(loss_value)  # Show extremely tiny losses with 9 decimals
    elif loss_value < 1e-4:
        return "{:.8f}".format(loss_value)  # Show very small losses with 8 decimals
    elif loss_value < 1e-2:
        return "{:.6f}".format(loss_value)  # Show small losses with 6 decimals
    else:
        return "{:.4f}".format(loss_value)  # Show regular losses with 4 decimals



def train(config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None, xbm_helper=None, epoch=1):
    # set model train mode
    model.train()

    losses = AverageMeter()
    
    # XBM-related loss tracking
    if xbm_helper is not None:
        xbm_losses = AverageMeter()
        # current_iter accumulate across epochs, not reset per epoch
        current_iter = (epoch - 1) * len(dataloader)
        
        # show XBM status at the start of each epoch
        print(f"\n{'='*60}")
        print(f" XBM状态监控 - Epoch {epoch}")
        print(f"{'='*60}")
        print(f"  当前迭代起始: {current_iter}")
        print(f"  本epoch迭代数: {len(dataloader)}")
        print(f"  本epoch迭代范围: [{current_iter}, {current_iter + len(dataloader) - 1}]")
        print(f"  XBM启动迭代: {xbm_helper.start_iter}")
        print(f"  XBM队列大小: {xbm_helper.get_queue_size()}/{xbm_helper.xbm.memory_size}")
        
        if current_iter >= xbm_helper.start_iter:
            queue_fill_rate = xbm_helper.get_queue_size() / xbm_helper.xbm.memory_size * 100
            print(f"  🟢 XBM状态: 已启动并运行中 (队列填充率: {queue_fill_rate:.1f}%)")
        elif current_iter + len(dataloader) > xbm_helper.start_iter:
            will_start_at = xbm_helper.start_iter - current_iter
            print(f"  🟡 XBM状态: 本epoch将在第 {will_start_at} 步启动！")
        else:
            epochs_until_start = (xbm_helper.start_iter - current_iter) // len(dataloader) + 1
            print(f"  🔴 XBM状态: 未启动 (还需约 {epochs_until_start} 个epoch)")
        print(f"{'='*60}\n")

    # wait before starting progress bar
    time.sleep(0.1)

    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)

    step = 1

    if config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    # for loop over one epoch
    for query, reference, ids in bar:

        if scaler:
            with autocast():

                # data (batches) to device
                query = query.to(config.device)
                reference = reference.to(config.device)

                # Forward pass
                features1, features2 = model(query, reference)
                
                # Debug: check DataParallel output feature shapes (first step only)
                if step == 1:
                    print(f"\n 调试信息 - DataParallel输出:")
                    print(f"   features1.shape: {features1.shape}")
                    print(f"   features2.shape: {features2.shape}")
                
                # Handle loss-function return value (tuple or tensor)
                result = loss_function(features1, features2)
                if isinstance(result, tuple):
                    loss, loss_dict = result
                    # Debug: show loss breakdown (first few steps)
                    if step <= 3:
                        print(f"\n 损失分解 (step {step}):")
                        for key, value in loss_dict.items():
                            print(f"   {key}: {value:.4f}")
                else:
                    loss = result
            
            # move XBM processing outside autocast to avoid AMP issues
            if xbm_helper is not None:
                # Label-aware XBM: pass ids parameter
                xbm_result = xbm_helper.compute_and_update(loss_function, features1, features2, ids, current_iter)
                
                # Support two return formats (including added drone intra-domain loss)
                if isinstance(xbm_result, tuple):
                    xbm_loss, xbm_loss_dict = xbm_result
                else:
                    xbm_loss = xbm_result
                    xbm_loss_dict = {'xbm_cross': xbm_loss.item() if xbm_loss.item() != 0 else 0.0}
                
                # compare with .item() to ensure consistent types
                xbm_loss_value = xbm_loss.item()
                
                # show XBM info after loss breakdown (first few steps)
                if step <= 3:
                    print(f"\n XBM调试信息 (step {step}):")
                    print(f"   当前迭代: {current_iter}")
                    print(f"   XBM启动迭代: {xbm_helper.start_iter}")
                    print(f"   XBM状态: {'🟢 已启动' if current_iter >= xbm_helper.start_iter else '🔴 未启动'}")
                    print(f"   XBM队列大小: {xbm_helper.get_queue_size()}/{xbm_helper.xbm.memory_size}")
                    print(f"   XBM损失: {xbm_loss_value:.6f}")
                    # show component losses (if drone intra-domain XBM is enabled)
                    if 'xbm_drone2drone' in xbm_loss_dict:
                        print(f"   ├─ 跨域损失: {xbm_loss_dict['xbm_cross']:.6f}")
                        print(f"   └─ Drone同域损失: {xbm_loss_dict['xbm_drone2drone']:.6f}")
                    if xbm_loss_value > 0:
                        print(f"    XBM正在工作中！")
                
                if xbm_loss_value != 0.0:
                    loss = loss + xbm_loss
                    xbm_losses.update(xbm_loss_value)
                current_iter += 1
            
            losses.update(loss.item())

            scaler.scale(loss).backward()

            # Gradient clipping
            if config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), config.clip_grad)

                # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()

            # Scheduler (call after optimizer step and zero_grad)
            if scheduler is not None:
                scheduler.step()

        else:

            # data (batches) to device
            query = query.to(config.device)
            reference = reference.to(config.device)

            # Forward pass
            features1, features2 = model(query, reference)
            
            # Check whether features contain NaN or Inf
            if torch.isnan(features1).any() or torch.isnan(features2).any():
                print(f"警告: 模型输出包含NaN值!")
                print(f"features1 NaN: {torch.isnan(features1).sum().item()}")
                print(f"features2 NaN: {torch.isnan(features2).sum().item()}")
            
            if torch.isinf(features1).any() or torch.isinf(features2).any():
                print(f"警告: 模型输出包含Inf值!")
            
            # Handle loss-function return value (tuple or tensor)
            result = loss_function(features1, features2)
            if isinstance(result, tuple):
                loss, loss_dict = result
            else:
                loss = result
            
            # XBM processing (managed in helper to avoid DataParallel issues)
            if xbm_helper is not None:
                # Label-aware XBM: pass ids parameter
                xbm_result = xbm_helper.compute_and_update(loss_function, features1, features2, ids, current_iter)
                
                # Support two return formats (including added drone intra-domain loss)
                if isinstance(xbm_result, tuple):
                    xbm_loss, xbm_loss_dict = xbm_result
                else:
                    xbm_loss = xbm_result
                    xbm_loss_dict = {'xbm_cross': xbm_loss.item() if xbm_loss.item() != 0 else 0.0}
                
                # compare with .item() to ensure consistent types
                xbm_loss_value = xbm_loss.item()
                
                # show XBM info after loss breakdown (first few steps)
                if step <= 3:
                    print(f"\n XBM调试信息 (step {step}):")
                    print(f"   当前迭代: {current_iter}")
                    print(f"   XBM启动迭代: {xbm_helper.start_iter}")
                    print(f"   XBM状态: {'🟢 已启动' if current_iter >= xbm_helper.start_iter else '🔴 未启动'}")
                    print(f"   XBM队列大小: {xbm_helper.get_queue_size()}/{xbm_helper.xbm.memory_size}")
                    print(f"   XBM损失: {xbm_loss_value:.6f}")
                    # show component losses (if drone intra-domain XBM is enabled)
                    if 'xbm_drone2drone' in xbm_loss_dict:
                        print(f"   ├─ 跨域损失: {xbm_loss_dict['xbm_cross']:.6f}")
                        print(f"   └─ Drone同域损失: {xbm_loss_dict['xbm_drone2drone']:.6f}")
                    if xbm_loss_value > 0:
                        print(f"   XBM正在工作中！")
                
                if xbm_loss_value != 0.0:
                    loss = loss + xbm_loss
                    xbm_losses.update(xbm_loss_value)
                current_iter += 1
            
            # Check loss value
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"损失值异常! loss = {loss.item()}")
                continue  # Skip this batch
                
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()

            # Gradient clipping
            if config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), config.clip_grad)

                # Update model parameters (weights)
            optimizer.step()
            
            # Zero gradients for next step
            optimizer.zero_grad()

            # Scheduler (call after optimizer step and zero_grad)
            if scheduler is not None:
                scheduler.step()

        if config.verbose:
            # Display losses with dynamic formatter
            loss_val = loss.item()
            loss_avg_val = losses.avg
            
            monitor = {
                "loss": format_loss_display(loss_val),
                "loss_avg": format_loss_display(loss_avg_val),
                "lr": "{:.6f}".format(optimizer.param_groups[0]['lr'])
            }
            bar.set_postfix(ordered_dict=monitor)

        step += 1

    if config.verbose:
        bar.close()
    
    # Show XBM statistics at the end of each epoch
    if xbm_helper is not None:
        print(f"\n{'='*60}")
        print(f" XBM Epoch {epoch} 统计")
        print(f"{'='*60}")
        print(f"  XBM状态: {'🟢 已启动' if xbm_helper.xbm_started else '🔴 未启动'}")
        print(f"  当前队列大小: {xbm_helper.get_queue_size()}/{xbm_helper.xbm.memory_size}")
        queue_fill_rate = xbm_helper.get_queue_size() / xbm_helper.xbm.memory_size * 100
        print(f"  队列填充率: {queue_fill_rate:.1f}%")
        if xbm_helper.xbm_started and xbm_losses.count > 0:
            print(f"  平均XBM损失: {xbm_losses.avg:.6f}")
            print(f"  XBM损失更新次数: {xbm_losses.count}")
        print(f"{'='*60}\n")

    return losses.avg


def predict(config, model, dataloader):
    model.eval()

    # wait before starting progress bar
    time.sleep(0.1)

    if config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    img_features_list = []

    ids_list = []
    with torch.no_grad():

        for img, ids in bar:

            ids_list.append(ids)

            with autocast():

                img = img.to(config.device)
                img_feature = model(img)

                # normalize is calculated in fp32
                if config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)

            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))

        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0)
        ids_list = torch.cat(ids_list, dim=0).to(config.device)

    if config.verbose:
        bar.close()

    return img_features, ids_list


def evaluate(config, model, query_loader, gallery_loader, ranks=[1, 5, 10], step_size=1000, cleanup=True, return_details=False):
    """
    Evaluate model performance.
    
    Args:
        return_details: If True, return details for re-ranking (features, wrong IDs, etc.).
    
    Returns:
        If return_details=False: return CMC[0] (R@1).
        If return_details=True: return (CMC[0], details_dict).
            details_dict includes: img_features_query, img_features_gallery,
                                   ids_query, ids_gallery, wrong_query_indices.
    """
    print("Extract Features:")
    img_features_query, ids_query = predict(config, model, query_loader)
    img_features_gallery, ids_gallery = predict(config, model, gallery_loader)

    gl = ids_gallery.cpu().numpy()
    ql = ids_query.cpu().numpy()

    print("Compute Scores:")

    CMC = torch.IntTensor(len(ids_gallery)).zero_()
    ap = 0.0
    wrong_query_indices = []  # Record query indices that fail at R@1
    
    for i in tqdm(range(len(ids_query))):
        ap_tmp, CMC_tmp = eval_query(img_features_query[i], ql[i], img_features_gallery, gl)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
        
        # Record R@1-failed queries
        if CMC_tmp[0] == 0:
            wrong_query_indices.append(i)

    AP = ap / len(ids_query) * 100

    CMC = CMC.float()
    CMC = CMC / len(ids_query)  # average CMC

    # top 1%
    top1 = round(len(ids_gallery) * 0.01)

    string = []

    for i in ranks:
        string.append('Recall@{}: {:.4f}'.format(i, CMC[i - 1] * 100))

    string.append('Recall@top1: {:.4f}'.format(CMC[top1] * 100))
    string.append('AP: {:.4f}'.format(AP))

    print(' - '.join(string))

    # Return details for re-ranking
    if return_details:
        details = {
            'img_features_query': img_features_query,
            'img_features_gallery': img_features_gallery,
            'ids_query': ids_query,
            'ids_gallery': ids_gallery,
            'wrong_query_indices': np.array(wrong_query_indices),
            'ql': ql,
            'gl': gl,
        }
        return CMC[0], details

    # cleanup and free memory on GPU
    if cleanup:
        del img_features_query, ids_query, img_features_gallery, ids_gallery
        gc.collect()

    return CMC[0]


def eval_query(qf, ql, gf, gl):
    score = gf @ qf.unsqueeze(-1)

    score = score.squeeze().cpu().numpy()

    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]

    # good index
    query_index = np.argwhere(gl == ql)
    good_index = query_index

    # junk index
    junk_index = np.argwhere(gl == -1)

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc
