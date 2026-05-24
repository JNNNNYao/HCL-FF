import os
import time
import argparse

import torch
import torch.nn.functional as F

from data.cifar10_dataloader import get_cifar10_dataloader
from data.cifar100_dataloader import get_cifar100_dataloader
from data.mnist_dataloader import get_mnist_dataloader
from data.fmnist_dataloader import get_fmnist_dataloader
from data.tinyimagenet200_dataloader import get_tinyimagenet200_dataloader

from model.resnet import Resnet
from model.resnet_tinyimagenet200 import ResnetTinyImageNet200

from thop import profile, clever_format

def test_model(model, test_loader, device):
    model.eval()
    total = 0
    correct_goodness = 0
    correct_softmax = 0
    correct_ensemble = 0
    for x, labels in test_loader:
        x, labels = x.to(device), labels.to(device)
        with torch.no_grad():
            y_goodness, y_softmax = model(x)
            y_goodness = y_goodness.to(device)
            y_softmax = y_softmax.to(device)
            pred_goodness = torch.argmax(y_goodness, dim=1)
            pred_softmax = torch.argmax(y_softmax, dim=1)
            pred_ensemble = torch.argmax(F.softmax(y_goodness, dim=1) + F.softmax(y_softmax, dim=1), dim=1)
            correct_goodness += torch.eq(pred_goodness, labels).sum().float().item()
            correct_softmax += torch.eq(pred_softmax, labels).sum().float().item()
            correct_ensemble += torch.eq(pred_ensemble, labels).sum().float().item()
        total += labels.size(0)
    test_acc_goodness = 100 * correct_goodness / total
    test_acc_softmax = 100 * correct_softmax / total
    test_acc_ensemble = 100 * correct_ensemble / total
    return test_acc_goodness, test_acc_softmax, test_acc_ensemble


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HCL-FF Training')
    parser.add_argument('--task', type=str, default='cifar100')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--train_batch_size', type=int, default=512)
    parser.add_argument('--test_batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.04)
    parser.add_argument('--lr_min', type=float, default=0.0002)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2222)
    parser.add_argument('--task_dir', type=str, default='cifar100_linear')
    parser.add_argument('--arch', type=str, default='resnet')
    parser.add_argument('--save_step', type=int, default=5)
    parser.add_argument('--data_dir', type=str, default='../deeperforward/datasets')
    parser.add_argument('--pretrain', action='store_true')
    args = parser.parse_args()

    epochs = args.epochs
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    lr = args.lr
    lr_min = args.lr_min
    weight_decay = args.weight_decay
    dropout = args.dropout

    if args.task_dir is None:
        task_dir = os.path.join('./results/', f'{time.strftime("%Y%m%d-%H%M%S")}_{args.task}_{args.arch}')
    else:
        task_dir = os.path.join('./results/', args.task_dir)

    if task_dir is not None:
        # create directory if not exists
        if not os.path.exists(task_dir):
            os.makedirs(task_dir)

    in_channels = 3
    num_class = 10
    start_epoch = 0

    root = args.data_dir
    img_size = 32

    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    seed = 2222
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    checkpoint_path = './checkpoint/checkpoint.pth'
    if task_dir is not None:
        checkpoint_path = os.path.join(task_dir, 'checkpoint.pth')

    def layer_to_level_mapping(D, method='linear'):
        if method == 'linear':
            levels = [min(1 + (i * D) // 17, D) for i in range(17)]
        elif method == 'increasing':
            levels = [min(1 + i, D) for i in range(17)]
        else:
            raise NotImplementedError
        return levels

    task = args.task
    train_loader, valid_loader, test_loader, hierarchy = None, None, None, None
    if task == 'cifar10':
        planes = [100, 200, 400, 800]
        train_loader, valid_loader, test_loader, hierarchy = get_cifar10_dataloader(root=root, train_batch_size=train_batch_size,
                                                                                    test_batch_size=test_batch_size, seed=seed)
        layer_to_level = layer_to_level_mapping(len(hierarchy), method='linear')
    elif task == 'cifar100':
        num_class = 100
        planes = [100, 200, 400, 800]
        train_loader, valid_loader, test_loader, hierarchy = get_cifar100_dataloader(root=root, train_batch_size=train_batch_size,
                                                                                     test_batch_size=test_batch_size, seed=seed)
        layer_to_level = layer_to_level_mapping(len(hierarchy), method='increasing')
    elif task == 'mnist':
        in_channels = 1
        planes = [40, 80, 160, 320] 
        train_loader, valid_loader, test_loader, hierarchy = get_mnist_dataloader(root=root, train_batch_size=train_batch_size,
                                                                       test_batch_size=test_batch_size, seed=seed)
        layer_to_level = layer_to_level_mapping(len(hierarchy), method='linear')
    elif task == 'fmnist':
        in_channels = 1
        planes = [40, 80, 160, 320]
        train_loader, valid_loader, test_loader, hierarchy = get_fmnist_dataloader(root=root, train_batch_size=train_batch_size,
                                                                        test_batch_size=test_batch_size, seed=seed)
        layer_to_level = layer_to_level_mapping(len(hierarchy), method='linear')
    elif task == 'tinyimagenet200':
        num_class = 200
        planes = [200, 400, 800, 1600]
        train_loader, valid_loader, test_loader, hierarchy = get_tinyimagenet200_dataloader(root=root, train_batch_size=train_batch_size,
                                                                                            test_batch_size=test_batch_size, seed=seed)
        layer_to_level = layer_to_level_mapping(len(hierarchy), method='increasing')
    else:
        raise NotImplementedError
    
    if args.pretrain:
        hierarchy = None
        layer_to_level = None
    print(f'Layer to level mapping: {layer_to_level}')

    # model
    model = None
    if args.arch == 'resnet':
        if task == 'tinyimagenet200':
            model = ResnetTinyImageNet200(in_channels=in_channels, num_class=num_class, dropout=dropout,
                                          planes=planes, learning_rate=lr, lr_min=lr_min,
                                          weight_decay=weight_decay, device=device, epochs=epochs,
                                          hierarchy=hierarchy, layer_to_level=layer_to_level)
        else:
            model = Resnet(in_channels=in_channels, num_class=num_class, dropout=dropout,
                           planes=planes, learning_rate=lr, lr_min=lr_min,
                           weight_decay=weight_decay, device=device, epochs=epochs,
                           hierarchy=hierarchy, layer_to_level=layer_to_level)
    else:
        raise NotImplementedError

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        start_epoch += 1
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        for i, optimizer in enumerate(model.optimizers):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'][i])
        for i, optimizer in enumerate(model.optimizers_proj_heads):
            optimizer.load_state_dict(checkpoint['optimizer_proj_heads_state_dict'][i])
        model.optimizer_clf_head.load_state_dict(checkpoint['optimizer_clf_head_state_dict'])
        for i, scheduler in enumerate(model.schedulers):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'][i])
        for i, scheduler in enumerate(model.schedulers_proj_heads):
            scheduler.load_state_dict(checkpoint['scheduler_proj_heads_state_dict'][i])
        model.scheduler_clf_head.load_state_dict(checkpoint['scheduler_clf_head_state_dict'])
        print(f'Load checkpoint from {checkpoint_path} start from epoch {start_epoch + 1}')

    input_data = torch.randn(1, in_channels, 32, 32, device=device)
    flops, params = profile(model, inputs=(input_data,))
    flops, params = clever_format([flops, params], "%.3f")
    print(f'FLOPs: {flops}, Params: {params}')

    train_acc, valid_acc, test_acc = 0., 0., 0.
    for epoch in range(start_epoch, epochs):

        start_time = time.time()
        model.train()
        losses = model.update(train_loader)
        with open(os.path.join(task_dir, 'losses.txt'), 'a') as f:
            f.write(f'Epoch {epoch}:\n')
            f.write(f'avg_supcon_loss: \t\t\t\t' + ','.join([f'{loss:.4f}' for loss in losses["avg_supcon_loss"]]) + '\n')
            f.write(f'avg_supcon_pos_neg_gap: \t\t' + ','.join([f'{loss:.4f}' for loss in losses["avg_supcon_pos_neg_gap"]]) + '\n')
            f.write(f'avg_supcon_top1_pos_rate: \t\t' + ','.join([f'{loss:.4f}' for loss in losses["avg_supcon_top1_pos_rate"]]) + '\n')
            f.write(f'avg_g_ce_loss: \t\t\t\t\t' + ','.join([f'{loss:.4f}' for loss in losses["avg_g_ce_loss"]]) + '\n')
            f.write(f'avg_head_ce_loss: \t\t\t\t' + f'{losses["avg_head_ce_loss"]:.4f}' + '\n')
        model.eval()
        train_time = time.time() - start_time

        # pruning
        pruning_time = time.time()
        valid_acc, before_acc = model.pruning(valid_loader)
        pruning_time = time.time() - pruning_time

        test_time = time.time()
        test_acc_goodness, test_acc_softmax, test_acc_ensemble = test_model(model, test_loader, device)
        test_time = time.time() - test_time

        if epoch % args.save_step == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': [optimizer.state_dict() for optimizer in model.optimizers],
                'optimizers_proj_heads_state_dict': [optimizer.state_dict() for optimizer in model.optimizers_proj_heads],
                'optimizer_clf_head_state_dict': model.optimizer_clf_head.state_dict(),
                'scheduler_state_dict': [scheduler.state_dict() for scheduler in model.schedulers],
                'scheduler_proj_heads_state_dict': [scheduler.state_dict() for scheduler in model.schedulers_proj_heads],
                'scheduler_clf_head_state_dict': model.scheduler_clf_head.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)

            train_acc_goodness, train_acc_softmax, train_acc_ensemble = test_model(model, train_loader, device)
            end_time = time.time() - start_time

            info = f'Epoch: {(epoch + 1):03d}/{epochs:03d}: ' \
                   f'Train Acc Goodness: {train_acc_goodness:.2f}% Train Acc Softmax: {train_acc_softmax:.2f}% Train Acc Ensemble: {train_acc_ensemble:.2f}% || Test training-set Time: {end_time:.2f}s'
            print(info)

            # saving the accuracy of training and testing into csv file
            if task_dir is not None:
                with open(os.path.join(task_dir, 'accuracy.csv'), 'a') as f:
                    f.write(f'{epoch},{train_acc_goodness:.2f},{train_acc_softmax:.2f},{train_acc_ensemble:.2f},{valid_acc:.2f},{test_acc_goodness:.2f},{test_acc_softmax:.2f},{test_acc_ensemble:.2f}\n')

        info = f'Epoch: {(epoch + 1):03d}/{epochs:03d}: ' \
               f'Pruning ({model.start_layer:02d}->{model.end_layer:02d}): ' \
               f'Valid Acc:{before_acc:.2f}% -> {valid_acc:.2f}% ' \
               f'Test Acc Goodness: {test_acc_goodness:.2f}% Test Acc Softmax: {test_acc_softmax:.2f}% Test Acc Ensemble: {test_acc_ensemble:.2f}% || ' \
               f'Train Time: {train_time:.2f}s, ' \
               f'Pruning Time: {pruning_time:.2f}s, ' \
               f'Test Time: {test_time:.2f}s || ' \
               f'lr: {model.optimizers[0].param_groups[0]["lr"]:.5f}'
        print(info)

    # Finishing training

    start_time = time.time()
    train_acc_goodness, train_acc_softmax, train_acc_ensemble = test_model(model, train_loader, device)
    end_time = time.time() - start_time

    print(f'Final: Train Acc Goodness: {train_acc_goodness:.2f}% \t Train Acc Softmax: {train_acc_softmax:.2f}% \t Train Acc Ensemble: {train_acc_ensemble:.2f}% \t || Test training-set Time: {end_time:.2f}s')
    if task_dir is not None and args.save_step != 1:
        with open(os.path.join(task_dir, 'accuracy.csv'), 'a') as f:
            f.write(f'{epochs},{train_acc_goodness:.2f},{train_acc_softmax:.2f},{train_acc_ensemble:.2f},{valid_acc:.2f},{test_acc_goodness:.2f},{test_acc_softmax:.2f},{test_acc_ensemble:.2f}\n')

    train_layer_acc_list = model.test_local_acc(train_loader)
    test_layer_acc_list = model.test_local_acc(test_loader)
    if task_dir is not None:
        with open(os.path.join(task_dir, 'layer_acc.csv'), 'w') as f:
            for i in range(len(train_layer_acc_list)):
                f.write(f'{train_layer_acc_list[i]},{test_layer_acc_list[i]}\n')

    model = model.to('cpu')
    if task_dir is not None:
        model_path = os.path.join(task_dir, 'model.pth')
        torch.save(model.state_dict(), model_path)
    # remove checkpoint
    os.remove(checkpoint_path)
