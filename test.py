import os
import sys
import argparse

import torch
import torch.nn.functional as F

from data.cifar10_dataloader import get_cifar10_dataloader
from data.cifar100_dataloader import get_cifar100_dataloader
from data.mnist_dataloader import get_mnist_dataloader
from data.fmnist_dataloader import get_fmnist_dataloader

from model.resnet import Resnet


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
    parser = argparse.ArgumentParser(description='HCL-FF Testing')
    parser.add_argument('--task', type=str, default='cifar100')
    parser.add_argument('--arch', type=str, default='resnet')
    parser.add_argument('--task_dir', type=str, default=None)
    parser.add_argument('--save_result', action='store_true', default=False)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2222)
    parser.add_argument('--data_dir', type=str, default='../deeperforward/datasets')
    parser.add_argument('--nopruning', action='store_true', default=False)
    args = parser.parse_args()

    task = args.task
    arch = args.arch
    task_dir = os.path.join('./results', args.task_dir)
    save_result = args.save_result
    seed = args.seed
    root = args.data_dir

    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    # error
    if task_dir is None:
        print('Please specify task_dir by --task_dir')
        sys.exit(1)

    model_dir = os.path.join(task_dir, 'model.pth')
    if not os.path.exists(model_dir):
        print('No model.pth in {}'.format(task_dir))
        sys.exit(1)

    in_channels = 3
    num_class = 10
    train_loader, valid_loader, test_loader, hierarchy = None, None, None, None
    if task == 'cifar10':
        planes = [100, 200, 400, 800]
        train_loader, valid_loader, test_loader, hierarchy = get_cifar10_dataloader(root=root, train_batch_size=128,
                                                                                    test_batch_size=128, seed=seed)
    elif task == 'cifar100':
        num_class = 100
        planes = [100, 200, 400, 800]
        train_loader, valid_loader, test_loader, hierarchy = get_cifar100_dataloader(root=root, train_batch_size=128,
                                                                                     test_batch_size=128, seed=seed)
    elif task == 'mnist':
        in_channels = 1
        planes = [40, 80, 160, 320] 
        train_loader, valid_loader, test_loader = get_mnist_dataloader(root=root, train_batch_size=128,
                                                                       test_batch_size=128, seed=seed)
    elif task == 'fmnist':
        in_channels = 1
        planes = [40, 80, 160, 320] 
        train_loader, valid_loader, test_loader = get_fmnist_dataloader(root=root, train_batch_size=128,
                                                                        test_batch_size=128, seed=seed)
    else:
        raise NotImplementedError

    # model
    model = None
    if args.arch == 'resnet':
        model = Resnet(in_channels=in_channels, num_class=num_class, planes=planes, device=device)
    else:
        raise NotImplementedError

    # load model
    model.load_state_dict(torch.load(model_dir), strict=False)
    model = model.to(device)

    if args.nopruning:
        model.start_layer = torch.tensor(1)
        model.end_layer = torch.tensor(len(model.layers))

    model.eval()
    train_acc, train_acc_softmax, train_acc_ensemble = test_model(model, train_loader, device)
    test_acc, test_acc_softmax, test_acc_ensemble = test_model(model, test_loader, device)
    print('Train Acc (Goodness): {:.2f}%, Train Acc (Softmax): {:.2f}%, Train Acc (Ensemble): {:.2f}%, Test Acc (Goodness): {:.2f}%, Test Acc (Softmax): {:.2f}%, Test Acc (Ensemble): {:.2f}%'.format(train_acc, train_acc_softmax, train_acc_ensemble, test_acc, test_acc_softmax, test_acc_ensemble))
    print('Start layer: {}, End layer: {}'.format(model.start_layer, model.end_layer))

    if save_result:
        train_layer_acc_list = model.test_local_acc(train_loader)
        test_layer_acc_list = model.test_local_acc(test_loader)

    if save_result:
        with open(os.path.join(task_dir, 'layer_acc.csv'), 'w') as f:
            for i in range(len(train_layer_acc_list)):
                f.write(f'{train_layer_acc_list[i]},{test_layer_acc_list[i]}\n')
        print('Save layer_acc.csv to {}'.format(task_dir))
