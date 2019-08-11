import os
from os.path import join, exists
import argparse
from tqdm import tqdm
import numpy as np
from mpi4py import MPI

import torch
import torch.utils.data as data
import torch.optim as optim
import horovod.torch as hvd

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import save_image

from model import PixelSNAIL


def metric_average(val, name):
    tensor = val.clone()
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def train(model, optimizer, device, train_loader, epoch):
    model.train()

    train_loader.sampler.set_epoch(epoch)

    train_losses = []
    if hvd.rank() == 0:
        pbar = tqdm(total=len(train_loader.sampler))
    for x, _ in train_loader:
        optimizer.zero_grad()
        x = x.to(device)
        loss = -model.log_prob(x) / np.prod(x.shape)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item() / np.log(2.))
        avg_loss = np.mean(train_losses[-10:])

        for g in optimizer.param_groups:
            g['lr'] *= args.lr_decay

        if hvd.rank() == 0:
            pbar.update(x.shape[0])
            pbar.set_description('Epoch {}, Train Loss {:.4f} bits/dim'.format(epoch, avg_loss))
    if hvd.rank() == 0:
        pbar.close()


def test(model, device, test_loader, epoch):
    model.eval()

    total_loss = 0
    for x, _ in test_loader:
        with torch.no_grad():
            x = x.to(device)
            loss = -model.log_prob(x) / np.prod(x.shape)
            total_loss += loss * x.shape[0]
    test_loss = total_loss / len(test_loader.sampler) / np.log(2.)
    test_loss = metric_average(test_loss, 'avg_loss')

    if hvd.rank() == 0:
        print('Epoch {}, Test Loss {:.4f} bits/dim'.format(epoch, test_loss))
    return avg_loss


def sample(model, device, epoch):
    model.eval()
    samples = model.sample(8, device)

    folder = './samples'
    if not exists(folder):
        os.makedirs(folder)
    filename = join(folder, 'sample_epoch{}.png'.format(epoch))
    save_image(samples, filename, nrow=8)


def main():
    hvd.init()

    seed = args.seed + hvd.rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    torch.cuda.set_device(hvd.local_rank())

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    kwargs = {'num_workers': 4, 'pin_memory': True, 'batch_size': args.batch_size}
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_sampler = data.distributed.DistributedSampler(trainset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = data.DataLoader(trainset, sampler=train_sampler, **kwargs)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_sampler = data.distributed.DistributedSampler(testset, num_replicas=hvd.size(), rank=hvd.rank())
    test_loader = data.DataLoader(testset, sampler=test_sampler, **kwargs)

    device = torch.device('cuda:{}'.format(hvd.local_rank()))
    obs_dim = trainset[0][0].shape

    model = PixelSNAIL(obs_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters())

    if hvd.rank() == 0:
        total_parameters = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
        print('Total Parameters {}'.format(total_parameters))

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    folder = './models'
    if not exists(folder):
        os.makedirs(folder)
    model_fname = join(folder, 'pixel_snail.pt')

    for epoch in range(args.epochs):
        MPI.COMM_WORLD.Barrier()
        train(model, optimizer, device, train_loader, epoch)
        test(model, device, test_loader, epoch)

        if hvd.rank() == 0:
            sample(model, device, epoch)

        torch.save(dict(model=model, optimizer=optimizer.state_dict()), model_fname)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr_decay', type=float, default=0.99999)

    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    main()
