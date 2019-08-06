import os
from os.path import join, exists
import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.utils.data as data
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import save_image

from model import PixelSNAIL


def train(model, optimizer, device, train_loader, epoch):
    model.train()

    train_losses = []
    pbar = tqdm(total=len(train_loader.sampler))
    for x, _ in train_loader:
        optimizer.zero_grad()
        x = x.to(device)
        loss = -model.log_prob(x)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item() / np.log(2.))
        avg_loss = np.mean(train_losses[-10:])
        pbar.update(x.shape[0])
        pbar.set_description('Epoch {}, Train Loss {:.4f} bits/dim'.format(epoch, avg_loss))
    pbar.close()


def test(model, device, test_loader, epoch):
    model.eval()

    total_loss = 0
    for x, _ in test_loader:
        with torch.no_grad():
            x = x.to(device)
            loss = -model.log_prob(x)
            total_loss += loss * x.shape[0]
    avg_loss = total_loss / len(test_loader.sampler) / np.log(2.)
    print('Epoch {}, Test Loss {:.4f} bits/dim'.format(epoch, avg_loss))
    return avg_loss


def sample(model, device, epoch):
    model.eval()
    samples = model.sample(64, device)

    folder = './samples'
    if not exists(folder):
        os.makedirs(folder)
    filename = join(folder, 'sample_epoch{}.png'.format(epoch))
    save_image(samples, filename, nrow=8)


def main():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    kwargs = {'num_workers': 4, 'pin_memory': True, 'batch_size': args.batch_size}
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = data.DataLoader(trainset, shuffle=True, **kwargs)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = data.DataLoader(testset, shuffle=False, **kwargs)

    device = torch.device('cuda:{}'.format(args.gpu))
    obs_dim = trainset[0][0].shape

    model = PixelSNAIL(obs_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for e in range(args.epochs):
        train(model, optimizer, device, train_loader, epoch)
        test(model, device, test_loader, epoch)
        sample(model, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    main()
