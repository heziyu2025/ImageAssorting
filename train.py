import torch
from torch import nn
import argparse

from model import device, model
from dataloaders import train_dataloader, test_dataloader
from test import test

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch')
    parser.add_argument('-t', '--target', help='target correct')
    parser.add_argument('-f', '--file', help='model file name')
    return parser.parse_args()

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

args = parse_args()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

for t in range(int(args.epoch)):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)

    correct, test_loss = test(test_dataloader, model, loss_fn)
    if args.target != None and correct >= float(args.target):
        break
print("Done!")

torch.save(model.state_dict(), f"models/{args.file}.pth")
print(f"Saved PyTorch Model State to models/{args.file}.pth")