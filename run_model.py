from train.model import NeuralNetwork, device
from train.dataset import test_data

import torch
import matplotlib.pyplot as plt
import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', help='model file name')
    parser.add_argument('-i', '--index', help='the index of the image')
    return parser.parse_args()

args = parse_args()

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load(f"models/{args.file}.pth"))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[int(args.index)][0], test_data[int(args.index)][1]

with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]


    x_np = x.cpu().numpy()
    x_np = x_np.transpose((1, 2, 0))

    plt.imshow(x_np)
    plt.title(f'Predicted: "{predicted}", Actual: "{actual}"')
    plt.axis('off')
    plt.show()

    print(f'Predicted: "{predicted}", Actual: "{actual}"')