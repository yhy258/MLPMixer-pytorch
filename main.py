import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from module import MLPMixer

from tqdm.notebook import tqdm
import numpy as np

dataset = datasets.CIFAR10(
    root = "./.data",
    train = True,
    transform =transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]),
    download = True
)

dataset, valid_dataset = torch.utils.data.random_split(dataset, [45000, 5000])

dataloader = DataLoader(dataset, batch_size =32, shuffle = True)
val_dataloader = DataLoader(valid_dataset, batch_size =32, shuffle = True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLPMixer(in_channels = 3, dim = 256, token_mix = 128, channel_mix = 1024, img_size = 32, patch_size = 4, depth = 8, num_classes = 10)
model = model.to(DEVICE)

optim = torch.optim.Adam(params=model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(50):
    losses = []
    model.train()
    for img, label in tqdm(dataloader):
        img = img.to(DEVICE)
        label = label.to(DEVICE)

        pred = model(img)

        loss = criterion(pred, label)
        losses.append(loss)

        optim.zero_grad()
        loss.backward()
        optim.step()

    model.eval()
    scores = 0
    all_data = 0
    with torch.no_grad():
        for val_image, val_label in tqdm(val_dataloader):
            val_image = val_image.to(DEVICE)
            val_label = val_label.to(DEVICE)

            val_pred = model(val_image)
            arg = F.softmax(val_pred, dim=1).argmax(dim=1)
            score = (arg == val_label).sum().item()
            scores += score
            all_data += len(val_image)

    print("Loss : {} Score : {}".format(np.mean(losses), scores / all_data))

