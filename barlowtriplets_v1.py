import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torchvision.transforms import Compose, ToTensor, Normalize

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

from byol_transform_v2 import (
    BYOLTransform,
    BYOLView1Transform,
    BYOLView2Transform,
    BYOLView3Transform,
)
from barlowtriplets_v1_loss import BarlowTwinsLossThreeHead
from lightly.models.modules import BarlowTwinsProjectionHead

class BarlowTriplets(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = BarlowTwinsProjectionHead(512, 2048, 2048)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

def train_barlow_triplets(model, dataloader, criterion, optimizer, device, epochs):
    # Training loop remains the same
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x0, x1, x2 = batch[0]
            x0 = x0.to(device)
            x1 = x1.to(device)
            x2 = x2.to(device)
            z0 = model(x0)
            z1 = model(x1)
            z2 = model(x2)
            loss = criterion(z0, z1, z2)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch: {epoch}, Loss: {total_loss / len(dataloader):.4f}')

def evaluate(model, dataloader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            output = model.backbone(images).flatten(start_dim=1)
            features.extend(output.cpu().numpy())
            labels.extend(targets.numpy())

    features = np.array(features)
    labels = np.array(labels)


    # KNN Evaluation
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(features, labels)
    knn_pred = knn.predict(features)
    # Compute Top-1 accuracy
    top1_accuracy = accuracy_score(labels, knn_pred)
    # Compute Top-5 accuracy
    top5_accuracy = np.mean(np.any(labels[:, np.newaxis] == np.argsort(knn.predict_proba(features), axis=1)[:, -5:], axis=1))
    # Compute Precision, Recall, and F1 score
    precision = precision_score(labels, knn_pred, average='macro')
    recall = recall_score(labels, knn_pred, average='macro')
    f1 = f1_score(labels, knn_pred, average='macro')

    return top1_accuracy, top5_accuracy, precision, recall, f1

if __name__ == '__main__':
    # Model and training setup
    resnet = torchvision.models.resnet18()
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    model = BarlowTriplets(backbone)
    # Remaining setup and training code remains the same


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    transform = BYOLTransform(
        view_1_transform=BYOLView1Transform(input_size=32, gaussian_blur=0.0),
        view_2_transform=BYOLView2Transform(input_size=32, gaussian_blur=0.0),
        view_3_transform=BYOLView3Transform(input_size=32, gaussian_blur=0.0),
    )

    dataset = CIFAR10("datasets/cifar10", download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True, num_workers=8)

    criterion = BarlowTwinsLossThreeHead()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

    # Train the model
    train_barlow_triplets(model, dataloader, criterion, optimizer, device, epochs=200)

    # Evaluate the model
    test_transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_dataset = CIFAR10("datasets/cifar10", train=False, download=True, transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8)

    top1_accuracy, top5_accuracy, precision, recall, f1 = evaluate(model, test_dataloader, device)

    print(f'Top-1 Accuracy: {top1_accuracy:.4f}')
    print(f'Top-5 Accuracy: {top5_accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')
