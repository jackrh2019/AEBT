# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import copy

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torchvision.transforms import Compose, ToTensor, Normalize
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.byol_transform import (
    BYOLTransform,
    BYOLView1Transform,
    BYOLView2Transform,
)
from lightly.utils.scheduler import cosine_schedule

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
class BYOL(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = BYOLProjectionHead(512, 1024, 256)
        self.prediction_head = BYOLPredictionHead(256, 1024, 256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z



def train_BYOL(model, dataloader, criterion, optimizer, device, epochs):
    print("Starting Training")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        momentum_val = cosine_schedule(epoch, epochs, 0.996, 1)
        for batch in dataloader:
            x0, x1 = batch[0]
            update_momentum(model.backbone, model.backbone_momentum, m=momentum_val)
            update_momentum(
                model.projection_head, model.projection_head_momentum, m=momentum_val
            )
            x0 = x0.to(device)
            x1 = x1.to(device)
            p0 = model(x0)
            z0 = model.forward_momentum(x0)
            p1 = model(x1)
            z1 = model.forward_momentum(x1)
            loss = 0.5 * (criterion(p0, z1) + criterion(p1, z0))
            total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = total_loss / len(dataloader)
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

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
    resnet = torchvision.models.resnet18()
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    model = BYOL(backbone)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # We disable resizing and gaussian blur for cifar10.
    transform = BYOLTransform(
        view_1_transform=BYOLView1Transform(input_size=32, gaussian_blur=0.0),
        view_2_transform=BYOLView2Transform(input_size=32, gaussian_blur=0.0),
    )
    dataset = torchvision.datasets.CIFAR10(
        "datasets/cifar10", download=True, transform=transform
    )
    # or create a dataset from a folder containing images or videos:
    # dataset = LightlyDataset("path/to/folder", transform=transform)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )

    criterion = NegativeCosineSimilarity()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

    # Train the model
    train_BYOL(model, dataloader, criterion, optimizer, device, epochs = 200)

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