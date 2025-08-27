import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torchvision.transforms import Compose, ToTensor, Normalize
import torchvision.transforms as T
from torch import Tensor
import torch.nn.functional as F

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from PIL.Image import Image

from typing import Tuple
from tqdm import tqdm
from scipy.stats import entropy
from datetime import datetime
import os
import subprocess

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


from vis_v6 import visualize_batch_3views

# === Transforms

from lightly.transforms.gaussian_blur import GaussianBlur
from lightly.transforms.multi_view_transform import MultiViewTransform
from lightly.transforms.rotation import random_rotation_transform
from lightly.transforms.solarize import RandomSolarization
from lightly.transforms.utils import IMAGENET_NORMALIZE


class BYOLView1Transform:
    def __init__(
            self,
            input_size: int = 224,
            cj_prob: float = 0.8,
            cj_strength: float = 1.0,
            cj_bright: float = 0.4,
            cj_contrast: float = 0.4,
            cj_sat: float = 0.2,
            cj_hue: float = 0.1,
            min_scale: float = 0.08,
            random_gray_scale: float = 0.2,
            gaussian_blur: float = 1.0,
            solarization_prob: float = 0.0,
            kernel_size: Optional[float] = None,
            sigmas: Tuple[float, float] = (0.1, 2),
            vf_prob: float = 0.0,
            hf_prob: float = 0.5,
            rr_prob: float = 0.0,
            rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
            normalize: Union[None, Dict[str, List[float]]] = IMAGENET_NORMALIZE,
    ):
        color_jitter = T.ColorJitter(
            brightness=cj_strength * cj_bright,
            contrast=cj_strength * cj_contrast,
            saturation=cj_strength * cj_sat,
            hue=cj_strength * cj_hue,
        )

        transform = [
            T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0)),
            random_rotation_transform(rr_prob=rr_prob, rr_degrees=rr_degrees),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=vf_prob),
            T.RandomApply([color_jitter], p=cj_prob),
            T.RandomGrayscale(p=random_gray_scale),
            GaussianBlur(kernel_size=kernel_size, sigmas=sigmas, prob=gaussian_blur),
            RandomSolarization(prob=solarization_prob),
            T.ToTensor(),
        ]
        if normalize:
            transform += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]
        self.transform = T.Compose(transform)

    def __call__(self, image: Union[Tensor, Image]) -> Tensor:
        """
        Applies the transforms to the input image.

        Args:
            image:
                The input image to apply the transforms to.

        Returns:
            The transformed image.

        """
        transformed: Tensor = self.transform(image)
        return transformed


class BYOLView2Transform:
    def __init__(
            self,
            input_size: int = 224,
            cj_prob: float = 0.8,
            cj_strength: float = 1.0,
            cj_bright: float = 0.4,
            cj_contrast: float = 0.4,
            cj_sat: float = 0.2,
            cj_hue: float = 0.1,
            min_scale: float = 0.08,
            random_gray_scale: float = 0.2,
            gaussian_blur: float = 0.1,
            solarization_prob: float = 0.2,
            kernel_size: Optional[float] = None,
            sigmas: Tuple[float, float] = (0.1, 2),
            vf_prob: float = 0.0,
            hf_prob: float = 0.5,
            rr_prob: float = 0.0,
            rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
            normalize: Union[None, Dict[str, List[float]]] = IMAGENET_NORMALIZE,
    ):
        color_jitter = T.ColorJitter(
            brightness=cj_strength * cj_bright,
            contrast=cj_strength * cj_contrast,
            saturation=cj_strength * cj_sat,
            hue=cj_strength * cj_hue,
        )

        transform = [
            T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0)),
            random_rotation_transform(rr_prob=rr_prob, rr_degrees=rr_degrees),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=vf_prob),
            T.RandomApply([color_jitter], p=cj_prob),
            T.RandomGrayscale(p=random_gray_scale),
            GaussianBlur(kernel_size=kernel_size, sigmas=sigmas, prob=gaussian_blur),
            RandomSolarization(prob=solarization_prob),
            T.ToTensor(),
        ]
        if normalize:
            transform += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]
        self.transform = T.Compose(transform)

    def __call__(self, image: Union[Tensor, Image]) -> Tensor:
        """
        Applies the transforms to the input image.

        Args:
            image:
                The input image to apply the transforms to.

        Returns:
            The transformed image.

        """
        transformed: Tensor = self.transform(image)
        return transformed


class BYOLView3Transform:
    def __init__(
            self,
            input_size: int = 224,
            cj_prob: float = 0.8,
            cj_strength: float = 1.0,
            cj_bright: float = 0.4,
            cj_contrast: float = 0.4,
            cj_sat: float = 0.2,
            cj_hue: float = 0.1,
            min_scale: float = 0.08,
            random_gray_scale: float = 0.2,
            gaussian_blur: float = 0.1,
            solarization_prob: float = 0.2,
            kernel_size: Optional[float] = None,
            sigmas: Tuple[float, float] = (0.1, 2),
            vf_prob: float = 0.0,
            hf_prob: float = 0.5,
            rr_prob: float = 0.0,
            rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
            normalize: Union[None, Dict[str, List[float]]] = IMAGENET_NORMALIZE,
    ):
        color_jitter = T.ColorJitter(
            brightness=cj_strength * cj_bright,
            contrast=cj_strength * cj_contrast,
            saturation=cj_strength * cj_sat,
            hue=cj_strength * cj_hue,
        )

        transform = [
            T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0)),
            random_rotation_transform(rr_prob=rr_prob, rr_degrees=rr_degrees),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=vf_prob),
            T.RandomApply([color_jitter], p=cj_prob),
            T.RandomGrayscale(p=random_gray_scale),
            GaussianBlur(kernel_size=kernel_size, sigmas=sigmas, prob=gaussian_blur),
            RandomSolarization(prob=solarization_prob),
            T.ToTensor(),
        ]
        if normalize:
            transform += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]
        self.transform = T.Compose(transform)

    def __call__(self, image: Union[Tensor, Image]) -> Tensor:
        """
        Applies the transforms to the input image.

        Args:
            image:
                The input image to apply the transforms to.

        Returns:
            The transformed image.

        """
        transformed: Tensor = self.transform(image)
        return transformed


class BYOLTransform(MultiViewTransform):
    """Implements the transformations for BYOL[0].

    Input to this transform:
        PIL Image or Tensor.

    Output of this transform:
        List of Tensor of length 3.

    Applies the following augmentations by default:
        - Random resized crop
        - Random horizontal flip
        - Color jitter
        - Random gray scale
        - Gaussian blur
        - Solarization
        - ImageNet normalization

    Note that SimCLR v1 and v2 use similar augmentations. In detail, BYOL has
    asymmetric gaussian blur and solarization. Furthermore, BYOL has weaker
    color jitter compared to SimCLR.

    - [0]: Bootstrap Your Own Latent, 2020, https://arxiv.org/pdf/2006.07733.pdf

    Input to this transform:
        PIL Image or Tensor.

    Output of this transform:
        List of [tensor, tensor, tensor].

    Attributes:
        view_1_transform: The transform for the first view.
        view_2_transform: The transform for the second view.
        view_3_transform: The transform for the third view.
    """

    def __init__(
            self,
            view_1_transform: Optional[BYOLView1Transform] = None,
            view_2_transform: Optional[BYOLView2Transform] = None,
            view_3_transform: Optional[BYOLView3Transform] = None,
    ):
        # We need to initialize the transforms here
        view_1_transform = view_1_transform or BYOLView1Transform()
        view_2_transform = view_2_transform or BYOLView2Transform()
        view_3_transform = view_3_transform or BYOLView3Transform()
        super().__init__(transforms=[view_1_transform, view_2_transform, view_3_transform])


# === Support functions

def print_log(a_str, log_full_path):
    print(a_str)

    # Append text to the log file
    try:
        with open(log_full_path, "a") as file:
            file.write(a_str)
            file.write('\n')
    except Exception as e:
        print(f"Failed to append to the log file: {e}")


# === Models

from lightly.models.modules import BarlowTwinsProjectionHead


# from barlowtriplets_v1_loss import BarlowTwinsLossThreeHead

class BarlowTwinsLossThreeHead(torch.nn.Module):
    def __init__(self, lambda_param: float = 5e-3):
        super(BarlowTwinsLossThreeHead, self).__init__()
        self.lambda_param = lambda_param

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor, z_c: torch.Tensor) -> torch.Tensor:
        # normalize representations along the batch dimension
        z_a_norm, z_b_norm, z_c_norm = _normalize(z_a, z_b, z_c)

        N = z_a.size(0)

        # cross-correlation matrices
        c_ab = z_a_norm.T @ z_b_norm / N
        c_ac = z_a_norm.T @ z_c_norm / N
        c_bc = z_b_norm.T @ z_c_norm / N

        # loss calculation for each cross-correlation matrix
        loss_ab = _calculate_loss(c_ab, self.lambda_param)
        loss_ac = _calculate_loss(c_ac, self.lambda_param)
        loss_bc = _calculate_loss(c_bc, self.lambda_param)

        # sum the losses
        total_loss = (loss_ab + loss_ac + loss_bc) / 3

        return total_loss


def _normalize(*args: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """Helper function to normalize tensors along the batch dimension."""
    combined = torch.stack(args, dim=0)  # Shape: k x N x D (k = number of tensors)
    normalized = F.batch_norm(
        combined.flatten(0, 1),
        running_mean=None,
        running_var=None,
        weight=None,
        bias=None,
        training=True,
    ).view_as(combined)
    return tuple(normalized)


def _off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def _calculate_loss(c_matrix, lambda_param):
    """Calculate the loss for a given cross-correlation matrix."""
    invariance_loss = torch.diagonal(c_matrix).add_(-1).pow_(2).sum()
    redundancy_reduction_loss = _off_diagonal(c_matrix).pow_(2).sum()
    return invariance_loss + lambda_param * redundancy_reduction_loss


# Define the transformation encoder network
# As of 06/03/25 use the same architecture as the main encoder
class TransformEncoderRN(nn.Module):
    def __init__(self, input_dim, transform_dim):
        super(TransformEncoderRN, self).__init__()
        self.network = nn.Sequential(
            # nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.Linear(input_dim, transform_dim),
            nn.ReLU()
            # nn.Linear(512, 64),
            # nn.ReLU(),
            # #nn.MaxPool2d(2, 2),
            # #nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            # nn.Linear(64, 64),
            # nn.ReLU(),
            # #nn.MaxPool2d(2, 2),
            # #nn.Flatten(),
            # nn.Linear(64, transform_dim),
            # nn.ReLU()
        )

    def forward(self, x):
        return self.network(x)


# Define the encoder network (ResNet-like backbone for CIFAR10/100)
class EncoderRN(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(EncoderRN, self).__init__()
        self.network = nn.Sequential(
            # nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.Linear(input_dim, latent_dim),
            nn.ReLU()
            # nn.Linear(512, 64),
            # nn.ReLU(),
            # #nn.MaxPool2d(2, 2),
            # #nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            # nn.Linear(64, 64),
            # nn.ReLU(),
            # #nn.MaxPool2d(2, 2),
            # #nn.Flatten(),
            # nn.Linear(64, latent_dim),
            # nn.ReLU()
        )

    def forward(self, x):
        return self.network(x)


class Z_To_NCL_Branch(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Z_To_NCL_Branch, self).__init__()
        self.branch = nn.Sequential(
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x):
        return self.branch(x)


# Define the decoder network
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 64 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),
            nn.Sigmoid()  # Ensures output is in [0,1]
        )

    def forward(self, z):
        return self.network(z)


# Define the main model combining encoder, decoder and transformation embeddings
class AENCLModel(nn.Module):
    def __init__(self, backbone, latent_dim, z_to_ncl_dim, transform_dim):
        super(AENCLModel, self).__init__()
        self.backbone = backbone
        self.encoder = EncoderRN(512, latent_dim)
        self.transform_encoder = TransformEncoderRN(512, transform_dim)
        self.decoder = Decoder(latent_dim + transform_dim)
        self.z_to_ncl_branch = Z_To_NCL_Branch(latent_dim, z_to_ncl_dim)

    def forward(self, x):
        backboneout = self.backbone(x)
        backboneout = backboneout.view(backboneout.size(0), -1)  # Flatten the features
        z = self.encoder(backboneout)
        zb = self.z_to_ncl_branch(z)
        t = self.transform_encoder(backboneout)  # Get transformation embeddings
        zt = torch.cat([z, t], dim=1)  # Concatenate image and transformation embeddings
        x_rec = self.decoder(zt)

        return z, zb, t, backboneout, x_rec


class BarlowTripletsModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = BarlowTwinsProjectionHead(512, 2048, 2048)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return x, z


# Define the main model combining encoder, decoder and transformation embeddings
class AEBTModel(nn.Module):
    def __init__(self, backbone, latent_dim, transform_dim):
        super(AEBTModel, self).__init__()
        self.backbone = backbone
        self.encoder = EncoderRN(512, latent_dim)
        self.transform_encoder = TransformEncoderRN(512, transform_dim)
        self.decoder = Decoder(latent_dim+transform_dim)
        #self.decoder = Decoder(transform_dim)
        self.projection_head = BarlowTwinsProjectionHead(latent_dim, 2048, 2048)

    def forward(self, x):
        backboneout = self.backbone(x)
        backboneout = backboneout.view(backboneout.size(0), -1)  # Flatten the features
        z = self.encoder(backboneout)
        zb = self.projection_head(z)
        t = self.transform_encoder(backboneout)  # Get transformation embeddings
        zt = torch.cat([z, t], dim=1)  # Concatenate image and transformation embeddings
        x_rec = self.decoder(zt)
        #x_rec = self.decoder(t)

        return z, zb, t, backboneout, x_rec


def train_barlow_triplets(model, dataloader, batch_iter, criterion, optimizer, device, epochs, num_batches,
                          log_full_path):
    # Training loop remains the same
    model.train()
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        bi = 0
        print_log(f'Epoch {epoch + 1} ...', log_full_path)
        for batch in tqdm(dataloader, desc="Processing batches"):
            # print(f'Batch: {bi+1}')
            x0, x1, x2 = batch[0]
            x0 = x0.to(device)
            x1 = x1.to(device)
            x2 = x2.to(device)
            # print(f'type(batch[0]) {type(batch[0])}; len(batch[0]) {len(batch[0])}; x1.shape {x1.shape}')

            for _ in range(batch_iter):
                _, z0 = model(x0)
                _, z1 = model(x1)
                _, z2 = model(x2)
                loss = criterion(z0, z1, z2)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            bi += 1
            if bi >= num_batches:
                break
        losses.append(total_loss)
        print_log(f'Epoch: {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}', log_full_path)

    all_losses = {}
    all_losses['tot'] = losses
    return all_losses


# Define the loss functions
def reconstruction_loss(x, x_reconstructed):
    return nn.MSELoss()(x, x_reconstructed)


def ncl_mse_loss(z1, z2):  # redundant, but can extend later if necessary
    return nn.MSELoss()(z1, z2)


# Function from ChatGPT - 07/03/25
def negative_cosine_similarity(x1, x2):
    """Computes the negative cosine similarity loss between two tensors."""
    x1 = F.normalize(x1, p=2, dim=-1)  # Normalize each sample in x1
    x2 = F.normalize(x2, p=2, dim=-1)  # Normalize each sample in x2
    return - (x1 * x2).sum(dim=-1).mean()  # Negative mean cosine similarity


def cosine_similarity(x1, x2):
    """Computes the negative cosine similarity loss between two tensors."""
    x1 = F.normalize(x1, p=2, dim=-1)  # Normalize each sample in x1
    x2 = F.normalize(x2, p=2, dim=-1)  # Normalize each sample in x2
    return (x1 * x2).sum(dim=-1).mean()  # mean cosine similarity


# Function from ChatGPT - 15/03/25
def covariance_loss(z, t):
    z_mean = z - z.mean(dim=0)
    t_mean = t - t.mean(dim=0)
    cov = (z_mean.T @ t_mean) / (z.size(0) - 1)
    return torch.norm(cov, p='fro')  # Frobenius norm of the covariance matrix


# Function from ChatGPT - 17/03/25
def decorrelation_loss_1(a, b, lambda_var=1.0, lambda_cov=1.0):
    # Center the embeddings
    a = a - a.mean(dim=0, keepdim=True)
    b = b - b.mean(dim=0, keepdim=True)

    N = a.size(0)  # batch size

    # Variance preservation (encourage unit variance)
    var_a = a.var(dim=0, unbiased=False)
    var_b = b.var(dim=0, unbiased=False)
    var_loss = ((var_a - 1) ** 2).mean() + ((var_b - 1) ** 2).mean()

    # Cross-covariance decorrelation
    cov_ab = (a.T @ b) / (N - 1)
    cov_loss = (cov_ab ** 2).mean()

    # Total loss
    total_loss = lambda_var * var_loss + lambda_cov * cov_loss
    return total_loss


def train_aencl(model, train_loader, batch_iter, optimizer, device, epochs, num_batches, latent_dim, alpha, num_zb,
                log_full_path):
    rec_losses = []
    dec_losses = []
    ncl_losses = []
    tot_losses = []

    for epoch in range(epochs):
        model.train()
        bi = 0  # batch index
        # Losses within each episode
        ep_rec_loss = 0.0
        ep_dec_loss = 0.0
        ep_ncl_loss = 0.0
        ep_tot_loss = 0.0
        print_log(f'Epoch {epoch + 1} ...', log_full_path)
        for x, _ in tqdm(train_loader, desc="Processing batches"):

            # print(f'Batch: {bi+1}')

            x1, x2, x3 = x

            x1 = x1.to(device)
            x2 = x2.to(device)
            # x3 = x3.to(device) # ignore x3 as of 07/03/25
            # disp_images(x1, x2, 30) # for testing/debugging

            # x0 = x0.to(device)
            # x1 = x1.to(device)
            # x2 = x2.to(device)

            # Maybe uncomment later.
            # x1 = x1 + 0.05 * torch.randn_like(x)
            # x2 = x2 - 0.05 * torch.randn_like(x)

            for _ in range(batch_iter):

                z1, zb1, t1, resnetout, x1_rec = model(x1)
                z2, zb2, t2, resnetout, x2_rec = model(x2)

                # Reconstruction loss
                loss_reconstruction = (reconstruction_loss(x1, x1_rec) + reconstruction_loss(x2, x2_rec)) / 2

                # NCL loss
                if num_zb == 1:
                    # loss_ncl = negative_cosine_similarity(zb1,z2)
                    loss_ncl = ncl_mse_loss(zb1, z2)
                else:
                    # loss_ncl = negative_cosine_similarity(zb1,zb2)
                    loss_ncl = ncl_mse_loss(zb1, zb2)

                # z/t decorrelation loss
                # loss_decorrel = (covariance_loss(z1,t1) + covariance_loss(z2,t2))/2
                loss_decorrel = (decorrelation_loss_1(z1, t1) + decorrelation_loss_1(z2, t2)) / 2

                # Combined loss
                loss = alpha['ncl'] * loss_ncl + alpha['rec'] * loss_reconstruction + alpha['dec'] * loss_decorrel
                # Sum loss components within each episode
                ep_rec_loss += loss_reconstruction.item()
                ep_dec_loss += loss_decorrel.item()
                ep_ncl_loss += loss_ncl.item()
                ep_tot_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bi += 1
            if bi >= num_batches:
                break

        # Display and store intermediate results
        rec_losses.append(ep_rec_loss)
        dec_losses.append(ep_dec_loss)
        ncl_losses.append(ep_ncl_loss)
        tot_losses.append(ep_tot_loss)
        print_log(
            f"Epoch {epoch + 1}/{epochs}, Reconstruction Loss: {ep_rec_loss:.4f}, Decorrelation Loss: {ep_dec_loss:.4f}, NCL Loss: {ep_ncl_loss:.4f}",
            log_full_path
        )
        print_log(f"Epoch {epoch + 1}/{epochs}, Total Loss: {ep_tot_loss:.4f}", log_full_path)

    all_losses = {}
    all_losses['rec'] = rec_losses
    all_losses['dec'] = dec_losses
    all_losses['ncl'] = ncl_losses
    all_losses['tot'] = tot_losses

    return all_losses


def train_aebt(model, train_loader, batch_iter, bt_criterion, optimizer, device, epochs, num_batches, latent_dim, alpha,
               log_full_path):
    rec_losses = []
    dec_losses = []
    bt_losses = []
    tot_losses = []

    for epoch in range(epochs):
        model.train()
        bi = 0  # batch index
        # Losses within each episode
        ep_rec_loss = 0.0
        ep_dec_loss = 0.0
        ep_bt_loss = 0.0
        ep_tot_loss = 0.0
        print_log(f'Epoch {epoch + 1} ...', log_full_path)
        for x, _ in tqdm(train_loader, desc="Processing batches"):

            # print(f'Batch: {bi+1}')

            x1, x2, x3 = x
            # print(f'type(x) {type(x)}; len(x) {len(x)}; x1.shape {x1.shape}')

            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)
            # disp_images(x1, x2, 30) # for testing/debugging

            # Maybe uncomment later.
            # x1 = x1 + 0.05 * torch.randn_like(x)
            # x2 = x2 - 0.05 * torch.randn_like(x)

            for _ in range(batch_iter):

                z1, zb1, t1, resnetout, x1_rec = model(x1)
                z2, zb2, t2, resnetout, x2_rec = model(x2)
                z3, zb3, t3, resnetout, x3_rec = model(x3)

                # Reconstrucion loss
                loss_reconstruction = (reconstruction_loss(x1, x1_rec) + reconstruction_loss(x2,
                                                                                             x2_rec) + reconstruction_loss(
                    x3, x3_rec)) / 3

                # Barlow Triplet loss
                loss_bt = bt_criterion(zb1, zb2, zb3)

                # z/t decorrelation loss
                # loss_decorrel = (covariance_loss(z1,t1) + covariance_loss(z2,t2) + covariance_loss(z3,t3))/3
                if alpha['dec'] > 0:
                    # loss_decorrel = (cosine_similarity(z1,t1) + cosine_similarity(z2,t2) + cosine_similarity(z3,t3))/3
                    loss_decorrel = (decorrelation_loss_1(z1, t1) + decorrelation_loss_1(z2, t2) + decorrelation_loss_1(
                        z3, t3)) / 3
                else:
                    loss_decorrel = 0.0

                # Combined loss
                loss = alpha['ncl'] * loss_bt + alpha['rec'] * loss_reconstruction + alpha['dec'] * loss_decorrel

                # Sum loss components within each episode
                ep_rec_loss += loss_reconstruction.item()
                ep_bt_loss += loss_bt.item()
                ep_tot_loss += loss.item()
                if alpha['dec'] > 0:
                    ep_dec_loss += loss_decorrel.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bi += 1
            if bi >= num_batches:
                break

        # Display and store intermediate results
        rec_losses.append(ep_rec_loss)
        dec_losses.append(ep_dec_loss)
        bt_losses.append(ep_bt_loss)
        tot_losses.append(ep_tot_loss)
        print_log(
            f"Epoch {epoch + 1}/{epochs}, Reconstruction Loss: {ep_rec_loss:.4f}, Decorrelation Loss: {ep_dec_loss:.4f}, BT Loss: {ep_bt_loss:.4f}",
            log_full_path
        )
        print_log(f"Epoch {epoch + 1}/{epochs}, Total Loss: {ep_tot_loss:.4f}", log_full_path)

    all_losses = {}
    all_losses['rec'] = rec_losses
    all_losses['dec'] = dec_losses
    all_losses['bt'] = bt_losses
    all_losses['tot'] = tot_losses

    return all_losses


# Print performance metrics
def print_perform_metrics(sel_model, embedding_loc, k, top1_accuracy, topk_accuracy, precision, recall, f1, entrop,
                          log_full_path):
    print_log(f'=== {sel_model} / {embedding_loc}: performance metrics ===', log_full_path)
    print_log(
        f'T1-AC={top1_accuracy * 100:.1f}%; T{k}-AC={topk_accuracy * 100:.1f}%; Precision={precision * 100:.1f}%; Recall={recall * 100:.1f}%; F1={f1 * 100:.1f}%; ENT={entrop:.6f}',
        log_full_path)


def compute_entropy(embeddings, num_bins=50):
    """
    Compute the entropy of the embedding distribution.
    Args:
        embeddings (np.array): Embeddings array of shape (N, D), where N is the number of samples and D is the embedding dimension.
        num_bins (int): Number of bins for the histogram.
    Returns:
        float: Mean entropy across all dimensions.
    """
    entropies = []
    # Compute entropy for each embedding dimension
    for dim in range(embeddings.shape[1]):
        hist, _ = np.histogram(embeddings[:, dim], bins=num_bins, density=True)
        hist = hist + 1e-10  # Avoid log(0) by adding a small value
        entropies.append(entropy(hist))  # Compute entropy of histogram
    return np.mean(entropies)  # Return average entropy across all dimensions


# Compute performance metrics
def comp_metrics(features, labels, k):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(features, labels)
    knn_pred = knn.predict(features)
    # Compute Top-1 accuracy
    top1_accuracy = accuracy_score(labels, knn_pred)
    # Compute Top-5 accuracy
    # arg1 = labels[:, np.newaxis]
    # arg2 = np.argsort(knn.predict_proba(features), axis=1)[:, -5:]
    # print(f'np.any(labels[:, np.newaxis] --> {arg1}')
    # print(f'np.argsort(knn.predict_proba(features), axis=1)[:, -5:] --> {arg2}')
    topk_accuracy = np.mean(
        np.any(labels[:, np.newaxis] == np.argsort(knn.predict_proba(features), axis=1)[:, -k:], axis=1))
    # Compute Precision, Recall, and F1 score
    precision = precision_score(labels, knn_pred, average='macro')
    recall = recall_score(labels, knn_pred, average='macro')
    f1 = f1_score(labels, knn_pred, average='macro')
    # Compute entropy
    entrop = compute_entropy(features)

    return top1_accuracy, topk_accuracy, precision, recall, f1, entrop


# Evaluate
def evaluate(model, sel_model, dataloader, device, k, losses, log_full_path):
    model.eval()
    labels = []
    all_res = {}

    if sel_model == 'AEBT':
        features_z = []
        features_zt = []
        features_zb = []
        features_t = []
        features_backbone = []
        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(device)
                z, zb, t, backboneout, x_rec = model(images)
                zt = torch.cat([z, t], dim=1)
                features_z.extend(z.cpu().numpy())
                features_zt.extend(zt.cpu().numpy())
                features_zb.extend(zb.cpu().numpy())
                features_t.extend(t.cpu().numpy())
                features_backbone.extend(backboneout.cpu().numpy())
                labels.extend(targets.numpy())

        features_z = np.array(features_z)
        features_zt = np.array(features_zt)
        features_zb = np.array(features_zb)
        features_t = np.array(features_t)
        features_backbone = np.array(features_backbone)
        labels = np.array(labels)

        # 存特征和标签到结果
        all_res['features_z'] = features_z
        all_res['features_zt'] = features_zt
        all_res['features_zb'] = features_zb
        all_res['features_t'] = features_t
        all_res['features_backbone'] = features_backbone
        all_res['labels'] = labels

        # KNN 评估（保留原有）
        top1_accuracy, topk_accuracy, precision, recall, f1, ent = comp_metrics(features_backbone, labels, k)
        all_res['backbone'] = [top1_accuracy, topk_accuracy, precision, recall, f1, ent]
        print_perform_metrics(sel_model, 'backbone_out', k, top1_accuracy, topk_accuracy, precision, recall, f1, ent, log_full_path)
        top1_accuracy, topk_accuracy, precision, recall, f1, ent = comp_metrics(features_z, labels, k)
        all_res['z'] = [top1_accuracy, topk_accuracy, precision, recall, f1, ent]
        print_perform_metrics(sel_model, 'z', k, top1_accuracy, topk_accuracy, precision, recall, f1, ent, log_full_path)
        top1_accuracy, topk_accuracy, precision, recall, f1, ent = comp_metrics(features_zt, labels, k)
        all_res['zt'] = [top1_accuracy, topk_accuracy, precision, recall, f1, ent]
        print_perform_metrics(sel_model, 'zt', k, top1_accuracy, topk_accuracy, precision, recall, f1, ent, log_full_path)
        top1_accuracy, topk_accuracy, precision, recall, f1, ent = comp_metrics(features_zb, labels, k)
        all_res['zb'] = [top1_accuracy, topk_accuracy, precision, recall, f1, ent]
        print_perform_metrics(sel_model, 'zb', k, top1_accuracy, topk_accuracy, precision, recall, f1, ent, log_full_path)
        top1_accuracy, topk_accuracy, precision, recall, f1, ent = comp_metrics(features_t, labels, k)
        all_res['t'] = [top1_accuracy, topk_accuracy, precision, recall, f1, ent]
        print_perform_metrics(sel_model, 't', k, top1_accuracy, topk_accuracy, precision, recall, f1, ent, log_full_path)
        all_res['losses'] = losses
    else:
        print_log('The model type does not exist. Evaluation is not possible.', log_full_path)

    return all_res



def plot_feature_distribution(features, labels, method='tsne', title='Feature Distribution', save_path=None):
    # 降维
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2)
    reduced = reducer.fit_transform(features)
    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.7, s=10)
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar(scatter, label="Class")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_knn_confusion_matrix(features, labels, k=5, title='Confusion Matrix', save_path=None):
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(features, labels)
    preds = knn.predict(features)
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', xticks_rotation='vertical')
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()



def main():
    # === Main

    # === Parameters common to all conditions

    # -- Exper. parameters shared by all models (BT, AENCL, AEBT)
    # CL = continual learning
    num_epochs = 200  # CL -> set to 1; non-CL -> set to >= 1
    batch_iter = 1  # CL -> set to >= 1; non-CL -> set to 1
    batch_size = 256
    num_batches = float('inf')  # use a small number for quick tests
    sel_model = 'AEBT'  # options 'AENCL', 'BT' (Barlow Triplets), 'AEBT' (hybrid)
    k = 2  # for the k>1 knn case
    save_path = 'D:\python project\AE_NCL\Results'
    exper_string = 'v18'
    log_filename = 'log' + exper_string
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kill_job_at_end = True
    # --- Exper. parameters specific to AENCL
    z_to_ncl_dim = 10
    num_zb = 2  # number of z-to-ncl branches to consider in the ncl loss
    if num_zb == 1:
        z_to_ncl_dim = latent_dim  # This needs to be enforced because of the cosine similarity

    # Dataset & Transformations
    transform = BYOLTransform(
        view_1_transform=BYOLView1Transform(input_size=32, gaussian_blur=0.0),
        view_2_transform=BYOLView2Transform(input_size=32, gaussian_blur=0.0),
        view_3_transform=BYOLView3Transform(input_size=32, gaussian_blur=0.0),
    )
    dataset = CIFAR10("datasets/cifar10", download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True, num_workers=4)
    test_transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_dataset = CIFAR10("datasets/cifar10", train=False, download=True, transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
    raw_dataset = torchvision.datasets.CIFAR10(
        root="datasets/cifar10", train=False, download=True,
        transform=Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    #raw_dataset = CIFAR10("datasets/cifar10", train=False, download=True, transform=ToTensor())

    # Prepare the log_file

    # Adapted from ChatGPT, 16/03/25
    # Check if the log file exists, and create it if it doesn't
    log_full_path = save_path + log_filename + '.txt'
    if not os.path.exists(log_full_path):
        print(f"Log file '{log_full_path}' does not exist. Creating it...")
        with open(log_full_path, "w") as log_file:
            log_file.write("Log file created.\n")  # Optional: Write an initial message

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    init_message = '====== Results for model ' + sel_model + ', experiment ' + exper_string + ', and date-time ' + f'_{timestamp}' + ' ==='
    print_log(init_message, log_full_path)

    # === Condition 1 - Baseline (with all losses)

    print_log('=== Condition 1', log_full_path)

    # --- Experimental parameters specific to condition 1
    alpha = {}
    #alpha['ncl'] = 0.5  # NCL loss weight
    #alpha['ncl'] = 0.005  # NCL loss weight
    #alpha['rec'] = 0.5  # reconstruction loss weight
    #alpha['dec'] = 0.5  # z/t decorrelation loss weight
    alpha['ncl'] = 0.001  # NCL loss weight
    alpha['rec'] = 1  # reconstruction loss weight
    alpha['dec'] = 0.1  # z/t decorrelation loss weight

    # --- Exper. parameters specific to AENCL and AEBT
    latent_dim = 256
    transform_dim = 256
    cond_name = 'C1_all_loss'

    # Model and training setup
    resnet = torchvision.models.resnet18()
    backbone = nn.Sequential(*list(resnet.children())[:-1])

    # Model-specific parts
    if sel_model == 'BT':
        model = BarlowTripletsModel(backbone)
        model.to(device)
        criterion = BarlowTwinsLossThreeHead()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.06)
        losses = train_barlow_triplets(model, dataloader, batch_iter, criterion, optimizer, device, num_epochs, num_batches,
                                       log_full_path)
    elif sel_model == 'AENCL':
        model = AENCLModel(backbone, latent_dim, z_to_ncl_dim, transform_dim)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        losses = train_aencl(model, dataloader, batch_iter, optimizer, device, num_epochs, num_batches, latent_dim, alpha,
                             num_zb, log_full_path)
    elif sel_model == 'AEBT':
        model = AEBTModel(backbone, latent_dim, transform_dim)
        model.to(device)
        bt_criterion = BarlowTwinsLossThreeHead()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.06)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

        losses = train_aebt(model, dataloader, batch_iter, bt_criterion, optimizer, device, num_epochs, num_batches,
                            latent_dim, alpha, log_full_path)
    else:
        print_log(f'The model {sel_model} is not defined.', log_full_path)
    # Remaining setup and training code remains the same

    # Evaluate the model
    all_res = evaluate(model, sel_model, test_dataloader, device, k, losses, log_full_path)

    #特征分布可视化（PCA / t-SNE）

    features_z = np.array(all_res['features_z'])
    features_t = np.array(all_res['features_t'])
    features_zb = np.array(all_res['features_zb'])
    features_zt = np.array(all_res['features_zt'])
    features_backbone = np.array(all_res['features_backbone'])
    labels = np.array(all_res['labels'])

    # 特征分布可视化
    plot_feature_distribution(features_z, labels, method='tsne', title='z Distribution (t-SNE)')
    plot_feature_distribution(features_t, labels, method='tsne', title='t Distribution (t-SNE)')
    plot_feature_distribution(features_zb, labels, method='tsne', title='zb Distribution (t-SNE)')
    plot_feature_distribution(features_zt, labels, method='tsne', title='zt Distribution (t-SNE)')

    # KNN 混淆矩阵
    plot_knn_confusion_matrix(features_z, labels, k=5, title='KNN Confusion Matrix (z)')
    plot_knn_confusion_matrix(features_zb, labels, k=5, title='KNN Confusion Matrix (zb)')
    plot_knn_confusion_matrix(features_zt, labels, k=5, title='KNN Confusion Matrix (zt)')
    plot_knn_confusion_matrix(features_t, labels, k=5, title='KNN Confusion Matrix (t)')





    # Save results
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    filename = 'RES_' + cond_name + '_' + sel_model + '_' + exper_string + f'_{timestamp}.pt'
    file_path = save_path + filename
    torch.save(all_res, file_path)

    #show_recon_examples(model, test_dataloader, device, n=5)
    #show_input_aug_recon_examples(model, test_dataloader, raw_dataset, device, n=5)
    #visualize_batch(model, dataloader, device='cuda', n=5)

    visualize_batch_3views(model, dataloader, device='cuda', n=30)



    #Finish
    print_log('Finsihed processing. Experiment complete.', log_full_path)


    # === Wrapping up

    # Kill the current job?

    if kill_job_at_end == True:

        # Get the SLURM job ID from the environment
        job_id = os.environ.get('SLURM_JOB_ID')

        if job_id:
            # Use scancel to terminate the job
            subprocess.run(['scancel', job_id])
            print_log('Killed job.', log_full_path)
        else:
            print_log("SLURM_JOB_ID not found. Not running in a SLURM job?", log_full_path)


if __name__ == '__main__':
    main()



