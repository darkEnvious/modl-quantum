import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import r2c, c2r

import pennylane as qml
from pennylane import numpy as np
import time

# Quantum Convolutional Layer ======================
class QuanvLayer(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        seed = 1
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        super(QuanvLayer, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_channels = 1  # Input channels (real and imaginary)
        self.n_qubits = kernel_size * kernel_size * self.n_channels
        self.n_layers = 1  # Number of layers in variational circuit

        # Define the quantum device
        self.dev = qml.device('default.qubit', wires=self.n_qubits) # Trying Qulacs simulator | Never mind this garbage just wasted an entire 2 days of my life AND caused so much agony

        # Define the weight shapes
        weight_shapes = {"weights": (self.n_layers, self.n_qubits)}

        # Define the quantum circuit
        @qml.qnode(self.dev, interface='torch', cache = False, diff_method = 'backprop') #diff_method='parameter-shift's #Set Cache to False if doesn't work
        def circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(self.n_qubits))
            qml.templates.BasicEntanglerLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

        # Create the QNode and QuantumLayer
        self.q_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x = torch.sqrt(x[:, 0, :, :]**2 + x[:, 1, :, :]**2).unsqueeze(1) #Line to reduce number of channels to 1
        # Extract patches
        patches = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        # patches shape: (batch_size, channels, n_patches_h, n_patches_w, kernel_size, kernel_size)

        n_patches_h = patches.shape[2]
        n_patches_w = patches.shape[3]

        # Reshape patches to (n_patches_total, n_qubits)
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        patches = patches.reshape(-1, self.n_channels * self.kernel_size * self.kernel_size)

        # Normalize patches
        patches_min = patches.min(dim=1, keepdim=True)[0]
        patches_max = patches.max(dim=1, keepdim=True)[0]
        patches = (patches - patches_min) / (patches_max - patches_min + 1e-8) * np.pi

        # Apply the quantum layer to all patches
        outputs = self.q_layer(patches)
        # outputs shape: (n_patches_total, n_qubits)

        # Reshape back to (batch_size, n_qubits, n_patches_h, n_patches_w)
        outputs = outputs.view(batch_size, n_patches_h, n_patches_w, -1)
        outputs = outputs.permute(0, 3, 1, 2)  # shape: (batch_size, n_qubits, n_patches_h, n_patches_w)

        return outputs

# CNN Denoiser ======================
def conv_block(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class cnn_denoiser(nn.Module):
    def __init__(self, n_layers):
        super().__init__()

        # Initial downsampling layers
        self.initial_layers = nn.Sequential(
            conv_block(2, 2, stride=2),  # Downsample by factor of 2 (256 -> 128)
            conv_block(2, 2, stride=2),  # Downsample by factor of 2 (128 -> 64)
        )

        # Quanvolutional layer operates on reduced dimensions
        self.quanv = QuanvLayer(kernel_size=2, stride=2)

        # Remaining layers after Quanvolutional layer
        layers = []
        in_channels = self.quanv.n_qubits  # Update input channels to match QuanvLayer output

        layers.append(conv_block(in_channels, 64))

        # Adjust the number of layers accordingly
        remaining_layers = n_layers - 5

        for _ in range(remaining_layers):
            layers.append(conv_block(64, 64))

        layers.extend([
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),  # Upsample to 232x256
            nn.Conv2d(64, 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(2),
        ])

        self.nw = nn.Sequential(*layers)

    def forward(self, x):
        idt = x  # (batch_size, 2, nrow, ncol)
        # Initial downsampling layers
        x_down = self.initial_layers(x)  # (batch_size, 2, 64, 64)
        # Apply Quanvolutional layer
        x_quanv = self.quanv(x_down)  # Output shape: depends on QuanvLayer
        # Pass through the remaining CNN layers
        x_nw = self.nw(x_quanv)  # This will be upsampled back to original size
        # Add residual connection
        dw = x_nw + idt  # (batch_size, 2, nrow, ncol)
        return dw

# CG Algorithm ======================
class myAtA(nn.Module):
    """
    Performs DC step
    """
    def __init__(self, csm, mask, lam):
        super(myAtA, self).__init__()
        self.csm = csm  # complex (B x ncoil x nrow x ncol)
        self.mask = mask  # complex (B x nrow x ncol)
        self.lam = lam

    def forward(self, im):
        """
        :im: complex image (B x nrow x ncol)
        """
        im_coil = self.csm * im.unsqueeze(1)  # Split coil images (B x ncoil x nrow x ncol)
        k_full = torch.fft.fftn(im_coil, dim=(-2, -1), norm='ortho')  # Convert into k-space
        k_u = k_full * self.mask.unsqueeze(1)  # Apply mask
        im_u_coil = torch.fft.ifftn(k_u, dim=(-2, -1), norm='ortho')  # Convert back to image domain
        im_u = torch.sum(im_u_coil * torch.conj(self.csm), dim=1)  # Coil combine (B x nrow x ncol)
        return im_u + self.lam * im

def myCG(AtA, rhs, max_iter=10, tol=1e-10):
    """
    Performs Conjugate Gradient algorithm
    """
    # Initialize variables
    x = torch.zeros_like(rhs, dtype=torch.complex64)
    r = rhs.clone()
    p = r.clone()
    rTr = torch.sum(torch.conj(r) * r, dim=(-2, -1)).real

    for i in range(max_iter):
        Ap = AtA(p)
        alpha = rTr / (torch.sum(torch.conj(p) * Ap, dim=(-2, -1)).real + 1e-8)
        x = x + alpha.unsqueeze(-1).unsqueeze(-1) * p
        r_new = r - alpha.unsqueeze(-1).unsqueeze(-1) * Ap
        rTr_new = torch.sum(torch.conj(r_new) * r_new, dim=(-2, -1)).real
        if torch.sqrt(rTr_new).mean() < tol:
            break
        beta = rTr_new / (rTr + 1e-8)
        p = r_new + beta.unsqueeze(-1).unsqueeze(-1) * p
        r = r_new
        rTr = rTr_new

    return x

class data_consistency(nn.Module):
    def __init__(self):
        super().__init__()
        self.lam = nn.Parameter(torch.tensor(0.05), requires_grad=True)

    def forward(self, z_k, x0, csm, mask):
        rhs = x0 + self.lam * z_k  # (B, 2, nrow, ncol)
        AtA = myAtA(csm, mask, self.lam)
        rhs_complex = r2c(rhs, axis=1)
        rec = myCG(AtA, rhs_complex)
        rec_real = c2r(rec, axis=1)
        return rec_real

# Model ======================
class MoDL(nn.Module):
    def __init__(self, n_layers, k_iters):
        super().__init__()
        self.k_iters = k_iters
        self.dw = cnn_denoiser(n_layers)
        self.dc = data_consistency()

    def forward(self, x0, csm, mask):
        x_k = x0.clone()
        for k in range(self.k_iters):
            z_k = self.dw(x_k)
            x_k = self.dc(z_k, x0, csm, mask)
        return x_k