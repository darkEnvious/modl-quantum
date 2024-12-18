import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import r2c, c2r

import pennylane as qml
import numpy as np
import time

# Quantum Convolutional Layer ======================
class QuanvLayer(nn.Module):
    def __init__(self, kernel_size=2, stride=1):
        super(QuanvLayer, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_channels = 2  # Process both real and imaginary channels
        self.n_qubits = kernel_size * kernel_size  # Number of qubits remains the same
        self.n_layers = 1  # Number of layers in variational circuit

        # Define the quantum device with vectorization enabled
        self.dev = qml.device('default.qubit', wires=self.n_qubits)

        # Define the weight shapes for the quantum layer
        weight_shapes = {"weights": (self.n_layers, self.n_qubits)}

        # Define the quantum circuit with vectorization
        @qml.qnode(self.dev, interface='torch', diff_method="backprop", vectorized=True)
        def circuit(inputs, weights):
            # inputs shape: (batch_size, input_size)
            for i in range(self.n_qubits):
                qml.RX(inputs[:, 2 * i], wires=i)
                qml.RY(inputs[:, 2 * i + 1], wires=i)
            qml.templates.BasicEntanglerLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

        # Create the QNode and QuantumLayer
        self.q_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        # x: (batch_size, 2, height, width)

        # Extract patches
        patches = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        # patches shape: (batch_size, channels, new_height, new_width, kernel_size, kernel_size)

        n_patches_h = patches.shape[2]
        n_patches_w = patches.shape[3]

        # Move channels to last dimension and reshape
        patches = patches.permute(0, 2, 3, 4, 5, 1)
        # Shape: (batch_size, n_patches_h, n_patches_w, kernel_size, kernel_size, channels)
        patches = patches.reshape(batch_size, n_patches_h * n_patches_w, -1)
        # Shape: (batch_size, n_patches_total, input_size)

        # Normalize patches
        patches_min = patches.view(batch_size, -1).min(dim=1, keepdim=True)[0].unsqueeze(-1)
        patches_max = patches.view(batch_size, -1).max(dim=1, keepdim=True)[0].unsqueeze(-1)
        patches = (patches - patches_min) / (patches_max - patches_min + 1e-8) * np.pi

        # Reshape patches to (batch_size * n_patches_total, input_size)
        patches = patches.view(-1, self.n_channels * self.kernel_size * self.kernel_size)

        # Apply the quantum layer to all patches
        # Split patches into smaller batches to avoid memory issues
        outputs = []
        batch_size_patches = 1024  # Adjust this number as needed
        total_patches = patches.shape[0]
        for i in range(0, total_patches, batch_size_patches):
            batch_patches = patches[i:i+batch_size_patches]
            batch_outputs = self.q_layer(batch_patches)
            outputs.append(batch_outputs)

        outputs = torch.cat(outputs, dim=0)

        # Reshape back to (batch_size, n_qubits, n_patches_h, n_patches_w)
        outputs = outputs.view(batch_size, n_patches_h, n_patches_w, -1)
        outputs = outputs.permute(0, 3, 1, 2)  # shape: (batch_size, n_qubits, n_patches_h, n_patches_w)

        return outputs

# CNN Denoiser ======================
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class cnn_denoiser(nn.Module):
    def __init__(self, n_layers):
        super().__init__()

        self.quanv = QuanvLayer(kernel_size=2, stride=1)  # Adjusted stride

        layers = []
        in_channels = self.quanv.n_qubits

        layers.append(conv_block(in_channels, 64))

        for _ in range(n_layers - 2):
            layers.append(conv_block(64, 64))

        layers.extend([
            nn.Conv2d(64, 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(2),
        ])

        self.nw = nn.Sequential(*layers)

    def forward(self, x):
        idt = x  # (batch_size, 2, nrow, ncol)

        # Apply Quanvolutional layer
        x_quanv = self.quanv(x)  # Output shape: (batch_size, n_qubits, height', width')

        # Upsample to match original dimensions if needed
        x_quanv_resized = F.interpolate(x_quanv, size=(idt.shape[2], idt.shape[3]), mode='bilinear', align_corners=True)

        # Pass through the remaining CNN layers
        x_nw = self.nw(x_quanv_resized)  # This will be same size as idt

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