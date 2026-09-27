"""
Fourier Neural Operator (FNO) implementation

Based on: "Fourier Neural Operator for Parametric Partial Differential Equations"
Li et al., ICLR 2021

FNO implementation for 2D time-dependent problems
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import numpy as np


class SpectralConv2d(nn.Module):
    """
    2D Fourier layer

    Performs a linear transformation in Fourier space, which realizes a global convolution
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int
    ):
        """
        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            modes1: number of Fourier modes retained in the first dimension
            modes2: number of Fourier modes retained in the second dimension
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        # Learnable Fourier coefficients (complex weights)
        self.scale = 1 / (in_channels * out_channels)

        # Two sets of weights: positive and negative frequencies
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(
        self,
        input: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Complex matrix multiplication

        Args:
            input: (batch, in_channels, x, y)
            weights: (in_channels, out_channels, x, y)

        Returns:
            (batch, out_channels, x, y)
        """
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: (batch, in_channels, nx, ny)

        Returns:
            (batch, out_channels, nx, ny)
        """
        batchsize = x.shape[0]
        nx, ny = x.shape[-2], x.shape[-1]

        # Fourier transform
        x_ft = torch.fft.rfft2(x)

        # Create the output tensor
        out_ft = torch.zeros(
            batchsize, self.out_channels, nx, ny // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        # Multiply on the retained modes
        # Positive frequency part
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        # Negative frequency part (high frequency)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        # Inverse Fourier transform
        x = torch.fft.irfft2(out_ft, s=(nx, ny))

        return x


class FNOBlock2d(nn.Module):
    """
    Basic FNO block

    Contains: Fourier layer + local linear layer + activation function
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        activation: str = 'gelu'
    ):
        super().__init__()

        self.spectral_conv = SpectralConv2d(in_channels, out_channels, modes1, modes2)
        self.linear = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'silu': nn.SiLU()
        }
        self.activation = activations.get(activation, nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, channels, nx, ny)
        """
        x1 = self.spectral_conv(x)
        x2 = self.linear(x)
        x = x1 + x2
        x = self.activation(x)
        return x


class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator

    Learns the mapping input function -> output function
    Suitable for 2D time-dependent PDEs: learns u(x,y,t=0) -> u(x,y,t)
    """

    def __init__(
        self,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 32,
        n_layers: int = 4,
        in_channels: int = 3,   # default: x, y, u0 (or x, y, t)
        out_channels: int = 1,
        activation: str = 'gelu',
        padding: int = 0
    ):
        """
        Args:
            modes1, modes2: number of Fourier modes
            width: hidden layer width
            n_layers: number of FNO blocks
            in_channels: number of input channels
            out_channels: number of output channels
            activation: activation function
            padding: spatial padding (for non-periodic boundaries)
        """
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        self.padding = padding

        # Input projection layer
        self.fc0 = nn.Linear(in_channels, width)

        # FNO blocks
        self.fno_blocks = nn.ModuleList([
            FNOBlock2d(width, width, modes1, modes2, activation)
            for _ in range(n_layers)
        ])

        # Output projection layer
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

        self.activation = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        grid: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: input function values (batch, nx, ny, in_channels) or (batch, nx, ny)
            grid: optional grid coordinates (batch, nx, ny, 2)

        Returns:
            output function values (batch, nx, ny, out_channels) or (batch, nx, ny)
        """
        # Ensure the input is 4D
        if x.dim() == 3:
            x = x.unsqueeze(-1)

        # Add grid coordinates as extra channels
        if grid is not None:
            x = torch.cat([x, grid], dim=-1)

        batch_size, nx, ny, _ = x.shape

        # Input projection: (batch, nx, ny, in_channels) -> (batch, nx, ny, width)
        x = self.fc0(x)

        # Convert to (batch, width, nx, ny) for the convolutions
        x = x.permute(0, 3, 1, 2)

        # Optional padding (for non-periodic boundaries)
        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        # FNO blocks
        for block in self.fno_blocks:
            x = block(x)

        # Remove the padding
        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding]

        # Convert back to (batch, nx, ny, width)
        x = x.permute(0, 2, 3, 1)

        # Output projection
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        # If there is a single output channel, remove the last dimension
        if x.shape[-1] == 1:
            x = x.squeeze(-1)

        return x

    def get_grid(
        self,
        shape: Tuple[int, int],
        device: torch.device
    ) -> torch.Tensor:
        """
        Generate normalized grid coordinates

        Args:
            shape: (nx, ny)
            device: device

        Returns:
            grid tensor (1, nx, ny, 2)
        """
        nx, ny = shape
        x = torch.linspace(0, 1, nx, device=device)
        y = torch.linspace(0, 1, ny, device=device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        grid = torch.stack([X, Y], dim=-1)
        return grid.unsqueeze(0)


class FNO2dTime(nn.Module):
    """
    Time-dependent 2D FNO

    Learns the mapping u(x,y,t=0) -> u(x,y,t)
    Time is used as an additional input condition
    """

    def __init__(
        self,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 32,
        n_layers: int = 4,
        activation: str = 'gelu',
        padding: int = 0
    ):
        """
        Input: initial condition u0(x,y) + grid coordinates (x,y) + time t
        Output: solution u(x,y,t)
        """
        super().__init__()

        # Input channels: u0, x, y, t = 4
        self.fno = FNO2d(
            modes1=modes1,
            modes2=modes2,
            width=width,
            n_layers=n_layers,
            in_channels=4,
            out_channels=1,
            activation=activation,
            padding=padding
        )

    def forward(
        self,
        u0: torch.Tensor,
        t: torch.Tensor,
        grid: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            u0: initial condition (batch, nx, ny)
            t: time (batch,) or (batch, 1)
            grid: grid coordinates (batch, nx, ny, 2); generated automatically if None

        Returns:
            u: predicted solution (batch, nx, ny)
        """
        batch_size, nx, ny = u0.shape
        device = u0.device

        # Generate the grid
        if grid is None:
            grid = self.fno.get_grid((nx, ny), device)
            grid = grid.expand(batch_size, -1, -1, -1)

        # Handle the time dimension
        if t.dim() == 1:
            t = t.view(batch_size, 1, 1, 1)
        else:
            t = t.view(batch_size, 1, 1, 1)

        # Expand the time to the spatial dimensions
        t_expanded = t.expand(batch_size, nx, ny, 1)

        # Combine the inputs: (batch, nx, ny, 4)
        x = torch.cat([
            u0.unsqueeze(-1),  # (batch, nx, ny, 1)
            grid,              # (batch, nx, ny, 2)
            t_expanded         # (batch, nx, ny, 1)
        ], dim=-1)

        return self.fno(x)


class FNOTrainer:
    """FNO trainer"""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            model: FNO model
            optimizer: optimizer
            scheduler: learning rate scheduler
            device: compute device
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=1e-3)
        self.scheduler = scheduler

        self.history = {
            'loss': [],
            'val_loss': []
        }

    def train_step(
        self,
        u0: torch.Tensor,
        t: torch.Tensor,
        u_true: torch.Tensor,
        grid: Optional[torch.Tensor] = None
    ) -> float:
        """
        Single training step

        Args:
            u0: initial condition (batch, nx, ny)
            t: time (batch,)
            u_true: ground-truth solution (batch, nx, ny)
            grid: grid coordinates

        Returns:
            loss value
        """
        self.model.train()
        self.optimizer.zero_grad()

        u_pred = self.model(u0, t, grid)
        loss = F.mse_loss(u_pred, u_true)

        loss.backward()
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return loss.item()

    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> float:
        """
        Train for one epoch

        Args:
            dataloader: data loader

        Returns:
            average loss
        """
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            u0 = batch['input'].to(self.device)
            t = batch['t'].to(self.device)
            u_true = batch['output'].to(self.device)
            grid = batch.get('grid')
            if grid is not None:
                grid = grid.to(self.device)

            loss = self.train_step(u0, t, u_true, grid)
            total_loss += loss
            n_batches += 1

        avg_loss = total_loss / n_batches
        self.history['loss'].append(avg_loss)
        return avg_loss

    @torch.no_grad()
    def evaluate(
        self,
        u0: torch.Tensor,
        t: torch.Tensor,
        u_true: torch.Tensor,
        grid: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Evaluate the model

        Args:
            u0: initial condition
            t: time
            u_true: ground-truth solution
            grid: grid coordinates

        Returns:
            dictionary of evaluation metrics
        """
        self.model.eval()

        u_pred = self.model(u0.to(self.device), t.to(self.device), grid)
        u_true = u_true.to(self.device)

        # Pointwise absolute error
        abs_error = torch.abs(u_pred - u_true)

        metrics = {
            'mae': abs_error.mean().item(),
            'max_error': abs_error.max().item(),
            'mse': F.mse_loss(u_pred, u_true).item(),
            'relative_l2': (torch.norm(u_pred - u_true) / torch.norm(u_true)).item()
        }

        return metrics


def test_fno():
    """Test the FNO implementation"""
    print("=" * 60)
    print("Testing FNO implementation")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Test the base FNO2d
    print("\n1. Testing the base FNO2d...")
    model = FNO2d(
        modes1=12, modes2=12,
        width=32, n_layers=4,
        in_channels=3, out_channels=1
    ).to(device)

    print(f"   Number of model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test the input
    batch_size, nx, ny = 4, 64, 64
    x = torch.randn(batch_size, nx, ny, 3, device=device)

    y = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")

    # Test the time-dependent FNO
    print("\n2. Testing FNO2dTime...")
    model_t = FNO2dTime(
        modes1=12, modes2=12,
        width=32, n_layers=4
    ).to(device)

    print(f"   Number of model parameters: {sum(p.numel() for p in model_t.parameters()):,}")

    u0 = torch.randn(batch_size, nx, ny, device=device)
    t = torch.rand(batch_size, device=device)

    u = model_t(u0, t)
    print(f"   u0 shape: {u0.shape}")
    print(f"   t shape: {t.shape}")
    print(f"   Output shape: {u.shape}")

    # Test the gradients
    print("\n3. Testing gradient computation...")
    loss = u.mean()
    loss.backward()
    print("   Gradient computation succeeded!")

    print("\nAll tests passed!")


if __name__ == '__main__':
    test_fno()
