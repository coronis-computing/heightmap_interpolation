# Copyright (c) 2020 Coronis Computing S.L. (Spain)
# All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Author: Ricard Campos (ricard.campos@coronis.es)
try:
    import torch
    import torch.nn as nn

    _HAS_EXPERIMENTAL = True
except ImportError:
    _HAS_EXPERIMENTAL = False
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from heightmap_interpolation.interpolants.interpolant import Interpolant


class FourierFeatures(nn.Module):
    """
    Fourier feature encoding for coordinate inputs.
    Maps coordinates to high-dimensional space using sinusoidal functions.
    Helps network learn high-frequency details.
    """

    def __init__(self, input_dim, num_frequencies=10, scale=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies

        # Random Fourier features: sample from Gaussian
        # B matrix shape: (input_dim, num_frequencies)
        self.register_buffer("B", torch.randn(input_dim, num_frequencies) * scale)

    def forward(self, x):
        # x: (batch, input_dim)
        # x @ B: (batch, num_frequencies)
        x_proj = 2 * np.pi * x @ self.B

        # Concatenate sin and cos
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def get_output_dim(self):
        return 2 * self.num_frequencies


class MLPScatteredInterpolator(nn.Module):
    """Neural network for scattered data interpolation."""

    def __init__(
        self,
        input_dim=2,
        hidden_dims=[64, 128, 128, 64],
        output_dim=1,
        use_fourier=False,
        num_frequencies=10,
        fourier_scale=1.0,
    ):
        super().__init__()

        self.use_fourier = use_fourier

        # Optional Fourier feature encoding
        if use_fourier:
            self.fourier = FourierFeatures(input_dim, num_frequencies, fourier_scale)
            network_input_dim = self.fourier.get_output_dim()
        else:
            self.fourier = None
            network_input_dim = input_dim

        layers = []
        prev_dim = network_input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.Tanh(),  # Smooth activation for interpolation
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_fourier:
            x = self.fourier(x)
        return self.network(x)


class MLPScatteredDataInterpolatorTrainer:
    """Trainer for scattered data interpolation."""

    def __init__(self, model, lr=1e-3, smoothness_weight=0.0, device=None):
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss(
            reduction="none"
        )  # reduction='none' is necessary for per-sample weighting
        self.smoothness_weight = smoothness_weight
        self.history = {"loss": [], "data_loss": [], "smoothness_loss": []}

    def compute_laplacian_loss(self, X):
        """
        Compute Laplacian regularization: encourages smooth interpolation.
        Penalizes large second derivatives (curvature).
        """
        X = X.requires_grad_(True)
        y = self.model(X)

        # Compute first derivatives
        grads = torch.autograd.grad(y.sum(), X, create_graph=True)[0]

        # Compute second derivatives (Laplacian)
        laplacian = 0
        for i in range(X.shape[1]):
            grad_i = grads[:, i]
            second_deriv = torch.autograd.grad(grad_i.sum(), X, create_graph=True)[0][
                :, i
            ]
            laplacian += second_deriv**2

        return laplacian.mean()

    def compute_gradient_penalty(self, X):
        """
        Penalize large gradients: encourages smoother interpolation.
        Useful when you know the function shouldn't vary too rapidly.
        """
        X = X.requires_grad_(True)
        y = self.model(X)

        grads = torch.autograd.grad(y.sum(), X, create_graph=True)[0]
        gradient_penalty = (grads**2).sum(dim=1).mean()

        return gradient_penalty

    def compute_conservation_loss(self, X):
        """
        Example: enforce that integral over domain is conserved.
        Useful for physical quantities like mass, energy, etc.
        """
        y = self.model(X)
        # For example, enforce mean value constraint
        mean_value = y.mean()
        target_mean = 0.0  # Adjust based on your problem
        return (mean_value - target_mean) ** 2

    def compute_sample_weights(self, X_train, method="density", k=10):
        """
        Compute per-sample weights based on local density.

        Args:
            X_train: training coordinates
            method: 'density' (inverse density weighting) or 'uniform'
            k: number of neighbors to consider for density estimation

        Returns:
            weights: tensor of shape (N,) with higher weights for sparse regions
        """
        if method == "uniform":
            return torch.ones(len(X_train), device=self.device)

        # Compute pairwise distances
        X_train_np = X_train.cpu().numpy()
        from scipy.spatial import cKDTree

        tree = cKDTree(X_train_np)

        # For each point, find distance to k-th nearest neighbor
        distances, _ = tree.query(X_train_np, k=k + 1)  # k+1 because first is self
        avg_distances = distances[:, 1:].mean(axis=1)  # Average distance to k neighbors

        # Inverse density weighting: sparse areas get higher weights
        # Add small epsilon to avoid division by zero
        # weights = 1.0 / (avg_distances + 1e-6)
        weights = avg_distances + 1e-6

        # Normalize weights to have mean of 1
        weights = weights / weights.mean()

        return torch.FloatTensor(weights).to(self.device)

    def prepare_loss_plots(self):
        """Prepare loss plots. Called when show_loss_plots=True."""
        # Select an interactive backend lazily, so that merely importing this
        # module does not require a GUI backend (e.g. on headless machines).
        matplotlib.use("TkAgg")
        plt.ion()
        fig, self.axes = plt.subplots(1, 2, figsize=(14, 4))
        fig.suptitle("Training Progress (Live)", fontsize=14)

    def fit(
        self,
        X_train,
        y_train,
        epochs=5000,
        use_density_weighting=False,
        batch_sampling=None,
        verbose=True,
        show_loss_plots=False,
        verbose_every=500,
        loss_plots_every=500,
    ):
        """
        Train the interpolator with optional physics-informed losses.

        Args:
            X_train: coordinates (N, input_dim)
            y_train: values (N, output_dim)
            epochs: number of training epochs
            verbose: print progress
        """
        # Prepare the loss viewer
        if show_loss_plots:
            self.prepare_loss_plots()

        # Move data to device
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)

        # Compute sample weights if using density weighting
        sample_weights = None
        if use_density_weighting:
            print("Computing density-based sample weights...")
            sample_weights = self.compute_sample_weights(
                X_train, method="density", k=10
            )
            print(
                f"Weight range: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]"
            )

        self.model.train()

        if verbose:
            print(f"Starting training for {epochs} epochs...")

        n_samples = len(X_train)
        for epoch in range(epochs):
            self.optimizer.zero_grad()

            # Optionally use mini-batch training with density-aware sampling
            if batch_sampling is not None and n_samples > 256:
                batch_size = min(256, n_samples)
                if batch_sampling == "density_aware" and sample_weights is not None:
                    # Sample with probability proportional to weights (emphasize sparse regions)
                    probs = sample_weights / sample_weights.sum()
                    indices = torch.multinomial(probs, batch_size, replacement=False)
                else:
                    # Uniform random sampling
                    indices = torch.randperm(n_samples, device=self.device)[:batch_size]

                X_batch = X_train[indices]
                y_batch = y_train[indices]
                weights_batch = (
                    sample_weights[indices] if sample_weights is not None else None
                )
            else:
                X_batch = X_train
                y_batch = y_train
                weights_batch = sample_weights

            # Data fitting loss
            y_pred = self.model(X_batch)
            per_sample_loss = self.criterion(y_pred, y_batch).mean(dim=1)

            # Apply sample weights if provided
            if weights_batch is not None:
                data_loss = (per_sample_loss * weights_batch).mean()
            else:
                data_loss = per_sample_loss.mean()

            # Smoothness loss (optional)
            smoothness_loss = 0.0
            if self.smoothness_weight > 0:
                # Laplacian regularization for smoothness
                smoothness_loss = self.compute_laplacian_loss(X_train)
                # Gradient penalty for smoothness
                # smoothness_loss = self.compute_gradient_penalty(X_train)

            # Total loss
            total_loss = data_loss + self.smoothness_weight * smoothness_loss

            total_loss.backward()
            self.optimizer.step()

            self.history["loss"].append(total_loss.item())
            self.history["data_loss"].append(data_loss.item())
            self.history["smoothness_loss"].append(
                smoothness_loss
                if isinstance(smoothness_loss, float)
                else smoothness_loss.item()
            )

            # Ouput progress
            if verbose and (epoch + 1) % verbose_every == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, Total Loss: {total_loss.item():.6f}, "
                    f"Data: {data_loss.item():.6f}, Smoothness: {self.history['smoothness_loss'][-1]:.6f}"
                )
            if show_loss_plots and (epoch + 1) % loss_plots_every == 0:
                self.plot_loss()

        if show_loss_plots:
            print("Training complete!")
            print("Closing loss plots...")
            plt.ioff()
            plt.close()

        return self

    def predict(self, X):
        """Predict values at new coordinates."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_pred = self.model(X_tensor)
        return y_pred.cpu().numpy()

    def plot_loss(self):
        """Plot training losses."""
        # fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        self.axes[0].clear()
        self.axes[1].clear()

        self.axes[0].plot(self.history["loss"], label="Total Loss")
        self.axes[0].set_xlabel("Epoch")
        self.axes[0].set_ylabel("Loss")
        self.axes[0].set_title("Total Training Loss")
        self.axes[0].set_yscale("log")
        self.axes[0].grid(True)
        self.axes[0].legend()

        self.axes[1].plot(self.history["data_loss"], label="Data Loss")
        self.axes[1].plot(self.history["smoothness_loss"], label="Smoothness Loss")
        self.axes[1].set_xlabel("Epoch")
        self.axes[1].set_ylabel("Loss")
        self.axes[1].set_title("Loss Components")
        self.axes[1].set_yscale("log")
        self.axes[1].grid(True)
        self.axes[1].legend()

        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)


class MLPInterpolant(Interpolant):
    def __init__(
        self,
        x,
        y,
        z,  # Interpolant basis
        hidden_dims=[64, 128, 128, 128, 128, 64],
        output_dim=1,
        use_fourier=False,
        num_frequencies=10,
        fourier_scale=1.0,  # Model params
        lr=1e-3,
        smoothness_weight=0.0,
        use_density_weighting=True,
        epochs=10000,
        device=None,
        verbose=True,
        show_loss_plots=False,
    ):
        if not _HAS_EXPERIMENTAL:
            raise ImportError(
                "The MLP interpolant is an experimental feature and requires optional dependencies."
                "Install them with: pip install heightmap_interpolation[experimental]"
            )

        super().__init__(x, y, z)

        # Normalize data
        self.x_mean = x.mean()
        self.x_std = x.std()
        self.y_mean = y.mean()
        self.y_std = y.std()
        self.z_mean = z.mean()
        self.z_std = z.std()
        x = (x - self.x_mean) / self.x_std
        y = (y - self.y_mean) / self.y_std
        z = (z - self.z_mean) / self.z_std

        # Create the MLP model
        self.model = MLPScatteredInterpolator(
            input_dim=2,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            use_fourier=use_fourier,
            num_frequencies=num_frequencies,
            fourier_scale=fourier_scale,
        )

        # Create the trainer
        self.trainer = MLPScatteredDataInterpolatorTrainer(
            model=self.model, lr=lr, smoothness_weight=smoothness_weight, device=device
        )

        # Train the model on the input reference data (x, y, z)
        self.trainer.fit(
            np.column_stack((x, y)),
            np.expand_dims(z, 1),
            epochs=epochs,
            verbose=verbose,
            show_loss_plots=show_loss_plots,
            use_density_weighting=use_density_weighting,
        )

    def __call__(self, x, y):
        # Normalize x/y coordinates
        x = (x - self.x_mean) / self.x_std
        y = (y - self.y_mean) / self.y_std
        X = np.column_stack((x, y))
        z = self.trainer.predict(X).ravel()
        # Denormalize predicted zs
        z = z * self.z_std + self.z_mean
        return z
