import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict
import os

from ml.model import SantoNeuroNet, value_loss
from ml.dataset import SantoDataset


class SantoTrainer:
    """
    training neural network on datasets
    """
    def __init__(
            self,
            model: SantoNeuroNet,
            learning_rate: float = 0.001,
            weight_decay: float = 1e-4,
            device: str = "cpu" # Change to "cuda" if good GPU
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Track training metrics
        self.train_losses = []
        self.val_losses = []

    def prepare_dataloader(self, dataset: SantoDataset,
                           batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """
        Convert SantoDataset to PyTorch DataLoader
        """
        if len(dataset.states) == 0:
            raise ValueError("Dataset is empty!")

        # Convert to tensors
        states = torch.stack([torch.from_numpy(s) for s in dataset.states])
        actions = torch.stack([torch.from_numpy(a) for a in dataset.actions])
        values = torch.tensor(dataset.scores, dtype=torch.float32)

        # Concatenate board state + action encoding
        inputs = torch.cat([states, actions], dim=1)  # (N, 14, 5, 5)

        # Create dataset and dataloader
        torch_dataset = TensorDataset(inputs, values)
        return DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle)

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch
        """
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0

        for batch_inputs, batch_targets in dataloader:
            batch_inputs = batch_inputs.to(self.device)
            batch_targets = batch_targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            predictions = self.model(batch_inputs)

            # Compute loss
            loss_dict = value_loss(predictions, batch_targets)
            loss = loss_dict["loss"]
            mae = loss_dict["mae"]

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            total_mae += mae.item()
            num_batches += 1

        return {
            "loss": total_loss / num_batches,
            "mae": total_mae / num_batches
        }

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate on dataset without updating weights
        """
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_inputs, batch_targets in dataloader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)

                # Forward pass
                predictions = self.model(batch_inputs)

                # Compute loss
                loss_dict = value_loss(predictions, batch_targets)
                loss = loss_dict["loss"]
                mae = loss_dict["mae"]

                total_loss += loss.item()
                total_mae += mae.item()
                num_batches += 1

        return {
            "loss": total_loss / num_batches,
            "mae": total_mae / num_batches
        }

    def train_on_dataset(self, dataset_path: str, epochs: int = 10, batch_size: int = 32,
                         validation_split: float = 0.2, save_path: Optional[str] = None,
                         verbose: bool = True) -> Dict[str, list]:
        """
        Main training loop on a dataset file

        Args:
            dataset_path: Path to dataset file
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            save_path: Path to save best model
            verbose: Whether to print training progress

        Returns:
            Dictionary with training history
        """
        # Load dataset
        dataset = SantoDataset.load(dataset_path)
        print(f"Loaded dataset with {len(dataset.states)} samples")

        if len(dataset.states) == 0:
            print("Dataset is empty, skipping training")
            return {"train_losses": [], "val_losses": []}

        # Split into train/validation
        n_samples = len(dataset.states)
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val

        # Create train/val datasets
        train_dataset = SantoDataset()
        val_dataset = SantoDataset()

        train_dataset.states = dataset.states[:n_train]
        train_dataset.actions = dataset.actions[:n_train]
        train_dataset.scores = dataset.scores[:n_train]

        if n_val > 0:
            val_dataset.states = dataset.states[n_train:]
            val_dataset.actions = dataset.actions[n_train:]
            val_dataset.scores = dataset.scores[n_train:]

        # Create dataloaders
        train_loader = self.prepare_dataloader(train_dataset, batch_size, shuffle=True)
        val_loader = self.prepare_dataloader(val_dataset, batch_size, shuffle=False) if n_val > 0 else None

        print(f"Training on {n_train} samples, validating on {n_val} samples")

        best_val_loss = float('inf')
        history = {"train_losses": [], "val_losses": []}

        # Training loop
        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            history["train_losses"].append(train_metrics["loss"])

            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                history["val_losses"].append(val_metrics["loss"])
                val_loss = val_metrics["loss"]

                # Save model
                if save_path and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.model.save_checkpoint(save_path, self.optimizer, epoch)
                    if verbose:
                        print(f"Epoch {epoch + 1}/{epochs}: Saved new best model (val_loss: {val_loss:.4f})")

                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs}: "
                          f"train_loss: {train_metrics['loss']:.4f}, "
                          f"train_mae: {train_metrics['mae']:.4f}, "
                          f"val_loss: {val_loss:.4f}, "
                          f"val_mae: {val_metrics['mae']:.4f}")
            else:
                # No validation set
                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs}: "
                          f"train_loss: {train_metrics['loss']:.4f}, "
                          f"train_mae: {train_metrics['mae']:.4f}")

        return history


def train_from_dataset_file(dataset_path: str, model_save_path: str = "../learned_models/best.pt",
        epochs: int = 10, batch_size: int = 32, learning_rate: float = 0.001):

    # Create model and trainer
    model = SantoNeuroNet(
        in_channels=14,
        filters=64,
        n_conv_blocks=3,
        value_hidden=128
    )

    trainer = SantoTrainer(
        model=model,
        learning_rate=learning_rate,
        device="cpu"  # Change to "cuda" if good GPU
    )

    # Train
    print(f"Starting training on {dataset_path}")
    history = trainer.train_on_dataset(
        dataset_path=dataset_path,
        epochs=epochs,
        batch_size=batch_size,
        save_path=model_save_path,
        verbose=True
    )

    print(f"Training completed. Model saved to {model_save_path}")
    return history


if __name__ == "__main__":

    dataset_path = "../datasets/guided_games.npz"
    if os.path.exists(dataset_path):
        history = train_from_dataset_file(
            dataset_path=dataset_path,
            model_save_path="../learned_models/guided_model.pt",
            epochs=20,
            batch_size=16,
            learning_rate=0.001
        )
        print("Training history:", history)
    else:
        print(f"Dataset file not found: {dataset_path}")
        print("Please generate a dataset first using selfplay or other methods")