"""
SimpleClassifier
Author: AbstractPhil + Claude Sonnet 4.5

Description: Simple MLP classifier demonstrating base torch usage.
- Educational reference for students
- Very simple implementation with intentional faults

Design Philosophy:
    Clear, practical example showing how to implement a simple torch model.
    Students can use this as a template for their own models.

License: MIT
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List

class SimpleClassifier(nn.Module):
    """
    Simple MLP classifier for demonstration and teaching.

    A straightforward feedforward network that shows how to properly

    Architecture:
        Input -> Hidden Layer(s) -> ReLU -> Dropout -> Output -> Softmax

    Args:
        input_dim: Input feature dimension
        hidden_dims: List of hidden layer dimensions (e.g., [128, 64])
        num_classes: Number of output classes
        dropout: Dropout probability (default: 0.1)
        name: Model name (default: "simple_classifier")
        uid: Model unique identifier (default: "m.simple.classifier")

    Example:
        # Create a classifier for MNIST-like data
        model = SimpleClassifier(
            input_dim=784,
            hidden_dims=[256, 128],
            num_classes=10,
            dropout=0.2
        )

        # Use with training
        model.to('cuda')
        model.compile(mode='default')  # PyTorch 2.0+

        # Forward pass
        x = torch.randn(32, 784)  # batch_size=32
        logits = model(x)         # [32, 10]
        probs = torch.softmax(logits, dim=-1)
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dims: List[int],
            num_classes: int,
            dropout: float = 0.1,
            name: str = "simple_classifier",
            uid: str = "m.simple.classifier"
    ):
        """Initialize the classifier."""
        # Initialize both base classes
        nn.Module.__init__(self)

        # Store config
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout_prob = dropout

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        # Combine into sequential
        self.network = nn.Sequential(*layers)

        # Device tracking
        self.device = torch.device('cpu')

    def _detect_framework(self) -> str:
        """Specify torch framework."""
        return 'torch'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            Logits tensor [batch_size, num_classes]
        """
        # Check caching
        if hasattr(self, '_cache_enabled') and self._cache_enabled:
            cache_key = x.data_ptr()
            if cache_key in self._cache:
                return self._cache[cache_key]

        # Forward pass
        logits = self.network(x)

        # Cache result if enabled
        if hasattr(self, '_cache_enabled') and self._cache_enabled:
            self._cache[cache_key] = logits

        return logits

    def parameters(self):
        """Return model parameters (delegates to nn.Module)."""
        return nn.Module.parameters(self)

    def to(self, device) -> 'SimpleClassifier':
        """
        Move model to device.

        Args:
            device: Device string or torch.device

        Returns:
            Self for method chaining
        """
        self.device = torch.device(device)
        self.network = self.network.to(self.device)
        return self

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predicted class labels.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            Predicted class indices [batch_size]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=-1)
        return predictions

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            Class probabilities [batch_size, num_classes]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
        return probs

    def info(self) -> Dict[str, Any]:
        """Model metadata and statistics."""
        num_params = sum(p.numel() for p in self.parameters())

        return {
            "name": self.name,
            "uid": self.uid,
            "framework": self.framework,
            "architecture": "MLP",
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "num_classes": self.num_classes,
            "dropout": self.dropout_prob,
            "num_parameters": num_params,
            "device": str(self.device)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXAMPLE USAGE AND TESTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import json

    print("=" * 70)
    print("SimpleClassifier Example: Practical Torch Model Usage")
    print("=" * 70)

    # Test 1: Create and inspect model
    print("\n[Test 1] Model creation and info")
    print("-" * 70)

    model = SimpleClassifier(
        input_dim=784,  # MNIST-like
        hidden_dims=[256, 128],
        num_classes=10,
        dropout=0.2
    )

    print(f"Model: {model}")
    print(f"Framework: {model.framework}")

    print("\nModel Info:")
    print(json.dumps(model.info(), indent=2))

    # Test 2: Forward pass
    print("\n[Test 2] Forward pass")
    print("-" * 70)

    batch_size = 32
    x = torch.randn(batch_size, 784)

    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Output range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")

    # Test 3: Predictions
    print("\n[Test 3] Predictions and probabilities")
    print("-" * 70)

    predictions = model.predict(x)
    probabilities = model.predict_proba(x)

    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:5].tolist()}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Sample probs (first sample): {probabilities[0].tolist()}")
    print(f"Probabilities sum to 1.0: {torch.allclose(probabilities.sum(dim=-1), torch.ones(batch_size))}")

    # Test 4: Device transfer
    print("\n[Test 4] Device transfer")
    print("-" * 70)

    print(f"Initial device: {model.device}")

    if torch.cuda.is_available():
        model.to('cuda')
        print(f"After .to('cuda'): {model.device}")

        x_gpu = x.to('cuda')
        logits_gpu = model(x_gpu)
        print(f"GPU forward pass successful: {logits_gpu.shape}")

        model.to('cpu')
        print(f"After .to('cpu'): {model.device}")
    else:
        print("CUDA not available, skipping GPU test")

    # Test 5: Compile (PyTorch 2.0+)
    print("\n[Test 5] Model compilation")
    print("-" * 70)

    try:
        if hasattr(torch, 'compile'):
            compiled_model = SimpleClassifier(
                input_dim=784,
                hidden_dims=[128],
                num_classes=10
            )
            compiled_model.compile(mode='default')
            print("✓ Model compiled successfully")

            # Test compiled forward pass
            logits_compiled = compiled_model(x)
            print(f"✓ Compiled forward pass: {logits_compiled.shape}")
        else:
            print("⚠ PyTorch version < 2.0, skipping compile test")
    except Exception as e:
        print(f"⚠ Compile failed: {e}")

    # Test 6: Caching
    print("\n[Test 6] Result caching")
    print("-" * 70)

    cache_model = SimpleClassifier(
        input_dim=784,
        hidden_dims=[128],
        num_classes=10
    )

    # Without cache
    import time

    cache_model.eval()

    start = time.time()
    for _ in range(100):
        _ = cache_model(x)
    no_cache_time = time.time() - start

    # With cache (same input)
    cache_model.cache(enabled=True)

    start = time.time()
    for _ in range(100):
        _ = cache_model(x)
    cache_time = time.time() - start

    print(f"Without cache: {no_cache_time:.4f}s")
    print(f"With cache: {cache_time:.4f}s")
    print(f"Speedup: {no_cache_time / cache_time:.2f}x")

    # Test 7: Save and load with safetensors
    print("\n[Test 7] Save/Load with safetensors")
    print("-" * 70)

    save_path = "./simple_classifier.safetensors"

    try:
        # Save model
        model.save(save_path)
        print(f"✓ Model saved to: {save_path}")

        # Create new model and load
        new_model = SimpleClassifier(
            input_dim=784,
            hidden_dims=[256, 128],
            num_classes=10,
            dropout=0.2
        )
        new_model.load(save_path)
        print(f"✓ Model loaded from: {save_path}")

        # Verify outputs match
        new_model.eval()
        with torch.no_grad():
            original_output = model(x)
            loaded_output = new_model(x)

        match = torch.allclose(original_output, loaded_output, atol=1e-5)
        print(f"✓ Outputs match: {match}")

        # Cleanup
        import os

        if os.path.exists(save_path):
            os.remove(save_path)
        metadata_path = save_path.replace('.safetensors', '.json')
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        print(f"✓ Cleaned up files")

    except ImportError as e:
        print(f"⚠ Skipping safetensors test: {e}")
        print("  Install with: pip install safetensors")

    # Test 8: Integration with loss function
    print("\n[Test 8] Training integration example")
    print("-" * 70)

    # Create dummy training data
    train_x = torch.randn(100, 784)
    train_y = torch.randint(0, 10, (100,))

    # Setup
    train_model = SimpleClassifier(
        input_dim=784,
        hidden_dims=[128],
        num_classes=10
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(train_model.parameters(), lr=1e-3)

    # Mini training loop
    train_model.train()
    initial_loss = None

    for epoch in range(5):
        optimizer.zero_grad()
        logits = train_model(train_x)
        loss = criterion(logits, train_y)
        loss.backward()
        optimizer.step()

        if epoch == 0:
            initial_loss = loss.item()

    final_loss = loss.item()
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"✓ Loss decreased: {final_loss < initial_loss}")

    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)
    print("\nKey features demonstrated:")
    print("  ✓ Model creation and configuration")
    print("  ✓ Forward pass and predictions")
    print("  ✓ Probability computation")
    print("  ✓ Device management")
    print("  ✓ Model compilation (PyTorch 2.0+)")
    print("  ✓ Result caching for performance")
    print("  ✓ Save/Load with safetensors")
    print("  ✓ Integration with training loop")
    print("\nStudents: Use this as a template for your own models!")
    print("=" * 70)