import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict
import matplotlib.pyplot as plt


@dataclass
class CantorDistributionConfig:
    """Configuration for multi-model Cantor distribution test."""
    n_models: int = 4
    model_dim: int = 128
    cantor_depth: int = 8
    shared_alpha: bool = True  # Single alpha vs per-model
    learning_rate: float = 0.001
    mask_floor: float = 0.1


def build_cantor_mask(size: int, depth: int) -> torch.Tensor:
    """Build Cantor set mask for connections."""
    mask = torch.zeros(size, size)
    for i in range(size):
        for j in range(size):
            if i == j:
                mask[i, j] = 1.0
            else:
                index = (i * size + j) / (size * size)
                valid = True
                x = index
                for _ in range(depth):
                    x *= 3.0
                    digit = int(x)
                    x -= digit
                    if digit == 1:
                        valid = False
                        break
                if valid:
                    mask[i, j] = 1.0
    return mask


class CantorRegulatedModel(nn.Module):
    """
    Single model that respects shared Cantor constitution.
    """

    def __init__(self, model_id: int, dim: int, cantor_mask: torch.Tensor,
                 shared_alpha: nn.Parameter, mask_floor: float = 0.1):
        super().__init__()
        self.model_id = model_id
        self.dim = dim
        self.mask_floor = mask_floor

        # Constitution (shared, frozen)
        self.register_buffer("constitution", cantor_mask)

        # Global alpha (shared parameter reference)
        self.shared_alpha = shared_alpha

        # Local weights (this model's parameters)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)

        # Internal connectivity matrix (respects constitution)
        self.internal_weights = nn.Parameter(torch.randn(dim, dim) * 0.02)

    def forward(self, x: torch.Tensor) -> Dict:
        """
        Forward pass with constitutional constraints.
        Returns both output and diagnostic info.
        """
        batch_size = x.size(0)

        # Standard processing
        h = F.relu(self.fc1(x))
        h = self.fc2(h)

        # Apply constitutional internal communication
        # effective_weights = floor + alpha * constitution
        effective_weights = (
                                    self.mask_floor +
                                    self.shared_alpha * self.constitution
                            ) * self.internal_weights

        # Internal recurrence through constitutional connections
        communicated = torch.matmul(h, effective_weights)

        # Residual
        output = h + communicated

        return {
            'output': output,
            'hidden': h,
            'effective_connectivity': effective_weights.detach(),
            'alpha_contribution': self.shared_alpha.grad if self.shared_alpha.grad is not None else 0.0
        }


class MultiModelCantorTest(nn.Module):
    """
    Test harness for multiple models sharing Cantor constitution.
    """

    def __init__(self, cfg: CantorDistributionConfig):
        super().__init__()
        self.cfg = cfg
        self.n_models = cfg.n_models

        # THE CONSTITUTION - shared by all models
        cantor_mask = build_cantor_mask(cfg.model_dim, cfg.cantor_depth)
        self.register_buffer("shared_constitution", cantor_mask)

        # GLOBAL ALPHA - single parameter all models reference
        if cfg.shared_alpha:
            self.global_alpha = nn.Parameter(torch.tensor(0.5))
        else:
            # Per-model alphas for comparison
            self.global_alpha = nn.ParameterList([
                nn.Parameter(torch.tensor(0.5)) for _ in range(cfg.n_models)
            ])

        # MODELS - each has own weights but shares constitution + alpha
        self.models = nn.ModuleList([
            CantorRegulatedModel(
                model_id=i,
                dim=cfg.model_dim,
                cantor_mask=self.shared_constitution,
                shared_alpha=self.global_alpha if cfg.shared_alpha else self.global_alpha[i],
                mask_floor=cfg.mask_floor
            )
            for i in range(cfg.n_models)
        ])

        # Task-specific heads (e.g., classification)
        self.task_heads = nn.ModuleList([
            nn.Linear(cfg.model_dim, 10)  # 10-class classification
            for _ in range(cfg.n_models)
        ])

    def forward(self, x: torch.Tensor, model_indices: List[int] = None):
        """
        Run specified models (or all if None).
        Returns outputs and gradient census info.
        """
        if model_indices is None:
            model_indices = list(range(self.n_models))

        outputs = []
        model_info = []

        for idx in model_indices:
            # Model processes
            result = self.models[idx](x)

            # Task prediction
            logits = self.task_heads[idx](result['output'])

            outputs.append(logits)
            model_info.append({
                'model_id': idx,
                'output': result['output'],
                'connectivity': result['effective_connectivity'],
            })

        return {
            'logits': outputs,
            'model_info': model_info,
            'current_alpha': self.global_alpha.item() if self.cfg.shared_alpha else [a.item() for a in
                                                                                     self.global_alpha]
        }

    def compute_gradient_census(self) -> Dict:
        """
        Analyze gradient contributions from each model to shared alpha.
        This is the "voting" mechanism.
        """
        if self.cfg.shared_alpha:
            alpha_grad = self.global_alpha.grad
            if alpha_grad is None:
                return {'consensus': 0.0, 'votes': []}

            # In single alpha case, we need to track which models contributed
            # This is simplified - in practice you'd track per-model contributions
            return {
                'consensus': alpha_grad.item(),
                'magnitude': abs(alpha_grad.item()),
                'direction': 'increase' if alpha_grad.item() > 0 else 'decrease',
                'alpha_value': self.global_alpha.item()
            }
        else:
            # Per-model alphas
            votes = []
            for i, alpha in enumerate(self.global_alpha):
                if alpha.grad is not None:
                    votes.append({
                        'model_id': i,
                        'vote': alpha.grad.item(),
                        'alpha_value': alpha.item()
                    })

            avg_vote = sum(v['vote'] for v in votes) / len(votes) if votes else 0.0

            return {
                'votes': votes,
                'consensus': avg_vote,
                'agreement': self._measure_agreement(votes)
            }

    def _measure_agreement(self, votes: List[Dict]) -> float:
        """
        Measure how much models agree (low std = high agreement).
        """
        if not votes:
            return 0.0
        vote_values = [v['vote'] for v in votes]
        return 1.0 / (1.0 + torch.tensor(vote_values).std().item())


def run_cantor_distribution_test():
    """
    Test 1: Basic gradient census mechanism.
    Do multiple models successfully coordinate through shared alpha?
    """
    print("=" * 70)
    print("CANTOR GLOBAL DISTRIBUTION TEST")
    print("=" * 70)

    # Configuration
    cfg = CantorDistributionConfig(
        n_models=4,
        model_dim=64,  # Smaller for faster testing
        cantor_depth=6,
        shared_alpha=True,
        learning_rate=0.01
    )

    # Create multi-model system
    system = MultiModelCantorTest(cfg)
    optimizer = torch.optim.Adam(system.parameters(), lr=cfg.learning_rate)

    # Synthetic task: different models see different data distributions
    # Test if they can coordinate through alpha despite different objectives

    def generate_task_data(model_id: int, batch_size: int = 32):
        """Each model gets slightly different data distribution."""
        x = torch.randn(batch_size, cfg.model_dim)
        # Add model-specific bias
        x = x + torch.randn(1, cfg.model_dim) * 0.5 * model_id
        y = torch.randint(0, 10, (batch_size,))
        return x, y

    # Training loop
    n_steps = 100
    alpha_history = []
    gradient_history = []
    loss_history = {i: [] for i in range(cfg.n_models)}

    print(f"\nTraining {cfg.n_models} models with shared Cantor constitution...")
    print(f"Initial alpha: {system.global_alpha.item():.4f}")
    print(f"Constitution sparsity: {system.shared_constitution.mean().item():.4f}")
    print()

    for step in range(n_steps):
        optimizer.zero_grad()

        total_loss = 0

        # Each model trains on its task
        for model_id in range(cfg.n_models):
            x, y = generate_task_data(model_id)

            # Forward pass
            result = system(x, model_indices=[model_id])
            logits = result['logits'][0]

            # Task loss
            loss = F.cross_entropy(logits, y)
            total_loss += loss

            loss_history[model_id].append(loss.item())

        # Backward pass - all models contribute gradients to shared alpha
        total_loss.backward()

        # CENSUS: Examine gradient contributions before update
        census = system.compute_gradient_census()
        gradient_history.append(census)
        alpha_history.append(system.global_alpha.item())

        # Update
        optimizer.step()

        # Logging
        if (step + 1) % 20 == 0:
            print(f"Step {step + 1}/{n_steps}")
            print(f"  Alpha: {system.global_alpha.item():.4f}")
            print(f"  Gradient census:")
            print(f"    Consensus: {census['consensus']:.6f}")
            print(f"    Magnitude: {census['magnitude']:.6f}")
            print(f"    Direction: {census['direction']}")
            print(f"  Avg loss: {total_loss.item() / cfg.n_models:.4f}")
            print()

    # Analysis
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    final_alpha = system.global_alpha.item()
    alpha_change = final_alpha - 0.5

    print(f"\nAlpha Evolution:")
    print(f"  Initial: 0.5000")
    print(f"  Final: {final_alpha:.4f}")
    print(f"  Change: {alpha_change:+.4f} ({abs(alpha_change) / 0.5 * 100:.1f}%)")

    print(f"\nGradient Statistics:")
    grad_magnitudes = [g['magnitude'] for g in gradient_history]
    print(f"  Mean magnitude: {sum(grad_magnitudes) / len(grad_magnitudes):.6f}")
    print(f"  Max magnitude: {max(grad_magnitudes):.6f}")

    # Check if models reached consensus
    final_gradients = gradient_history[-10:]  # Last 10 steps
    final_directions = [g['direction'] for g in final_gradients]
    agreement = len(set(final_directions)) == 1

    print(f"\nConsensus Analysis:")
    print(f"  Final direction consensus: {agreement}")
    print(f"  Direction: {final_directions[-1] if final_directions else 'none'}")

    # Visualization
    plot_results(alpha_history, gradient_history, loss_history, cfg)

    return system, {
        'alpha_history': alpha_history,
        'gradient_history': gradient_history,
        'loss_history': loss_history,
        'final_alpha': final_alpha
    }


def plot_results(alpha_history, gradient_history, loss_history, cfg):
    """Visualize the coordination process."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Alpha evolution
    ax = axes[0, 0]
    ax.plot(alpha_history, linewidth=2)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Initial')
    ax.set_title('Global Alpha Evolution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Alpha Value')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Gradient magnitude
    ax = axes[0, 1]
    grad_mags = [abs(g['consensus']) for g in gradient_history]
    ax.plot(grad_mags, linewidth=2, color='orange')
    ax.set_title('Gradient Consensus Magnitude', fontsize=14, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('|Gradient|')
    ax.grid(True, alpha=0.3)

    # Loss curves
    ax = axes[1, 0]
    for model_id, losses in loss_history.items():
        ax.plot(losses, label=f'Model {model_id}', alpha=0.7)
    ax.set_title('Per-Model Task Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Gradient direction
    ax = axes[1, 1]
    directions = [1 if g['direction'] == 'increase' else -1 for g in gradient_history]
    ax.plot(directions, linewidth=2, color='green', marker='o', markersize=3)
    ax.set_title('Gradient Direction (Voting)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Direction (+1=increase, -1=decrease)')
    ax.set_ylim([-1.5, 1.5])
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cantor_distribution_test.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved: cantor_distribution_test.png")
    plt.close()


def test_consensus_breakdown():
    """
    Test 2: What happens when models DISAGREE?
    Give models conflicting objectives and see if alpha stays stable.
    """
    print("\n" + "=" * 70)
    print("CONSENSUS BREAKDOWN TEST")
    print("=" * 70)

    cfg = CantorDistributionConfig(
        n_models=4,
        model_dim=64,
        cantor_depth=8,
        shared_alpha=True,
        learning_rate=0.001
    )

    system = MultiModelCantorTest(cfg)
    optimizer = torch.optim.Adam(system.parameters(), lr=cfg.learning_rate)

    # CONFLICTING TASKS
    # Models 0,1: Want MORE connectivity (benefit from alpha increase)
    # Models 2,3: Want LESS connectivity (benefit from alpha decrease)

    def generate_conflicting_data(model_id: int, batch_size: int = 32):
        x = torch.randn(batch_size, cfg.model_dim)
        if model_id < 2:
            # These models benefit from high alpha (complex patterns)
            x = x + torch.randn(1, cfg.model_dim) * 2.0
        else:
            # These models benefit from low alpha (simple patterns)
            x = x * 0.5
        y = torch.randint(0, 10, (batch_size,))
        return x, y

    n_steps = 100
    alpha_history = []
    gradient_conflicts = []

    print("Training with conflicting objectives...")
    print(f"Models 0,1: Prefer high alpha (complex)")
    print(f"Models 2,3: Prefer low alpha (simple)\n")

    for step in range(n_steps):
        optimizer.zero_grad()

        total_loss = 0

        for model_id in range(cfg.n_models):
            x, y = generate_conflicting_data(model_id)
            result = system(x, model_indices=[model_id])
            loss = F.cross_entropy(result['logits'][0], y)
            total_loss += loss

        total_loss.backward()

        census = system.compute_gradient_census()
        alpha_history.append(system.global_alpha.item())
        gradient_conflicts.append(census['magnitude'])

        optimizer.step()

        if (step + 1) % 20 == 0:
            print(f"Step {step + 1}: Alpha={system.global_alpha.item():.4f}, "
                  f"Conflict={census['magnitude']:.6f}")

    print(f"\nResult:")
    print(f"  Alpha changed: {abs(alpha_history[-1] - alpha_history[0]):.4f}")
    print(f"  Expected: Minimal change due to conflict")
    print(f"  Avg conflict: {sum(gradient_conflicts) / len(gradient_conflicts):.6f}")

    # Check stability
    alpha_std = torch.tensor(alpha_history).std().item()
    print(f"  Alpha stability (std): {alpha_std:.6f}")

    if alpha_std < 0.05:
        print("  ✓ Alpha remained stable despite conflicting objectives")
    else:
        print("  ⚠ Alpha fluctuated - consensus mechanism may need tuning")

    return alpha_history, gradient_conflicts


if __name__ == "__main__":
    # Run tests
    system, results = run_cantor_distribution_test()

    print("\n" + "=" * 70)
    alpha_hist, conflicts = test_consensus_breakdown()
    print("=" * 70)

    print("\n✓ All tests complete")
    print("\nKey findings:")
    print("1. Gradient census mechanism: " +
          ("✓ Working" if results['alpha_history'][-1] != results['alpha_history'][0] else "⚠ No change detected"))
    print("2. Conflict resolution: " +
          ("✓ Stable" if torch.tensor(alpha_hist).std() < 0.05 else "⚠ Unstable"))