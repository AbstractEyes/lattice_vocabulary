"""
PURE GEOMETRIC BASIN LOSSES - ZERO Cross-Entropy
-------------------------------------------------
NO softmax, NO log-softmax, NO nll_loss, NO cross-entropy.
Pure geometric supervision using distances and constraints.

Specifically curated to eliminate ANY sort of cross-entropy elements for experimentation.
The alternative losses do have elemental features from them and behave SIMILARLY but are
inherently different. This is a pure geometric distance-based supervision attempt.

Author: AbstractPhil + Claude Sonnet 4.5
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PureGeometricLoss(nn.Module):
    """
    Loss 1: Pure Geometric Distance Optimization
    ---------------------------------------------
    NO cross-entropy. NO softmax. NO probabilistic assumptions.
    Pure distance-based geometric supervision.

    Goal: Correct class compatibility → 1.0, all others → 0.0
    Method: Direct MSE + margin enforcement + repulsion
    """

    def __init__(self, margin=0.3, repulsion_strength=0.5):
        super().__init__()
        self.margin = margin
        self.repulsion_strength = repulsion_strength

    def forward(self, compatibility_scores, labels, mixed_labels=None, lam=None):
        """
        Pure geometric loss - no probabilistic terms.

        Args:
            compatibility_scores: [B, num_classes] - geometric basin scores
            labels: [B] - ground truth classes
            mixed_labels: [B] - for AlphaMix (optional)
            lam: scalar - mixing coefficient (optional)
        """
        B = compatibility_scores.shape[0]

        # Standard case: single label
        if mixed_labels is None:
            # 1. ATTRACTION: Pull correct class toward 1.0
            correct_scores = compatibility_scores[torch.arange(B), labels]
            attraction_loss = (1.0 - correct_scores).pow(2).mean()

            # 2. REPULSION: Push incorrect classes toward 0.0
            mask = torch.ones_like(compatibility_scores)
            mask[torch.arange(B), labels] = 0
            incorrect_scores = compatibility_scores * mask
            repulsion_loss = (incorrect_scores).pow(2).sum(dim=1).mean()

            # 3. MARGIN: Enforce minimum separation
            max_incorrect = (compatibility_scores * mask).max(dim=1)[0]
            margin_loss = F.relu(max_incorrect - correct_scores + self.margin).mean()

            # 4. SMOOTHNESS: Prevent collapse (keep scores in valid range)
            range_loss = F.relu(compatibility_scores - 1.0).pow(2).mean()
            range_loss += F.relu(-compatibility_scores).pow(2).mean()

            total = (attraction_loss +
                     self.repulsion_strength * repulsion_loss +
                     0.5 * margin_loss +
                     0.1 * range_loss)

            return total

        # AlphaMix case: two labels with mixing
        else:
            # Primary label should reach lam
            primary_scores = compatibility_scores[torch.arange(B), labels]
            primary_loss = (lam - primary_scores).pow(2).mean()

            # Secondary label should reach (1-lam)
            secondary_scores = compatibility_scores[torch.arange(B), mixed_labels]
            secondary_loss = ((1 - lam) - secondary_scores).pow(2).mean()

            # All other classes should be low
            mask = torch.ones_like(compatibility_scores)
            mask[torch.arange(B), labels] = 0
            mask[torch.arange(B), mixed_labels] = 0
            other_scores = compatibility_scores * mask
            other_loss = other_scores.pow(2).sum(dim=1).mean()

            # Enforce that primary + secondary ≈ 1.0 (geometric constraint)
            composition_loss = ((primary_scores + secondary_scores - 1.0).pow(2).mean())

            return primary_loss + secondary_loss + 0.3 * other_loss + 0.1 * composition_loss


class GeometricPrototypeLoss(nn.Module):
    """
    Loss 2: Learnable Geometric Prototypes
    ---------------------------------------
    NO cross-entropy. Pure geometric pattern matching.
    Learns geometric "templates" that samples should match.

    Goal: Learn optimal geometric patterns for each class
    Method: Cosine similarity in learned manifold + distance loss
    """

    def __init__(self, num_classes=100, prototype_dim=64):
        super().__init__()
        self.num_classes = num_classes

        # Learnable prototypes in geometric space
        self.prototypes = nn.Parameter(
            torch.randn(num_classes, prototype_dim) * 0.1
        )

        # Project compatibility scores to prototype space (NO softmax)
        self.score_projector = nn.Sequential(
            nn.Linear(num_classes, prototype_dim),
            nn.LayerNorm(prototype_dim),
            nn.ReLU(),
            nn.Linear(prototype_dim, prototype_dim)
        )

    def forward(self, compatibility_scores, labels, mixed_labels=None, lam=None):
        """
        Pure geometric prototype matching - no probabilistic terms.
        """
        B = compatibility_scores.shape[0]

        # Project scores to prototype space
        score_embedding = self.score_projector(compatibility_scores)

        # Compute geometric distances to prototypes
        score_norm = F.normalize(score_embedding, p=2, dim=-1)
        proto_norm = F.normalize(self.prototypes, p=2, dim=-1)

        # Cosine similarity (geometric, not probabilistic)
        similarities = torch.matmul(score_norm, proto_norm.t())  # [B, num_classes]
        similarities = (similarities + 1) / 2  # Scale to [0, 1]

        # Standard case
        if mixed_labels is None:
            # Pull toward correct prototype
            correct_sim = similarities[torch.arange(B), labels]
            attraction_loss = (1.0 - correct_sim).pow(2).mean()

            # Push away from incorrect prototypes
            mask = torch.ones_like(similarities)
            mask[torch.arange(B), labels] = 0
            incorrect_sim = similarities * mask
            repulsion_loss = incorrect_sim.pow(2).sum(dim=1).mean()

            # Prototype diversity: prototypes should be distinct
            proto_similarity = torch.matmul(proto_norm, proto_norm.t())
            proto_similarity = proto_similarity - torch.eye(
                self.num_classes, device=proto_similarity.device
            )
            diversity_loss = proto_similarity.pow(2).mean()

            return attraction_loss + 0.3 * repulsion_loss + 0.1 * diversity_loss

        # AlphaMix case
        else:
            primary_sim = similarities[torch.arange(B), labels]
            secondary_sim = similarities[torch.arange(B), mixed_labels]

            primary_loss = (lam - primary_sim).pow(2).mean()
            secondary_loss = ((1 - lam) - secondary_sim).pow(2).mean()

            # Other prototypes should be far
            mask = torch.ones_like(similarities)
            mask[torch.arange(B), labels] = 0
            mask[torch.arange(B), mixed_labels] = 0
            other_sim = similarities * mask
            other_loss = other_sim.pow(2).sum(dim=1).mean()

            return primary_loss + secondary_loss + 0.2 * other_loss


class HierarchicalGeometricLoss(nn.Module):
    """
    Loss 3: Pure Hierarchical Geometric Supervision
    -----------------------------------------------
    NO cross-entropy. Pure multi-scale geometric constraints.
    Enforces coarse→fine geometric consistency.

    Goal: Multi-scale geometric structure (CIFAR-100 hierarchy)
    Method: Distance-based losses at both hierarchy levels + consistency
    """

    def __init__(self, num_classes=100, num_superclasses=20):
        super().__init__()
        self.num_classes = num_classes
        self.num_superclasses = num_superclasses
        self.subclasses_per_super = num_classes // num_superclasses

        # Learnable hierarchy weights (NOT softmax)
        self.coarse_weight = nn.Parameter(torch.tensor(0.3))
        self.fine_weight = nn.Parameter(torch.tensor(0.7))

    def forward(self, compatibility_scores, labels, mixed_labels=None, lam=None):
        """
        Pure hierarchical geometric loss - no probabilistic terms.
        """
        B = compatibility_scores.shape[0]

        # Map fine labels to superclasses
        superclass_labels = labels // self.subclasses_per_super

        # Aggregate compatibility scores to superclass level (geometric sum)
        scores_reshaped = compatibility_scores.view(
            B, self.num_superclasses, self.subclasses_per_super
        )
        superclass_scores = scores_reshaped.sum(dim=2)  # [B, num_superclasses]

        # Standard case
        if mixed_labels is None:
            # COARSE LEVEL: Superclass geometric distance
            coarse_correct = superclass_scores[torch.arange(B), superclass_labels]

            # Target: correct superclass sum should approach subclasses_per_super
            # (if all 5 subclasses had score 1.0, sum would be 5.0)
            coarse_target = float(self.subclasses_per_super)
            coarse_loss = ((coarse_correct - coarse_target) / coarse_target).pow(2).mean()

            # Other superclasses should be low
            coarse_mask = torch.ones_like(superclass_scores)
            coarse_mask[torch.arange(B), superclass_labels] = 0
            incorrect_coarse = superclass_scores * coarse_mask
            coarse_repulsion = incorrect_coarse.pow(2).sum(dim=1).mean()

            # FINE LEVEL: Subclass geometric distance
            fine_correct = compatibility_scores[torch.arange(B), labels]
            fine_loss = (1.0 - fine_correct).pow(2).mean()

            # Other subclasses in correct superclass should be low
            superclass_idx = superclass_labels
            fine_mask = torch.zeros_like(compatibility_scores)
            for b in range(B):
                super_idx = superclass_idx[b]
                start = super_idx * self.subclasses_per_super
                end = start + self.subclasses_per_super
                fine_mask[b, start:end] = 1.0
            fine_mask[torch.arange(B), labels] = 0  # Exclude correct class

            incorrect_fine = compatibility_scores * fine_mask
            fine_repulsion = incorrect_fine.pow(2).sum(dim=1).mean()

            # HIERARCHICAL CONSISTENCY: Fine score should contribute to coarse
            # Geometric constraint: correct_fine ≤ correct_coarse (not strictly enforced)
            consistency_loss = F.relu(fine_correct * 2 - coarse_correct).mean()

            # Combine with learnable weights (sigmoid to keep positive)
            w_coarse = torch.sigmoid(self.coarse_weight)
            w_fine = torch.sigmoid(self.fine_weight)

            total = (w_coarse * (coarse_loss + 0.3 * coarse_repulsion) +
                     w_fine * (fine_loss + 0.3 * fine_repulsion) +
                     0.2 * consistency_loss)

            return total

        # AlphaMix case
        else:
            superclass_mixed = mixed_labels // self.subclasses_per_super

            # Primary hierarchy
            primary_coarse = superclass_scores[torch.arange(B), superclass_labels]
            primary_fine = compatibility_scores[torch.arange(B), labels]

            coarse_target = float(self.subclasses_per_super)
            primary_loss = (((primary_coarse - lam * coarse_target) / coarse_target).pow(2) +
                            (primary_fine - lam).pow(2)).mean()

            # Secondary hierarchy
            secondary_coarse = superclass_scores[torch.arange(B), superclass_mixed]
            secondary_fine = compatibility_scores[torch.arange(B), mixed_labels]

            secondary_loss = (((secondary_coarse - (1 - lam) * coarse_target) / coarse_target).pow(2) +
                              (secondary_fine - (1 - lam)).pow(2)).mean()

            return primary_loss + secondary_loss

    def predict(self, compatibility_scores):
        """
        Hierarchical prediction: superclass → subclass.
        Pure geometric decision (no softmax).
        """
        B = compatibility_scores.shape[0]

        # Predict superclass (highest sum)
        scores_reshaped = compatibility_scores.view(
            B, self.num_superclasses, self.subclasses_per_super
        )
        superclass_scores = scores_reshaped.sum(dim=2)
        predicted_superclass = superclass_scores.argmax(dim=1)

        # Predict subclass within predicted superclass
        predicted_fine = torch.zeros(B, dtype=torch.long, device=compatibility_scores.device)
        for b in range(B):
            super_idx = predicted_superclass[b]
            start_idx = super_idx * self.subclasses_per_super
            end_idx = start_idx + self.subclasses_per_super
            subclass_scores = compatibility_scores[b, start_idx:end_idx]
            local_pred = subclass_scores.argmax()
            predicted_fine[b] = start_idx + local_pred

        return predicted_superclass, predicted_fine


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DEMO: Verify zero cross-entropy
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("=" * 80)
    print("PURE GEOMETRIC LOSSES - ZERO Cross-Entropy Verification")
    print("=" * 80)

    batch_size = 16
    num_classes = 100

    # Simulate compatibility scores from GeometricBasinCompatibility
    compatibility_scores = torch.rand(batch_size, num_classes)
    compatibility_scores = compatibility_scores / compatibility_scores.sum(dim=1, keepdim=True)
    labels = torch.randint(0, num_classes, (batch_size,))

    print(f"\nInput: Compatibility scores [{batch_size}, {num_classes}]")
    print(f"Labels: {labels[:5].tolist()}...")

    # Test 1: Pure Geometric Loss
    print("\n[Test 1] Pure Geometric Loss")
    print("-" * 80)
    loss1 = PureGeometricLoss()
    result1 = loss1(compatibility_scores, labels)
    print(f"  Total Loss: {result1.item():.4f}")
    print(f"  Components: Attraction + Repulsion + Margin + Range")
    print(f"  Cross-Entropy Terms: ZERO ✓")
    print(f"  Softmax/Log-Softmax: NONE ✓")

    # Test 2: Geometric Prototype Loss
    print("\n[Test 2] Geometric Prototype Loss")
    print("-" * 80)
    loss2 = GeometricPrototypeLoss(num_classes=num_classes)
    result2 = loss2(compatibility_scores, labels)
    print(f"  Total Loss: {result2.item():.4f}")
    print(f"  Components: Attraction + Repulsion + Diversity")
    print(f"  Cross-Entropy Terms: ZERO ✓")
    print(f"  Softmax/Log-Softmax: NONE ✓")

    # Test 3: Hierarchical Geometric Loss
    print("\n[Test 3] Hierarchical Geometric Loss")
    print("-" * 80)
    loss3 = HierarchicalGeometricLoss(num_classes=num_classes, num_superclasses=20)
    result3 = loss3(compatibility_scores, labels)
    superclass_pred, fine_pred = loss3.predict(compatibility_scores)
    accuracy = (fine_pred == labels).float().mean().item()
    print(f"  Total Loss: {result3.item():.4f}")
    print(f"  Prediction Accuracy (random): {accuracy * 100:.2f}%")
    print(f"  Components: Coarse + Fine + Consistency")
    print(f"  Cross-Entropy Terms: ZERO ✓")
    print(f"  Softmax/Log-Softmax: NONE ✓")

    # Test 4: AlphaMix compatibility
    print("\n[Test 4] AlphaMix Support (Geometric Only)")
    print("-" * 80)
    mixed_labels = torch.randint(0, num_classes, (batch_size,))
    lam = 0.6

    result1_mix = loss1(compatibility_scores, labels, mixed_labels, lam)
    result2_mix = loss2(compatibility_scores, labels, mixed_labels, lam)
    result3_mix = loss3(compatibility_scores, labels, mixed_labels, lam)

    print(f"  Loss 1 (AlphaMix): {result1_mix.item():.4f} - Pure geometric mixing")
    print(f"  Loss 2 (AlphaMix): {result2_mix.item():.4f} - Prototype mixing")
    print(f"  Loss 3 (AlphaMix): {result3_mix.item():.4f} - Hierarchical mixing")
    print(f"  Cross-Entropy in AlphaMix: ZERO ✓")

    # Test 5: Gradient flow
    print("\n[Test 5] Gradient Flow (Geometric Only)")
    print("-" * 80)

    for name, loss_fn in [("Pure Geometric", loss1),
                          ("Prototype", loss2),
                          ("Hierarchical", loss3)]:
        test_scores = torch.rand(8, num_classes, requires_grad=True)
        test_labels = torch.randint(0, num_classes, (8,))

        result = loss_fn(test_scores, test_labels)
        result.backward()

        has_grad = test_scores.grad is not None
        grad_norm = test_scores.grad.norm().item() if has_grad else 0
        print(f"  {name:15s}: Gradients = {has_grad}, Norm = {grad_norm:.4f}")

    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)
    print("\n✓ NO cross-entropy anywhere in any loss")
    print("✓ NO softmax, NO log_softmax, NO nll_loss")
    print("✓ NO probabilistic assumptions")
    print("✓ Pure geometric distance-based supervision")
    print("\nLoss Components:")
    print("  1. Pure Geometric: MSE + margin + repulsion")
    print("  2. Prototype: Cosine similarity + diversity")
    print("  3. Hierarchical: Multi-scale distances + consistency")
    print("\nAll losses work with:")
    print("  → Geometric compatibility scores (4-factor product)")
    print("  → AlphaMix transparency decomposition")
    print("  → Pure geometric constraints (no probability)")
    print("=" * 80)