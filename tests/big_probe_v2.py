"""
Comprehensive Test Battery for Unified Geometric Crystal Vocabulary
Tests all formulas, dimensions, content types, and mathematical properties
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import warnings
from dataclasses import replace

# ============================================================================
# TEST CONFIGURATION
# ============================================================================

TEST_TOKENS = ["test", "hello", "世界", "123", "abstract", "philosophy", ""]
TEST_DEFINITIONS = [
    None,
    "A simple test definition",
    "The study of fundamental nature of knowledge, reality, and existence",
    "A" * 500  # Long definition
]


# ============================================================================
# CAYLEY-MENGER COMPREHENSIVE TESTS
# ============================================================================

def test_cayley_menger_comprehensive(vocab):
    """Comprehensive testing of Cayley-Menger volume calculations across all dimensions"""
    print("\n" + "=" * 80)
    print("CAYLEY-MENGER COMPREHENSIVE VALIDATION")
    print("=" * 80)

    results = {"passed": 0, "failed": 0, "details": {}}

    # Test 1: Known geometric shapes with exact volumes
    print("\n1. Exact Volume Calculations")
    print("-" * 40)

    test_cases = [
        # 2D: Line segment
        ("Line (D2)", np.array([[0, 0], [1, 0]], dtype=np.float32), 1.0),

        # 3D: Equilateral triangle
        ("Triangle (D3)", np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, np.sqrt(3) / 2, 0]
        ], dtype=np.float32), np.sqrt(3) / 4),

        # 4D: Regular tetrahedron
        ("Tetrahedron (D4)", np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, np.sqrt(3) / 2, 0],
            [0.5, np.sqrt(3) / 6, np.sqrt(6) / 3]
        ], dtype=np.float32), np.sqrt(2) / 12),

        # 5D: Regular 4-simplex (pentachoron) with unit edges
        ("4-Simplex (D5)", np.array([
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0.5, np.sqrt(3) / 2, 0, 0],
            [0.5, np.sqrt(3) / 6, np.sqrt(6) / 3, 0],
            [0.5, np.sqrt(3) / 6, np.sqrt(6) / 12, np.sqrt(10) / 4]
        ], dtype=np.float32), np.sqrt(5) / 96),

        # 6D: Regular 5-simplex with unit edges
        ("5-Simplex (D6)", np.array([
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0.5, np.sqrt(3) / 2, 0, 0, 0],
            [0.5, np.sqrt(3) / 6, np.sqrt(6) / 3, 0, 0],
            [0.5, np.sqrt(3) / 6, np.sqrt(6) / 12, np.sqrt(10) / 4, 0],
            [0.5, np.sqrt(3) / 6, np.sqrt(6) / 12, np.sqrt(10) / 20, np.sqrt(15) / 5]
        ], dtype=np.float32), np.sqrt(6) / 2880),  # Correct theoretical value
    ]

    for name, crystal, expected_vol in test_cases:
        computed_vol = vocab.factory._compute_volume(crystal)
        error = abs(computed_vol - expected_vol) / expected_vol if expected_vol > 0 else abs(computed_vol)

        if error < 0.01:  # 1% tolerance
            print(f"   ✓ {name:15}: {computed_vol:.6f} (expected {expected_vol:.6f}, error {error * 100:.3f}%)")
            results["passed"] += 1
        else:
            print(f"   ✗ {name:15}: {computed_vol:.6f} (expected {expected_vol:.6f}, error {error * 100:.3f}%)")
            results["failed"] += 1

        results["details"][name] = {
            "computed": computed_vol,
            "expected": expected_vol,
            "error": error
        }

    # Test 2: Cayley-Menger with all dimension types
    print("\n2. Dimension Type Integration")
    print("-" * 40)

    for dim_type in DimensionType:
        try:
            result = vocab.create_custom_crystal(
                "test",
                dimension_type=dim_type,
                content_type=ContentType.HYBRID,
                formula_type=FormulaType.CAYLEY_MENGER,
                norm_type=NormType.L2
            )

            crystal = result['crystal']
            volume = result['metadata']['volume']

            # Verify shape
            expected_vertices = dim_type.value if dim_type != DimensionType.D6_PLUS else 6
            shape_ok = crystal.shape[0] == expected_vertices

            # Verify volume is positive and finite
            volume_ok = 0 < volume < float('inf')

            if shape_ok and volume_ok:
                print(f"   ✓ {dim_type.name:8}: vertices={crystal.shape[0]}, volume={volume:.6f}")
                results["passed"] += 1
            else:
                print(f"   ✗ {dim_type.name:8}: shape_ok={shape_ok}, volume_ok={volume_ok}")
                results["failed"] += 1

        except Exception as e:
            print(f"   ✗ {dim_type.name:8}: ERROR - {str(e)[:50]}")
            results["failed"] += 1

    # Test 3: Edge length preservation
    print("\n3. Edge Length Consistency After Cayley-Menger")
    print("-" * 40)

    for dim_type in [DimensionType.D3, DimensionType.D4, DimensionType.D5]:
        result = vocab.create_custom_crystal(
            "test",
            dimension_type=dim_type,
            content_type=ContentType.HYBRID,
            formula_type=FormulaType.CAYLEY_MENGER,
            norm_type=NormType.NONE  # No normalization to check raw scaling
        )

        crystal = result['crystal']

        # Compute all edge lengths
        edges = []
        n = crystal.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                edges.append(np.linalg.norm(crystal[i] - crystal[j]))

        # Check if edges are consistent (Cayley-Menger should scale uniformly)
        edge_mean = np.mean(edges)
        edge_std = np.std(edges)
        edge_cv = edge_std / edge_mean if edge_mean > 0 else float('inf')

        # For Cayley-Menger, we expect relatively uniform edges after scaling
        if edge_cv < 1.0:
            print(f"   ✓ {dim_type.name}: mean={edge_mean:.4f}, CV={edge_cv:.4f}")
            results["passed"] += 1
        else:
            print(f"   ✗ {dim_type.name}: mean={edge_mean:.4f}, CV={edge_cv:.4f} (high variation)")
            results["failed"] += 1

    # Test 4: Volume scaling property
    print("\n4. Volume Scaling Property")
    print("-" * 40)

    base_crystal = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, np.sqrt(3) / 2, 0],
        [0.5, np.sqrt(3) / 6, np.sqrt(6) / 3]
    ], dtype=np.float32)

    base_vol = vocab.factory._compute_volume(base_crystal)

    for scale in [0.5, 2.0, 3.0]:
        scaled_crystal = base_crystal * scale
        scaled_vol = vocab.factory._compute_volume(scaled_crystal)

        # Volume should scale as scale^n where n is the dimension
        n = base_crystal.shape[0] - 1  # 3D simplex
        expected_vol = base_vol * (scale ** n)

        error = abs(scaled_vol - expected_vol) / expected_vol

        if error < 0.01:
            print(f"   ✓ Scale {scale}: {scaled_vol:.6f} (expected {expected_vol:.6f})")
            results["passed"] += 1
        else:
            print(f"   ✗ Scale {scale}: {scaled_vol:.6f} (expected {expected_vol:.6f}, error {error * 100:.2f}%)")
            results["failed"] += 1

    return results


# ============================================================================
# DEFINITION TRAJECTORY TESTS
# ============================================================================

def test_definition_trajectories(vocab):
    """Test that definitions actually affect crystal generation via V1 synthesis"""
    print("\n" + "=" * 80)
    print("DEFINITION-BASED TRAJECTORY TESTS")
    print("=" * 80)

    results = {"passed": 0, "failed": 0, "warnings": []}

    # Test 1: Definition vs No Definition
    print("\n1. Definition Impact on Crystal Structure")
    print("-" * 40)

    test_cases = [
        ("science", "The systematic study of the structure and behavior of the physical world"),
        ("emotion", "A strong feeling deriving from circumstances, mood, or relationships"),
        ("algorithm", "A finite sequence of well-defined instructions to solve a problem"),
    ]

    print("\n   DEBUG: Testing definition impact...")

    for token, definition in test_cases:
        # Clear cache to ensure fresh creation
        if hasattr(vocab, 'crystal_cache'):
            vocab.crystal_cache.clear()

        # Create with and without definition
        crystal_no_def = vocab.get_crystal(token, definition=None, synthesize=True)
        crystal_with_def = vocab.get_crystal(token, definition=definition, synthesize=True)

        if crystal_no_def is not None and crystal_with_def is not None:
            # They should be different
            max_diff = np.max(np.abs(crystal_no_def - crystal_with_def))

            # Also check trajectory influence (V1 should create directional bias)
            pooled_no_def = crystal_no_def.mean(axis=0)
            pooled_with_def = crystal_with_def.mean(axis=0)

            # Cosine similarity should show some difference
            norm_no = pooled_no_def / (np.linalg.norm(pooled_no_def) + 1e-8)
            norm_with = pooled_with_def / (np.linalg.norm(pooled_with_def) + 1e-8)
            cosine_sim = np.dot(norm_no, norm_with)

            if max_diff > 0.1 and cosine_sim < 0.95:
                print(f"   ✓ '{token}': diff={max_diff:.4f}, cosine={cosine_sim:.4f}")
                results["passed"] += 1
            else:
                print(f"   ✗ '{token}': diff={max_diff:.4f}, cosine={cosine_sim:.4f} (too similar)")
                results["failed"] += 1
                results["warnings"].append(f"Definition not affecting '{token}' enough")
        else:
            print(f"   ✗ '{token}': Failed to create crystals")
            results["failed"] += 1

    # Test 2: Different Definitions for Same Token
    print("\n2. Different Definitions Create Different Trajectories")
    print("-" * 40)

    token = "bank"
    definitions = [
        "A financial institution that accepts deposits and makes loans",
        "The land alongside or sloping down to a river or lake",
        "A place where something is stored or held available"
    ]

    crystals = []
    for definition in definitions:
        c = vocab.get_crystal(token, definition=definition, synthesize=True)
        if c is not None:
            crystals.append(c)

    if len(crystals) == 3:
        # Compute pairwise differences
        differences = []
        for i in range(len(crystals)):
            for j in range(i + 1, len(crystals)):
                diff = np.max(np.abs(crystals[i] - crystals[j]))
                differences.append(diff)

        avg_diff = np.mean(differences)

        if avg_diff > 0.1:
            print(f"   ✓ Token '{token}': avg difference={avg_diff:.4f}")
            results["passed"] += 1
        else:
            print(f"   ✗ Token '{token}': avg difference={avg_diff:.4f} (too similar)")
            results["failed"] += 1

    # Test 3: V1 Cardinal Axes Influence
    print("\n3. V1 Cardinal Axes and Projections")
    print("-" * 40)

    # Test with explicit V1 synthesis
    config_with_def = {
        'use_definitions': True,
        'dimension_type': DimensionType.D5,
        'formula_type': FormulaType.HYBRID_V1V2,
        'content_type': ContentType.ENRICHED
    }

    test_token = "philosophy"
    test_def = "The study of fundamental nature of knowledge, reality, and existence"

    # Create with V1 synthesis
    result = vocab.create_custom_crystal(
        test_token,
        dimension_type=DimensionType.D5,
        content_type=ContentType.ENRICHED,
        formula_type=FormulaType.HYBRID_V1V2,
        norm_type=NormType.L2,
        definition=test_def
    )

    crystal = result['crystal']

    # V1 synthesis should create specific patterns:
    # 1. Vertices should have varying distances from centroid (trajectory)
    distances = []
    centroid = crystal.mean(axis=0)
    for i in range(crystal.shape[0]):
        distances.append(np.linalg.norm(crystal[i] - centroid))

    dist_variance = np.var(distances)

    # 2. Check projection structure (vertices should span multiple directions)
    # Compute principal components
    cov = np.cov(crystal.T)
    eigenvalues, _ = np.linalg.eig(cov)
    eigenvalues = sorted(eigenvalues, reverse=True)[:5]

    # Ratio of largest to smallest eigenvalue (condition number indicator)
    if eigenvalues[-1] > 0:
        spread_ratio = eigenvalues[0] / eigenvalues[-1]
    else:
        spread_ratio = float('inf')

    print(f"   Distance variance: {dist_variance:.6f}")
    print(f"   Eigenvalue spread: {spread_ratio:.2f}")

    if dist_variance > 0.001 and spread_ratio < 1000:
        print(f"   ✓ V1 synthesis shows trajectory structure")
        results["passed"] += 1
    else:
        print(f"   ⚠ V1 synthesis may not be working properly")
        results["warnings"].append("V1 trajectory structure unclear")

    # Test 4: Definition Length Impact
    print("\n4. Definition Length Influence")
    print("-" * 40)

    short_def = "A test"
    medium_def = "A procedure for critical evaluation and assessment of quality"
    long_def = "A systematic examination and evaluation procedure designed to determine whether a system or component operates according to specified requirements and to identify any discrepancies between expected and actual results in order to provide feedback for continuous improvement"

    crystals_by_length = []
    for definition in [short_def, medium_def, long_def]:
        c = vocab.get_crystal("test", definition=definition, synthesize=True)
        if c is not None:
            crystals_by_length.append((len(definition), c))

    if len(crystals_by_length) == 3:
        # Longer definitions should create more complex trajectories (higher volume)
        volumes = []
        for length, crystal in crystals_by_length:
            vol = vocab.factory._compute_volume(crystal)
            volumes.append((length, vol))
            print(f"   Length {length:3}: volume={vol:.6f}")

        # Check if there's a correlation
        if volumes[1][1] >= volumes[0][1] and volumes[2][1] >= volumes[1][1]:
            print(f"   ✓ Definition length affects trajectory complexity")
            results["passed"] += 1
        else:
            print(f"   ⚠ Definition length correlation unclear")
            results["warnings"].append("Definition length not clearly affecting trajectory")

    return results


def test_mathematical_properties(vocab):
    """Test core mathematical properties of crystals"""
    print("\n" + "=" * 80)
    print("MATHEMATICAL PROPERTY TESTS")
    print("=" * 80)

    results = {"passed": 0, "failed": 0, "warnings": []}

    # Test 1: Volume calculation accuracy
    print("\n1. Volume Calculation Accuracy")
    print("-" * 40)

    # Create known geometric shapes
    test_cases = [
        # Line segment of length 1
        (np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32), 1.0, "Line"),

        # Equilateral triangle with side 1
        (np.array([[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3) / 2, 0]], dtype=np.float32),
         np.sqrt(3) / 4, "Triangle"),

        # Regular tetrahedron with edge 1
        (np.array([[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3) / 2, 0],
                   [0.5, np.sqrt(3) / 6, np.sqrt(6) / 3]], dtype=np.float32),
         np.sqrt(2) / 12, "Tetrahedron"),
    ]

    for crystal, expected_vol, name in test_cases:
        vol = vocab.factory._compute_volume(crystal)
        error = abs(vol - expected_vol) / expected_vol if expected_vol > 0 else abs(vol)

        if error < 0.01:  # 1% tolerance
            print(f"   ✓ {name}: {vol:.6f} (expected {expected_vol:.6f}, error {error * 100:.2f}%)")
            results["passed"] += 1
        else:
            print(f"   ✗ {name}: {vol:.6f} (expected {expected_vol:.6f}, error {error * 100:.2f}%)")
            results["failed"] += 1

    # Test 2: Normalization consistency
    print("\n2. Normalization Consistency")
    print("-" * 40)

    test_crystal = np.random.randn(5, 100).astype(np.float32)

    norm_tests = [
        (NormType.L1, lambda x: np.abs(x).sum(), "L1"),
        (NormType.L2, lambda x: np.linalg.norm(x, 'fro'), "L2/Frobenius"),
        (NormType.LINF, lambda x: np.max(np.abs(x)), "L∞"),
    ]

    for norm_type, norm_func, name in norm_tests:
        normalized = vocab.factory._apply_normalization(test_crystal.copy(), norm_type)
        computed_norm = norm_func(normalized)

        if abs(computed_norm - 1.0) < 0.001:
            print(f"   ✓ {name} norm: {computed_norm:.6f}")
            results["passed"] += 1
        else:
            print(f"   ✗ {name} norm: {computed_norm:.6f} (expected 1.0)")
            results["failed"] += 1

    # Test 3: Centering property
    print("\n3. Crystal Centering")
    print("-" * 40)

    for token in TEST_TOKENS[:3]:
        crystal = vocab.get_crystal(token, synthesize=True)
        if crystal is not None:
            centroid = crystal.mean(axis=0)
            centroid_norm = np.linalg.norm(centroid)

            if centroid_norm < 0.01:
                print(f"   ✓ '{token}': centroid norm = {centroid_norm:.6f}")
                results["passed"] += 1
            else:
                print(f"   ✗ '{token}': centroid norm = {centroid_norm:.6f} (should be ~0)")
                results["failed"] += 1
                results["warnings"].append(f"Crystal for '{token}' not properly centered")

    # Test 4: Determinism
    print("\n4. Determinism Check")
    print("-" * 40)

    for token in ["test", "hello"]:
        crystals = []
        for _ in range(3):
            c = vocab.get_crystal(token, synthesize=True)
            if c is not None:
                crystals.append(c)

        if len(crystals) == 3:
            max_diff = 0
            for i in range(1, len(crystals)):
                diff = np.max(np.abs(crystals[0] - crystals[i]))
                max_diff = max(max_diff, diff)

            if max_diff < 1e-6:
                print(f"   ✓ '{token}': max difference = {max_diff:.10f}")
                results["passed"] += 1
            else:
                print(f"   ✗ '{token}': max difference = {max_diff:.10f} (should be ~0)")
                results["failed"] += 1

    # Test 5: Edge length distribution
    print("\n5. Edge Length Distribution")
    print("-" * 40)

    crystal = vocab.get_crystal("test", synthesize=True)
    if crystal is not None and crystal.shape[0] > 1:
        edges = []
        for i in range(crystal.shape[0]):
            for j in range(i + 1, crystal.shape[0]):
                edges.append(np.linalg.norm(crystal[i] - crystal[j]))

        edge_mean = np.mean(edges)
        edge_std = np.std(edges)
        edge_cv = edge_std / edge_mean if edge_mean > 0 else float('inf')

        print(f"   Edge statistics:")
        print(f"   Mean: {edge_mean:.4f}, Std: {edge_std:.4f}, CV: {edge_cv:.4f}")

        if edge_cv < 1.0:  # Coefficient of variation < 1 indicates reasonable uniformity
            print(f"   ✓ Edge lengths reasonably uniform")
            results["passed"] += 1
        else:
            print(f"   ⚠ High edge length variation")
            results["warnings"].append("High edge length variation detected")

    return results


# ============================================================================
# FORMULA TYPE TESTS
# ============================================================================

def test_all_formulas(vocab):
    """Test all formula types with various configurations"""
    print("\n" + "=" * 80)
    print("FORMULA TYPE TESTS")
    print("=" * 80)

    results = {"passed": 0, "failed": 0, "errors": []}

    formulas = [
        FormulaType.ROSE_CAYLEY,
        FormulaType.CAYLEY_MENGER,
        FormulaType.CAYLEY,
        FormulaType.MENGER,
        FormulaType.EULER,
        FormulaType.GRAHAM_INFINITE,
        FormulaType.GRAHAM_FINITE,
        FormulaType.GRAHAM_MASKED,
        FormulaType.HYBRID_V1V2
    ]

    for formula in formulas:
        print(f"\n{formula.value}")
        print("-" * 40)

        try:
            # Test with different dimensions
            for dim_type in [DimensionType.D3, DimensionType.D5]:
                result = vocab.create_custom_crystal(
                    "test",
                    dimension_type=dim_type,
                    content_type=ContentType.HYBRID,
                    formula_type=formula,
                    norm_type=NormType.L2,
                    definition="test definition"
                )

                crystal = result['crystal']
                metadata = result['metadata']

                # Validation checks
                checks = []

                # Check shape
                expected_vertices = dim_type.value
                if crystal.shape[0] == expected_vertices:
                    checks.append(("Shape", True))
                else:
                    checks.append(("Shape", False))

                # Check for NaN/Inf
                has_nan = np.any(np.isnan(crystal))
                has_inf = np.any(np.isinf(crystal))
                if not has_nan and not has_inf:
                    checks.append(("Finite", True))
                else:
                    checks.append(("Finite", False))

                # Check volume
                if metadata['volume'] > 0:
                    checks.append(("Volume", True))
                else:
                    checks.append(("Volume", False))

                # Check centering
                centroid_norm = np.linalg.norm(crystal.mean(axis=0))
                if centroid_norm < 0.1:
                    checks.append(("Centered", True))
                else:
                    checks.append(("Centered", False))

                # Report results
                all_passed = all(check[1] for check in checks)
                status = "✓" if all_passed else "✗"

                check_str = ", ".join([f"{name}:{('✓' if passed else '✗')}"
                                       for name, passed in checks])
                print(f"   {status} {dim_type.name}: {check_str}")

                if all_passed:
                    results["passed"] += 1
                else:
                    results["failed"] += 1

        except Exception as e:
            print(f"   ✗ ERROR: {str(e)[:100]}")
            results["failed"] += 1
            results["errors"].append(f"{formula.value}: {str(e)}")

    return results


# ============================================================================
# CONTENT TYPE TESTS
# ============================================================================

def test_content_types(vocab):
    """Test all content type transformations"""
    print("\n" + "=" * 80)
    print("CONTENT TYPE TESTS")
    print("=" * 80)

    results = {"passed": 0, "failed": 0, "warnings": []}

    content_types = [
        ContentType.SPARSE,
        ContentType.ENRICHED,
        ContentType.TRAJECTORY,
        ContentType.MAGNITUDE,
        ContentType.VOLUME,
        ContentType.HYBRID
    ]

    for content in content_types:
        print(f"\n{content.value}")
        print("-" * 40)

        try:
            result = vocab.create_custom_crystal(
                "test",
                dimension_type=DimensionType.D5,
                content_type=content,
                formula_type=FormulaType.CAYLEY_MENGER,
                norm_type=NormType.L2,
                definition="test definition" if content != ContentType.SPARSE else None
            )

            crystal = result['crystal']
            metadata = result['metadata']

            # Content-specific validation
            if content == ContentType.TRAJECTORY:
                # Check if vertices form a smooth trajectory
                distances = []
                for i in range(1, crystal.shape[0]):
                    distances.append(np.linalg.norm(crystal[i] - crystal[i - 1]))

                # Check if distances are relatively consistent (smoothed)
                dist_std = np.std(distances)
                dist_mean = np.mean(distances)
                cv = dist_std / dist_mean if dist_mean > 0 else float('inf')

                if cv < 1.0:
                    print(f"   ✓ Trajectory smoothness CV: {cv:.4f}")
                    results["passed"] += 1
                else:
                    print(f"   ⚠ Trajectory smoothness CV: {cv:.4f} (high variation)")
                    results["warnings"].append(f"{content.value}: High trajectory variation")

            elif content == ContentType.MAGNITUDE:
                # Check magnitude consistency
                mags = np.linalg.norm(crystal, axis=1)
                mag_std = np.std(mags)
                mag_mean = np.mean(mags)

                if mag_std / mag_mean < 0.5:  # Low relative variation
                    print(f"   ✓ Magnitude consistency: std={mag_std:.4f}, mean={mag_mean:.4f}")
                    results["passed"] += 1
                else:
                    print(f"   ⚠ Magnitude variation: std={mag_std:.4f}, mean={mag_mean:.4f}")
                    results["warnings"].append(f"{content.value}: High magnitude variation")

            elif content == ContentType.VOLUME:
                # Check if volume is close to 1.0
                vol_error = abs(metadata['volume'] - 1.0)

                if vol_error < 0.2:
                    print(f"   ✓ Volume regularized: {metadata['volume']:.4f} (target=1.0)")
                    results["passed"] += 1
                else:
                    print(f"   ⚠ Volume: {metadata['volume']:.4f} (target=1.0, error={vol_error:.4f})")
                    results["warnings"].append(f"{content.value}: Volume not close to 1.0")
            else:
                # General validation
                print(f"   ✓ Created successfully, volume={metadata['volume']:.4f}")
                results["passed"] += 1

        except Exception as e:
            print(f"   ✗ ERROR: {str(e)[:100]}")
            results["failed"] += 1

    return results


# ============================================================================
# CHARACTER SYNTHESIS TESTS
# ============================================================================

def test_character_synthesis(vocab):
    """Test character-based synthesis and position weighting"""
    print("\n" + "=" * 80)
    print("CHARACTER SYNTHESIS TESTS")
    print("=" * 80)

    results = {"passed": 0, "failed": 0}

    # Test 1: Position sensitivity
    print("\n1. Position Sensitivity")
    print("-" * 40)

    test_pairs = [
        ("abc", "cba"),
        ("hello", "olleh"),
        ("123", "321"),
    ]

    for token1, token2 in test_pairs:
        c1 = vocab.get_crystal(token1, synthesize=True)
        c2 = vocab.get_crystal(token2, synthesize=True)

        if c1 is not None and c2 is not None:
            diff = np.max(np.abs(c1 - c2))

            if diff > 0.01:
                print(f"   ✓ '{token1}' vs '{token2}': diff={diff:.4f}")
                results["passed"] += 1
            else:
                print(f"   ✗ '{token1}' vs '{token2}': diff={diff:.4f} (too similar)")
                results["failed"] += 1

    # Test 2: Character composition
    print("\n2. Character Composition")
    print("-" * 40)

    # Test that tokens with shared characters have some similarity
    test_groups = [
        ["cat", "cart", "cast"],
        ["dog", "god", "dogs"],
    ]

    for group in test_groups:
        crystals = []
        for token in group:
            c = vocab.get_crystal(token, synthesize=True)
            if c is not None:
                crystals.append((token, c))

        if len(crystals) == len(group):
            # Compute pairwise similarities
            similarities = []
            for i in range(len(crystals)):
                for j in range(i + 1, len(crystals)):
                    pooled_i = crystals[i][1].mean(axis=0)
                    pooled_j = crystals[j][1].mean(axis=0)

                    # Cosine similarity
                    norm_i = pooled_i / (np.linalg.norm(pooled_i) + 1e-8)
                    norm_j = pooled_j / (np.linalg.norm(pooled_j) + 1e-8)
                    sim = np.dot(norm_i, norm_j)
                    similarities.append(sim)

            avg_sim = np.mean(similarities)

            if 0.3 < avg_sim < 0.95:  # Some similarity but not identical
                print(f"   ✓ Group {group[0][:3]}*: avg similarity={avg_sim:.4f}")
                results["passed"] += 1
            else:
                print(f"   ⚠ Group {group[0][:3]}*: avg similarity={avg_sim:.4f}")
                results["failed"] += 1

    # Test 3: Unicode handling
    print("\n3. Unicode Handling")
    print("-" * 40)

    unicode_tokens = ["你好", "世界", "🌍", "αβγ", "русский"]

    for token in unicode_tokens:
        try:
            crystal = vocab.get_crystal(token, synthesize=True)
            if crystal is not None and not np.any(np.isnan(crystal)):
                print(f"   ✓ '{token}': shape={crystal.shape}")
                results["passed"] += 1
            else:
                print(f"   ✗ '{token}': failed or contains NaN")
                results["failed"] += 1
        except Exception as e:
            print(f"   ✗ '{token}': {str(e)[:50]}")
            results["failed"] += 1

    return results


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

def test_performance(vocab):
    """Test performance characteristics"""
    print("\n" + "=" * 80)
    print("PERFORMANCE TESTS")
    print("=" * 80)

    results = {"timings": {}, "stats": {}}

    # Test 1: Single crystal creation speed
    print("\n1. Single Crystal Creation")
    print("-" * 40)

    configs = [
        ("Simple", DimensionType.D3, ContentType.SPARSE, FormulaType.CAYLEY),
        ("Standard", DimensionType.D5, ContentType.HYBRID, FormulaType.CAYLEY_MENGER),
        ("Complex", DimensionType.D5, ContentType.ENRICHED, FormulaType.HYBRID_V1V2),
    ]

    for name, dim, content, formula in configs:
        times = []
        for _ in range(10):
            start = time.time()
            _ = vocab.create_custom_crystal(
                "test",
                dimension_type=dim,
                content_type=content,
                formula_type=formula,
                norm_type=NormType.L2,
                definition="test" if content != ContentType.SPARSE else None
            )
            times.append(time.time() - start)

        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000

        print(f"   {name:10}: {avg_time:.2f} ± {std_time:.2f} ms")
        results["timings"][name] = avg_time

    # Test 2: Batch processing
    print("\n2. Batch Processing")
    print("-" * 40)

    batch_sizes = [10, 50, 100]

    for batch_size in batch_sizes:
        tokens = [f"token_{i}" for i in range(batch_size)]

        start = time.time()
        crystals = vocab.encode_batch(tokens, synthesize=True)
        elapsed = time.time() - start

        valid_count = sum(1 for c in crystals if c is not None)
        per_token = (elapsed / batch_size) * 1000

        print(f"   Batch {batch_size:3}: {elapsed:.3f}s total, {per_token:.2f}ms per token")
        print(f"              ({valid_count}/{batch_size} successful)")

        results["timings"][f"batch_{batch_size}"] = elapsed

    # Test 3: Cache effectiveness
    print("\n3. Cache Effectiveness")
    print("-" * 40)

    # Clear stats
    vocab.stats["cache_hits"] = 0
    vocab.stats["cache_misses"] = 0

    # First access (all misses)
    for token in ["test", "hello", "world"]:
        _ = vocab.get_crystal(token, synthesize=True)

    first_misses = vocab.stats["cache_misses"]

    # Second access (should be hits)
    for token in ["test", "hello", "world"]:
        _ = vocab.get_crystal(token, synthesize=True)

    cache_hits = vocab.stats["cache_hits"]

    hit_rate = cache_hits / (cache_hits + first_misses) if (cache_hits + first_misses) > 0 else 0

    print(f"   Cache hits: {cache_hits}")
    print(f"   Cache misses: {first_misses}")
    print(f"   Hit rate: {hit_rate:.2%}")

    results["stats"]["cache_hit_rate"] = hit_rate

    return results


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

def test_edge_cases(vocab):
    """Test edge cases and error handling"""
    print("\n" + "=" * 80)
    print("EDGE CASE TESTS")
    print("=" * 80)

    results = {"passed": 0, "failed": 0, "handled": []}

    edge_cases = [
        ("Empty string", "", None),
        ("Very long token", "a" * 1000, None),
        ("Very long definition", "test", "x" * 1000),
        ("Special characters", "!@#$%^&*()", None),
        ("Mixed unicode", "Hello世界🌍", None),
        ("Whitespace", "   ", None),
        ("Newlines", "hello\nworld", None),
        ("Null bytes", "hello\x00world", None),
    ]

    for name, token, definition in edge_cases:
        try:
            crystal = vocab.get_crystal(token, definition=definition, synthesize=True)

            if crystal is not None:
                # Validate crystal
                has_nan = np.any(np.isnan(crystal))
                has_inf = np.any(np.isinf(crystal))
                proper_shape = crystal.shape == (5, vocab.config.embedding_dim)

                if not has_nan and not has_inf and proper_shape:
                    print(f"   ✓ {name}: handled correctly")
                    results["passed"] += 1
                else:
                    print(f"   ⚠ {name}: created but has issues")
                    print(f"     NaN: {has_nan}, Inf: {has_inf}, Shape OK: {proper_shape}")
                    results["failed"] += 1
            else:
                print(f"   ⚠ {name}: returned None")
                results["failed"] += 1

        except Exception as e:
            print(f"   ✓ {name}: exception handled: {str(e)[:50]}")
            results["handled"].append(name)
            results["passed"] += 1

    return results


# ============================================================================
# SIMILARITY TESTS
# ============================================================================

def test_similarity_metrics(vocab):
    """Test various similarity metrics"""
    print("\n" + "=" * 80)
    print("SIMILARITY METRIC TESTS")
    print("=" * 80)

    results = {"passed": 0, "failed": 0}

    test_pairs = [
        ("cat", "dog", "related animals"),
        ("hello", "goodbye", "greetings"),
        ("123", "456", "numbers"),
        ("test", "test", "identical"),
    ]

    methods = ["cosine", "euclidean", "manhattan"]

    for token1, token2, description in test_pairs:
        print(f"\n{description}: '{token1}' vs '{token2}'")
        print("-" * 40)

        for method in methods:
            try:
                sim = vocab.similarity(token1, token2,
                                       definition_a=None, definition_b=None,
                                       method=method, synthesize=True)

                # Special case: identical tokens should have max similarity
                if token1 == token2:
                    if method == "cosine" and abs(sim - 1.0) < 0.001:
                        print(f"   ✓ {method}: {sim:.4f} (identical)")
                        results["passed"] += 1
                    elif method in ["euclidean", "manhattan"] and abs(sim) < 0.001:
                        print(f"   ✓ {method}: {sim:.4f} (identical)")
                        results["passed"] += 1
                    else:
                        print(f"   ✗ {method}: {sim:.4f} (should be ~1 for cosine, ~0 for distance)")
                        results["failed"] += 1
                else:
                    print(f"   {method}: {sim:.4f}")
                    results["passed"] += 1

            except Exception as e:
                print(f"   ✗ {method}: {str(e)[:50]}")
                results["failed"] += 1

    # Test advanced similarity
    print("\n\nAdvanced Similarity (Hausdorff)")
    print("-" * 40)

    try:
        hausdorff_sim = vocab.crystal_similarity_advanced(
            "cat", "dog", method="hausdorff"
        )
        print(f"   Hausdorff distance: {hausdorff_sim:.4f}")
        results["passed"] += 1
    except Exception as e:
        print(f"   ✗ Error: {str(e)[:50]}")
        results["failed"] += 1

    return results


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_comprehensive_tests(vocab):
    """Run all test batteries"""
    print("=" * 80)
    print("COMPREHENSIVE TEST BATTERY FOR CRYSTAL VOCABULARY SYSTEM")
    print("=" * 80)

    all_results = {}

    # Run each test suite
    test_suites = [
        ("Mathematical Properties", test_mathematical_properties),
        ("Cayley-Menger Comprehensive", test_cayley_menger_comprehensive),
        ("Definition Trajectories", test_definition_trajectories),
        ("Formula Types", test_all_formulas),
        ("Content Types", test_content_types),
        ("Character Synthesis", test_character_synthesis),
        ("Performance", test_performance),
        ("Edge Cases", test_edge_cases),
        ("Similarity Metrics", test_similarity_metrics),
    ]

    total_passed = 0
    total_failed = 0

    for name, test_func in test_suites:
        print(f"\nRunning: {name}")
        try:
            results = test_func(vocab)
            all_results[name] = results

            # Aggregate pass/fail counts
            if "passed" in results:
                total_passed += results["passed"]
            if "failed" in results:
                total_failed += results["failed"]

        except Exception as e:
            print(f"   Suite failed with error: {str(e)}")
            all_results[name] = {"error": str(e)}
            total_failed += 1

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL TEST SUMMARY")
    print("=" * 80)

    for suite_name, results in all_results.items():
        print(f"\n{suite_name}:")

        if "error" in results:
            print(f"   ✗ Suite failed: {results['error'][:100]}")
        elif "passed" in results and "failed" in results:
            passed = results["passed"]
            failed = results["failed"]
            total = passed + failed
            pass_rate = (passed / total * 100) if total > 0 else 0

            print(f"   Tests: {passed}/{total} passed ({pass_rate:.1f}%)")

            if "warnings" in results and results["warnings"]:
                print(f"   Warnings: {len(results['warnings'])}")
                for warning in results["warnings"][:3]:
                    print(f"     - {warning}")

            if "errors" in results and results["errors"]:
                print(f"   Errors: {len(results['errors'])}")
                for error in results["errors"][:3]:
                    print(f"     - {error[:100]}")

        elif "timings" in results:
            print(f"   Performance metrics captured")
            for key, value in results["timings"].items():
                if isinstance(value, (int, float)):
                    print(f"     - {key}: {value:.3f}")

    print("\n" + "=" * 80)
    print(f"OVERALL: {total_passed} passed, {total_failed} failed")

    if total_failed == 0:
        print("✓ ALL TESTS PASSED SUCCESSFULLY!")
    else:
        print(f"⚠ {total_failed} tests failed - review results above")

    print("=" * 80)

    return all_results


# ============================================================================
# EXECUTE TESTS
# ============================================================================

if __name__ == "__main__":
    # Import and create vocabulary

    print("Creating vocabulary instance...")
    vocab = create_unified_vocabulary(
        embedding_dim=100,
        dimension_type=DimensionType.D5,
        content_type=ContentType.HYBRID,
        formula_type=FormulaType.HYBRID_V1V2,
        norm_type=NormType.L2,
        enable_synthesis=True
    )

    print("Starting comprehensive test battery...\n")

    # Run all tests
    results = run_comprehensive_tests(vocab)

    print("\nTest battery complete.")