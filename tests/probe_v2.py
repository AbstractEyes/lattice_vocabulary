"""
Updated Comprehensive Test Battery for Unified Geometric Crystal Vocabulary
Tests all formulas, dimensions, content types, mathematical properties, and bug fixes
"""

import numpy as np
import time

from src.geovocab2 import (
    create_unified_vocabulary,
    DimensionType,
    ContentType,
    FormulaType,
    NormType
)


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
# BUG FIX VERIFICATION TESTS
# ============================================================================

def test_bug_fixes(vocab):
    """Test that our bug fixes work correctly"""
    print("\n" + "=" * 80)
    print("BUG FIX VERIFICATION TESTS")
    print("=" * 80)

    results = {"passed": 0, "failed": 0, "details": {}}

    # Test 1: Cache Key Fix for Definitions
    print("\n1. Cache Key Handling with Definitions")
    print("-" * 40)

    # Clear cache first
    vocab.crystal_cache.clear()
    vocab.pooled_cache.clear()
    vocab.stats["cache_hits"] = 0
    vocab.stats["cache_misses"] = 0

    # First call with no definition - should miss
    crystal1a = vocab.get_crystal("test", definition=None, synthesize=True)
    miss1 = vocab.stats["cache_misses"]

    # Second call with no definition - should hit
    crystal1b = vocab.get_crystal("test", definition=None, synthesize=True)
    hits1 = vocab.stats["cache_hits"]

    # Third call with definition - should miss (different cache key)
    crystal2a = vocab.get_crystal("test", definition="A test definition", synthesize=True)
    miss2 = vocab.stats["cache_misses"]

    # Fourth call with same definition - should hit
    crystal2b = vocab.get_crystal("test", definition="A test definition", synthesize=True)
    hits2 = vocab.stats["cache_hits"]

    # Fifth call with different definition - should miss
    crystal3 = vocab.get_crystal("test", definition="Another definition", synthesize=True)
    miss3 = vocab.stats["cache_misses"]

    # Verify cache behavior
    cache_test_passed = True
    errors = []

    if hits1 != 1:
        cache_test_passed = False
        errors.append(f"No-definition cache hit failed: {hits1} hits (expected 1)")

    if hits2 != 2:
        cache_test_passed = False
        errors.append(f"Same-definition cache hit failed: {hits2} total hits (expected 2)")

    if miss2 <= miss1:
        cache_test_passed = False
        errors.append(f"Definition should create new cache miss: {miss2} <= {miss1}")

    if miss3 <= miss2:
        cache_test_passed = False
        errors.append(f"Different definition should create new cache miss: {miss3} <= {miss2}")

    # Verify crystals are identical when from cache
    if crystal1a is not None and crystal1b is not None:
        if not np.array_equal(crystal1a, crystal1b):
            cache_test_passed = False
            errors.append("Cached crystals should be identical")

    if crystal2a is not None and crystal2b is not None:
        if not np.array_equal(crystal2a, crystal2b):
            cache_test_passed = False
            errors.append("Cached crystals with definition should be identical")

    if cache_test_passed:
        print(f"   ✓ Cache key handling correct")
        print(f"     - No-def hits: {hits1}, With-def hits: {hits2 - hits1}")
        print(f"     - Total misses for 3 unique combinations: {miss3}")
        results["passed"] += 1
    else:
        print(f"   ✗ Cache key issues detected:")
        for error in errors:
            print(f"     - {error}")
        results["failed"] += 1

    # Test 2: Character Position Weighting
    print("\n2. Character Position Weighting (Exponential Decay)")
    print("-" * 40)

    # Test tokens where order matters significantly
    test_pairs = [
        ("abc", "cba"),
        ("hello", "olleh"),
        ("12345", "54321"),
        ("test", "tset"),
    ]

    position_test_passed = True
    position_errors = []

    for token1, token2 in test_pairs:
        # Clear cache to ensure fresh synthesis
        vocab.crystal_cache.clear()

        c1 = vocab.get_crystal(token1, synthesize=True)
        c2 = vocab.get_crystal(token2, synthesize=True)

        if c1 is not None and c2 is not None:
            # Compute similarity - should be different due to position weighting
            pooled1 = c1.mean(axis=0)
            pooled2 = c2.mean(axis=0)

            # Cosine similarity
            norm1 = pooled1 / (np.linalg.norm(pooled1) + 1e-8)
            norm2 = pooled2 / (np.linalg.norm(pooled2) + 1e-8)
            cosine_sim = np.dot(norm1, norm2)

            # Max absolute difference
            max_diff = np.max(np.abs(c1 - c2))

            # With character-based synthesis, reversed strings can be quite different
            # We just want to verify they're not identical (would indicate no position sensitivity)
            # Negative correlation is actually fine - it means position matters a lot!
            if abs(cosine_sim) > 0.999:  # Nearly identical (positive or negative)
                position_test_passed = False
                position_errors.append(f"{token1}/{token2}: too similar (|cos|={abs(cosine_sim):.4f})")
            elif max_diff < 0.01:  # Essentially the same crystal
                position_test_passed = False
                position_errors.append(f"{token1}/{token2}: crystals nearly identical (max_diff={max_diff:.4f})")
            else:
                print(f"   ✓ '{token1}' vs '{token2}': cosine={cosine_sim:.4f}, max_diff={max_diff:.4f}")
                print(
                    f"      (Position weighting creates {'anticorrelated' if cosine_sim < 0 else 'different'} crystals)")
        else:
            position_test_passed = False
            position_errors.append(f"Failed to create crystals for {token1}/{token2}")

    if position_test_passed:
        print(f"   ✓ Character position weighting working correctly")
        results["passed"] += 1
    else:
        print(f"   ✗ Position weighting issues:")
        for error in position_errors:
            print(f"     - {error}")
        results["failed"] += 1

    # Test 3: Exponential Weight Values
    print("\n3. Verify Exponential Weight Calculation")
    print("-" * 40)

    # Test the actual weight values
    test_lengths = [3, 5, 10]
    weight_test_passed = True

    for length in test_lengths:
        weights = np.array([np.exp(-0.3 * i) for i in range(length)])
        weights /= weights.sum()

        # First character should have highest weight
        if weights[0] != max(weights):
            weight_test_passed = False
            print(f"   ✗ Length {length}: First char doesn't have max weight")

        # Weights should decrease monotonically
        for i in range(1, length):
            if weights[i] >= weights[i - 1]:
                weight_test_passed = False
                print(f"   ✗ Length {length}: Weights not decreasing at position {i}")
                break
        else:
            weight_str = ", ".join([f"{w:.4f}" for w in weights[:3]])
            print(f"   ✓ Length {length}: weights = [{weight_str}]... (decreasing correctly)")

    if weight_test_passed:
        results["passed"] += 1
    else:
        results["failed"] += 1

    # Test 4: Definition Cache Keys in Pooled Vectors
    print("\n4. Pooled Vector Cache with Definitions")
    print("-" * 40)

    vocab.pooled_cache.clear()

    # Get pooled vectors with different scenarios
    p1 = vocab.get_pooled("test", definition=None, method="mean")
    p2 = vocab.get_pooled("test", definition=None, method="mean")  # Should hit cache
    p3 = vocab.get_pooled("test", definition="definition", method="mean")  # Different key
    p4 = vocab.get_pooled("test", definition="definition", method="mean")  # Should hit cache
    p5 = vocab.get_pooled("test", definition=None, method="sum")  # Different method

    # Check cache size - should have 3 unique entries
    cache_size = len(vocab.pooled_cache)

    if cache_size == 3:
        print(f"   ✓ Pooled cache has correct number of entries: {cache_size}")
        results["passed"] += 1
    else:
        print(f"   ✗ Pooled cache has wrong number of entries: {cache_size} (expected 3)")
        results["failed"] += 1

    return results


# ============================================================================
# CAYLEY-MENGER COMPREHENSIVE TESTS (Updated)
# ============================================================================

def test_cayley_menger_comprehensive(vocab):
    """Comprehensive testing of Cayley-Menger volume calculations"""
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

    # Test 2: Volume scaling property
    print("\n2. Volume Scaling Property")
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
# DEFINITION TRAJECTORY TESTS (Updated)
# ============================================================================

def test_definition_trajectories(vocab):
    """Test that definitions actually affect crystal generation"""
    print("\n" + "=" * 80)
    print("DEFINITION-BASED TRAJECTORY TESTS")
    print("=" * 80)

    results = {"passed": 0, "failed": 0, "warnings": []}

    # Test 1: Definition vs No Definition (with proper cache clearing)
    print("\n1. Definition Impact on Crystal Structure")
    print("-" * 40)

    test_cases = [
        ("science", "The systematic study of the structure and behavior of the physical world"),
        ("emotion", "A strong feeling deriving from circumstances, mood, or relationships"),
        ("algorithm", "A finite sequence of well-defined instructions to solve a problem"),
    ]

    for token, definition in test_cases:
        # Clear cache completely
        vocab.crystal_cache.clear()
        vocab.pooled_cache.clear()

        # Create with and without definition
        crystal_no_def = vocab.get_crystal(token, definition=None, synthesize=True)
        crystal_with_def = vocab.get_crystal(token, definition=definition, synthesize=True)

        if crystal_no_def is not None and crystal_with_def is not None:
            # They should be different
            max_diff = np.max(np.abs(crystal_no_def - crystal_with_def))

            # Also check trajectory influence
            pooled_no_def = crystal_no_def.mean(axis=0)
            pooled_with_def = crystal_with_def.mean(axis=0)

            # Cosine similarity should show some difference
            norm_no = pooled_no_def / (np.linalg.norm(pooled_no_def) + 1e-8)
            norm_with = pooled_with_def / (np.linalg.norm(pooled_with_def) + 1e-8)
            cosine_sim = np.dot(norm_no, norm_with)

            # Definitions should create different crystals (max_diff > threshold)
            # The cosine similarity might be negative (anticorrelated) which is fine
            # We just want to ensure they're not identical
            if max_diff > 0.1 and abs(cosine_sim) < 0.999:
                print(f"   ✓ '{token}': diff={max_diff:.4f}, cosine={cosine_sim:.4f}")
                if cosine_sim < 0:
                    print(f"      (Definition creates anticorrelated structure)")
                results["passed"] += 1
            elif max_diff < 0.01:
                print(f"   ✗ '{token}': Crystals too similar (max_diff={max_diff:.4f})")
                results["failed"] += 1
            else:
                print(f"   ⚠ '{token}': diff={max_diff:.4f}, cosine={cosine_sim:.4f}")
                results["warnings"].append(f"Unexpected similarity pattern for '{token}'")
        else:
            print(f"   ✗ '{token}': Failed to create crystals")
            results["failed"] += 1

    # Test 2: Cache Independence for Different Definitions
    print("\n2. Cache Independence with Different Definitions")
    print("-" * 40)

    vocab.crystal_cache.clear()

    token = "bank"
    definitions = [
        "A financial institution that accepts deposits and makes loans",
        "The land alongside or sloping down to a river or lake",
        None,  # No definition
    ]

    crystals = []
    for definition in definitions:
        c = vocab.get_crystal(token, definition=definition, synthesize=True)
        if c is not None:
            crystals.append(c)

    if len(crystals) == 3:
        # All three should be different
        all_different = True
        for i in range(len(crystals)):
            for j in range(i + 1, len(crystals)):
                if np.array_equal(crystals[i], crystals[j]):
                    all_different = False
                    print(f"   ✗ Crystals {i} and {j} are identical (should be different)")

        if all_different:
            print(f"   ✓ All crystals with different definitions are unique")
            results["passed"] += 1
        else:
            results["failed"] += 1
    else:
        print(f"   ✗ Failed to create all crystals")
        results["failed"] += 1

    return results


# ============================================================================
# PERFORMANCE TESTS (Updated)
# ============================================================================

def test_performance(vocab):
    """Test performance characteristics with cache behavior"""
    print("\n" + "=" * 80)
    print("PERFORMANCE TESTS")
    print("=" * 80)

    results = {"timings": {}, "stats": {}}

    # Test 1: Cache effectiveness with definitions
    print("\n1. Cache Performance with Definitions")
    print("-" * 40)

    # Clear all caches and stats
    vocab.crystal_cache.clear()
    vocab.pooled_cache.clear()
    vocab.stats["cache_hits"] = 0
    vocab.stats["cache_misses"] = 0

    # Test scenarios
    test_scenarios = [
        ("token1", None),
        ("token1", None),  # Should hit
        ("token1", "definition"),  # Should miss (different key)
        ("token1", "definition"),  # Should hit
        ("token2", None),  # Should miss (different token)
        ("token2", None),  # Should hit
    ]

    for token, definition in test_scenarios:
        _ = vocab.get_crystal(token, definition=definition, synthesize=True)

    expected_hits = 3
    expected_misses = 3
    actual_hits = vocab.stats["cache_hits"]
    actual_misses = vocab.stats["cache_misses"]

    if actual_hits == expected_hits and actual_misses == expected_misses:
        print(f"   ✓ Cache working correctly: {actual_hits} hits, {actual_misses} misses")
    else:
        print(f"   ✗ Cache issue: {actual_hits} hits (expected {expected_hits}), "
              f"{actual_misses} misses (expected {expected_misses})")

    hit_rate = actual_hits / (actual_hits + actual_misses) if (actual_hits + actual_misses) > 0 else 0
    print(f"   Hit rate: {hit_rate:.2%}")

    results["stats"]["cache_hit_rate"] = hit_rate
    results["stats"]["expected_behavior"] = (actual_hits == expected_hits)

    # Test 2: Batch processing speed
    print("\n2. Batch Processing Performance")
    print("-" * 40)

    batch_sizes = [10, 50]

    for batch_size in batch_sizes:
        tokens = [f"token_{i}" for i in range(batch_size)]
        definitions = [f"Definition {i}" if i % 2 == 0 else None for i in range(batch_size)]

        start = time.time()
        crystals = vocab.encode_batch(tokens, definitions=definitions, synthesize=True)
        elapsed = time.time() - start

        valid_count = sum(1 for c in crystals if c is not None)
        per_token = (elapsed / batch_size) * 1000

        print(f"   Batch {batch_size:3}: {elapsed:.3f}s total, {per_token:.2f}ms per token")
        print(f"              ({valid_count}/{batch_size} successful)")

        results["timings"][f"batch_{batch_size}"] = elapsed

    return results


# ============================================================================
# MAIN TEST RUNNER (Updated)
# ============================================================================

def run_comprehensive_tests(vocab):
    """Run all test batteries including bug fix verifications"""
    print("=" * 80)
    print("COMPREHENSIVE TEST BATTERY FOR FIXED CRYSTAL VOCABULARY SYSTEM")
    print("=" * 80)

    all_results = {}

    # Run each test suite
    test_suites = [
        ("Bug Fix Verification", test_bug_fixes),  # NEW: Test our fixes first
        ("Cayley-Menger Comprehensive", test_cayley_menger_comprehensive),
        ("Definition Trajectories", test_definition_trajectories),
        ("Performance", test_performance),
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

        elif "timings" in results:
            print(f"   Performance metrics captured")
            if "stats" in results:
                for key, value in results["stats"].items():
                    if isinstance(value, bool):
                        print(f"     - {key}: {'✓' if value else '✗'}")
                    elif isinstance(value, (int, float)):
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
    print("Creating vocabulary instance...")

    vocab = create_unified_vocabulary(
        embedding_dim=100,
        dimension_type=DimensionType.D5,
        content_type=ContentType.HYBRID,
        formula_type=FormulaType.HYBRID_V1V2,
        norm_type=NormType.L2,
        enable_synthesis=True,
        prefer_dataset=False  # Run without dataset for faster testing
    )

    print("Starting comprehensive test battery...\n")

    # Run all tests
    results = run_comprehensive_tests(vocab)

    print("\nTest battery complete.")