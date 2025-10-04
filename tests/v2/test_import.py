def test_import():
    import geovocab2
    assert hasattr(geovocab, "GeometricVocab")
    assert hasattr(geovocab, "PretrainedGeometricVocab")


if __name__ == "__main__":
    test_import()
    print("Import test passed.")
#     x = torch.tensor([1.0, 2.0, 3.0])
#     is_valid, error_msg = formula.validate(x)
#     print(f"Validation: {is_valid}, {error_msg}")
#     assert is_valid, f"Validation failed: {error_msg}"
