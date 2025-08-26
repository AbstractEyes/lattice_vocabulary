def test_import():
    import geovocab
    assert hasattr(geovocab, "GeometricVocab")
    assert hasattr(geovocab, "PretrainedGeometricVocab")
