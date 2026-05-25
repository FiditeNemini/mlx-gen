def test_mlxgen_aliases_mflux_module():
    import mflux
    import mlxgen

    assert mlxgen is mflux


def test_mlxgen_submodule_import_matches_mflux():
    from mlxgen.models.z_image import ZImageTurbo as MlxgenZImageTurbo

    from mflux.models.z_image import ZImageTurbo as MfluxZImageTurbo

    assert MlxgenZImageTurbo is MfluxZImageTurbo
