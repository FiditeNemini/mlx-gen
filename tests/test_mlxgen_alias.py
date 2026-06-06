def test_mlxgen_aliases_mflux_module():
    import mflux
    import mlxgen

    assert mlxgen is mflux


def test_mlxgen_submodule_import_matches_mflux():
    from mlxgen.models.z_image import ZImageTurbo as MlxgenZImageTurbo

    from mflux.models.z_image import ZImageTurbo as MfluxZImageTurbo

    assert MlxgenZImageTurbo is MfluxZImageTurbo


def test_mlxgen_submodule_import_does_not_replace_mflux_parent_package():
    from mlxgen.models.z_image import ZImageTurbo as MlxgenZImageTurbo

    import mflux
    import mflux.models.common
    from mflux.models.z_image import ZImageTurbo as MfluxZImageTurbo

    assert MlxgenZImageTurbo is MfluxZImageTurbo
    assert mflux.models.__name__ == "mflux.models"
    assert hasattr(mflux.models, "common")
    assert mflux.models.common.__name__ == "mflux.models.common"
