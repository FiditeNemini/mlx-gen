"""Family dispatch for `mlxgen upscale`.

The load-bearing property is that adding SwiftVR changed nothing about SeedVR2: every
handle SeedVR2 accepted before must still resolve to the same catalog entry and the same
path, and the handle SeedVR2 rejected must still be rejected by SeedVR2's own resolver
with its own wording. These tests compare against `_resolve_seedvr2_model` directly
rather than against a copied expectation, so they cannot drift away from it.
"""

import pytest

from mflux.cli import mlx_gen
from mflux.models.common.cli.restore_dispatch import (
    classify_restore_family,
    peek_restore_family,
    resolve_restore_family,
)
from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.seedvr2.cli.seedvr2_upscale import _resolve_seedvr2_model

# Every handle form `_resolve_seedvr2_model` recognises, plus handles it recognises only
# by falling through to its substring heuristics.
SEEDVR2_HANDLES = [
    None,
    "seedvr2",
    "seedvr2-3b",
    "SeedVR2-3B",
    "seedvr2-7b",
    "seedvr2-7b-sharp",
    "ByteDance-Seed/SeedVR2-3B",
    "bytedance-seed/seedvr2_3b",
    "ByteDance-Seed/SeedVR2-7B",
    "bytedance-seed/seedvr2_7b",
    "numz/SeedVR2_comfyUI",
    "AbstractFramework/seedvr2-3b-8bit",
    "AbstractFramework/seedvr2-3b-4bit",
    "AbstractFramework/seedvr2-7b-8bit",
    "AbstractFramework/seedvr2-7b-4bit",
    "some-local-name-seedvr2-7b",
    "an-unrecognised-bare-string",
]

SWIFTVR_HANDLES = ["swiftvr", "SwiftVR", "SWIFTVR", "swiftvr-5b", "H-oliday/SwiftVR", "h-oliday/swiftvr"]


@pytest.mark.parametrize("handle", SEEDVR2_HANDLES)
def test_seedvr2_handles_still_classify_as_seedvr2(handle):
    assert classify_restore_family(handle, None) == "seedvr2"


@pytest.mark.parametrize("handle", SEEDVR2_HANDLES)
def test_seedvr2_handles_resolve_exactly_as_before(handle):
    expected_config, expected_path = _resolve_seedvr2_model(handle, None)
    route = resolve_restore_family(handle, None)
    assert route.family == "seedvr2"
    assert route.model_config is expected_config
    assert route.model_path == expected_path


@pytest.mark.parametrize("handle", SWIFTVR_HANDLES)
def test_swiftvr_handles_classify_as_swiftvr(handle):
    assert classify_restore_family(handle, None) == "swiftvr"
    route = resolve_restore_family(handle, None)
    assert route.family == "swiftvr"
    assert "swiftvr" in route.model_config.aliases


@pytest.mark.parametrize("handle", SEEDVR2_HANDLES + SWIFTVR_HANDLES)
@pytest.mark.parametrize("form", ["space", "equals", "short", "attached", "abbreviated"])
def test_argv_peek_agrees_with_post_parse_classification(handle, form):
    """A disagreement here dispatches to one family and loads the other's checkpoint.

    Every form argparse accepts must be read the same way by the pre-parse peek. The
    attached short form is the one a hand-rolled scanner gets wrong: argparse reads
    ``-mswiftvr`` as ``--model swiftvr``, and a token-equality scan sees an unknown flag,
    routes to SeedVR2, and the run dies telling the user to use ``mlxgen upscale`` - which
    is what they typed.
    """
    if handle is None:
        argv = []
    elif form == "space":
        argv = ["--model", handle]
    elif form == "equals":
        argv = [f"--model={handle}"]
    elif form == "short":
        argv = ["-m", handle]
    elif form == "attached":
        argv = [f"-m{handle}"]
    else:
        argv = ["--mod", handle]
    assert peek_restore_family([*argv, "--video-path", "clip.mp4"]) == classify_restore_family(handle, None)


@pytest.mark.parametrize("form", ["-mswiftvr", "--model=swiftvr", "--mod=swiftvr"])
def test_argparse_reads_the_same_handle_the_peek_reads(form):
    """The peek is only correct if it agrees with the parser the command actually uses."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m")
    parsed, _ = parser.parse_known_args([form, "--video-path", "clip.mp4"])
    assert parsed.model == "swiftvr"
    assert peek_restore_family([form, "--video-path", "clip.mp4"]) == "swiftvr"


def test_unparseable_argv_defers_to_the_family_parser_instead_of_guessing():
    """A dangling --model cannot be classified; SeedVR2's parser then reports the real
    usage error rather than this one inventing a message."""
    assert peek_restore_family(["--model"]) == "seedvr2"


def test_unknown_repo_id_keeps_the_seedvr2_sentence_and_names_swiftvr():
    with pytest.raises(ValueError) as exc:
        resolve_restore_family("acme/not-a-real-model", None)
    message = str(exc.value)
    assert message.startswith("Unsupported SeedVR2 model handle")
    assert "swiftvr" in message.lower()


def test_local_swiftvr_directory_is_detected(tmp_path):
    checkpoint = tmp_path / "swiftvr"
    (checkpoint / "transformer").mkdir(parents=True)
    (checkpoint / "reae.safetensors").touch()
    assert classify_restore_family(str(checkpoint), None) == "swiftvr"
    assert classify_restore_family(None, str(checkpoint)) == "swiftvr"
    assert peek_restore_family(["--path", str(checkpoint)]) == "swiftvr"


def test_local_seedvr2_directory_is_not_stolen_by_swiftvr(tmp_path):
    checkpoint = tmp_path / "seedvr2"
    checkpoint.mkdir()
    (checkpoint / "seedvr2_ema_3b.pth").touch()
    assert classify_restore_family(str(checkpoint), None) == "seedvr2"


def test_directory_claiming_both_families_fails_closed(tmp_path):
    checkpoint = tmp_path / "both"
    checkpoint.mkdir()
    (checkpoint / "reae.safetensors").touch()
    (checkpoint / "seedvr2_ema_3b.pth").touch()
    with pytest.raises(ValueError, match="cannot tell which one"):
        classify_restore_family(str(checkpoint), None)


def test_family_predicates_are_mutually_exclusive_across_the_catalog():
    """A handle matching two families silently reroutes prepare and download."""
    collisions = {}
    for key, model_config in AVAILABLE_MODELS.items():
        aliases = {alias.lower() for alias in model_config.aliases}
        model_key = mlx_gen._model_key(model_config.model_name, model_config.base_model)
        matched = [
            name
            for name, predicate in (
                ("swiftvr", mlx_gen._is_swiftvr),
                ("wan", mlx_gen._is_wan),
                ("seedvr2", mlx_gen._is_seedvr2),
            )
            if predicate(aliases, model_key)
        ]
        if len(matched) > 1:
            collisions[key] = matched
    assert collisions == {}


def test_swiftvr_download_fetches_only_its_four_files():
    """Without a SwiftVR branch the patterns are None and the whole repo is pulled."""
    model_config = AVAILABLE_MODELS["swiftvr"]
    patterns = mlx_gen._download_patterns(model_config, model_config.model_name)
    assert patterns == [
        "prompt_embedding.safetensors",
        "reae.safetensors",
        "transformer/*.safetensors",
        "transformer/config.json",
    ]
