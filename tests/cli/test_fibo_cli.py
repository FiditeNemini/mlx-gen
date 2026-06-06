from unittest.mock import patch

import pytest
from PIL import Image

from mflux.models.fibo.cli import fibo_edit


def test_fibo_edit_missing_prompt_is_parser_error(tmp_path, capsys):
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 64), (20, 30, 40)).save(source)

    with patch(
        "sys.argv",
        [
            "mflux-generate-fibo-edit",
            "--image-path",
            str(source),
        ],
    ):
        with pytest.raises(SystemExit):
            fibo_edit.main()

    assert "requires an edit instruction" in capsys.readouterr().err
