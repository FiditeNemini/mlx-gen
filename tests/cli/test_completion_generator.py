import pytest

from mflux.cli.completions.generator import CompletionGenerator


@pytest.mark.fast
def test_completion_generator_includes_fibo_edit_command():
    generator = CompletionGenerator()

    assert "mflux-generate-fibo-edit" in generator.commands

    parser = generator.create_parser_for_command("mflux-generate-fibo-edit")
    script = generator.generate_command_function("mflux-generate-fibo-edit", parser)

    assert "_mflux_generate_fibo_edit()" in script
    assert "--image-path" in script
    assert "--mask-path" in script
    assert "--masked-image-path" in script
    assert "--matte-output" in script
    assert "--prompt" in script
    assert "--lora-paths" not in script


@pytest.mark.fast
def test_completion_generator_fibo_does_not_advertise_lora():
    generator = CompletionGenerator()
    parser = generator.create_parser_for_command("mflux-generate-fibo")
    script = generator.generate_command_function("mflux-generate-fibo", parser)

    assert "--lora-paths" not in script


@pytest.mark.fast
@pytest.mark.parametrize("command", ["mflux-generate-flux2-edit", "mflux-generate-qwen-edit"])
def test_completion_generator_edit_backends_advertise_canvas_expansion_options(command):
    generator = CompletionGenerator()
    parser = generator.create_parser_for_command(command)
    script = generator.generate_command_function(command, parser)

    assert "--reframe-padding" in script
    assert "--outpaint-padding" in script
    assert "--image-outpaint-padding" in script
