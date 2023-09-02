import pytest

from edspdf.utils.package import package


def test_blank_package(frozen_pipeline, tmp_path):

    # Missing metadata makes poetry fail due to missing author / description
    with pytest.raises(Exception):
        package(
            pipeline=frozen_pipeline,
            root_dir=tmp_path,
            name="test-model",
            metadata={},
            project_type="poetry",
        )

    frozen_pipeline.package(
        root_dir=tmp_path,
        name="test-model",
        metadata={
            "description": "A test model",
            "authors": "Test Author <test.author@mail.com>",
        },
        project_type="poetry",
        distributions=["wheel"],
    )
    assert (tmp_path / "dist").is_dir()
    assert (tmp_path / "build").is_dir()
    assert (tmp_path / "dist" / "test_model-0.1.0-py3-none-any.whl").is_file()
    assert not (tmp_path / "dist" / "test_model-0.1.0.tar.gz").is_file()
    assert (tmp_path / "build" / "test-model").is_dir()


def test_package_with_files(frozen_pipeline, tmp_path):
    frozen_pipeline.save(tmp_path / "model")

    ((tmp_path / "test_model_trainer").mkdir(parents=True))
    (tmp_path / "test_model_trainer" / "__init__.py").write_text(
        """\
print("Hello World!")
"""
    )
    (tmp_path / "pyproject.toml").write_text(
        """\
[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"

[tool.poetry]
name = "test-model-trainer"
version = "0.0.0"
description = "A test model"
authors = ["Test Author <test.author@mail.com>"]

[tool.poetry.dependencies]
python = "^3.7"
torch = "^1.4.0"
"""
    )

    with pytest.raises(ValueError):
        package(
            pipeline=frozen_pipeline,
            root_dir=tmp_path,
            version="0.1.0",
            name="test-model",
            metadata={
                "description": "Wrong description",
                "authors": "Test Author <test.author@mail.com>",
            },
        )

    package(
        pipeline=tmp_path / "model",
        root_dir=tmp_path,
        check_dependencies=True,
        version="0.1.0",
        name="test-model",
        distributions=None,
        metadata={
            "description": "A test model",
            "authors": "Test Author <test.author@mail.com>",
        },
    )
    assert (tmp_path / "dist").is_dir()
    assert (tmp_path / "dist" / "test_model-0.1.0.tar.gz").is_file()
    assert (tmp_path / "dist" / "test_model-0.1.0-py3-none-any.whl").is_file()
    assert (tmp_path / "pyproject.toml").is_file()
