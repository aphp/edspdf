import pytest
from pydantic import ValidationError, validate_arguments

from edspdf import Pipeline
from edspdf.config import Config, MissingReference, Reference
from edspdf.layers.box_embedding import BoxEmbedding


class CustomClass:
    pass


pipeline_config = """[pipeline]
components = ["extractor","classifier","aggregator"]
components_config = ${components}

[components]

[components.extractor]
@factory = pdfminer-extractor
extract_style = false

[components.classifier]
@factory = deep-classifier
labels = []

[components.classifier.embedding]
@factory = box-embedding
size = 96
dropout_p = 0.2

[components.classifier.embedding.text_encoder]
@factory = box-text-embedding

[components.classifier.embedding.text_encoder.pooler]
@factory = cnn-pooler
out_channels = 64
kernel_sizes = [3,4,5]

[components.classifier.embedding.layout_encoder]
@factory = box-layout-embedding
n_positions = 128
x_mode = sin
y_mode = sin
w_mode = sin
h_mode = sin

[components.aggregator]
@factory = simple-aggregator

"""


def test_read_from_str():
    config = Config().from_str(pipeline_config, resolve=False)
    assert config == {
        "components": {
            "extractor": {
                "@factory": "pdfminer-extractor",
                "extract_style": False,
            },
            "classifier": {
                "@factory": "deep-classifier",
                "embedding": {
                    "@factory": "box-embedding",
                    "layout_encoder": {
                        "@factory": "box-layout-embedding",
                        "n_positions": 128,
                        "x_mode": "sin",
                        "y_mode": "sin",
                        "h_mode": "sin",
                        "w_mode": "sin",
                    },
                    "text_encoder": {
                        "@factory": "box-text-embedding",
                        "pooler": {
                            "@factory": "cnn-pooler",
                            "kernel_sizes": [3, 4, 5],
                            "out_channels": 64,
                        },
                    },
                    "dropout_p": 0.2,
                    "size": 96,
                },
                "labels": [],
            },
            "aggregator": {"@factory": "simple-aggregator"},
        },
        "pipeline": {
            "components": ["extractor", "classifier", "aggregator"],
            "components_config": Reference("components"),
        },
    }
    resolved = config.resolve(deep=True)
    assert isinstance(resolved["components"]["classifier"].embedding, BoxEmbedding)
    exported_config = config.to_str()
    assert exported_config == pipeline_config


def test_write_to_str():
    def reexport(s):
        config = Config().from_str(s).resolve(deep=True)
        return Config(pipeline=Pipeline(**config["pipeline"])).to_str()

    exported = reexport(pipeline_config)
    assert reexport(exported) == exported


def test_cast_parameters():
    @validate_arguments
    def function(a: str, b: str, c: int, d: int):
        assert a == b
        assert c == d

    config = """
[params]
a = okok.okok
b = "okok.okok"
c = "12"
d = 12
"""
    params = Config.from_str(config)["params"]
    assert params == {
        "a": "okok.okok",
        "b": "okok.okok",
        "c": "12",
        "d": 12,
    }
    function(**params)


def test_dump_error():
    with pytest.raises(TypeError):
        Config(test=CustomClass()).to_str()


def test_missing_error():
    with pytest.raises(MissingReference) as exc_info:
        Config.from_str(
            """
        [params]
        a = okok.okok
        b = ${missing}
        """
        ).resolve()
    assert (
        str(exc_info.value)
        == "Could not interpolate the following references: ${missing}"
    )


def test_type_hinted_instantiation_error():
    @validate_arguments
    def function(embedding: BoxEmbedding):
        ...

    params = Config.from_str(
        """
    [embedding]
    size = "ok"
    """
    )
    with pytest.raises(ValidationError) as exc_info:
        function(**params)
    assert str(exc_info.value) == (
        "1 validation error for Function\n"
        "embedding -> size\n"
        "  value is not a valid integer (type=type_error.integer)"
    )


def test_factory_instantiation_error():
    with pytest.raises(ValidationError) as exc_info:
        Config.from_str(
            """
        [embedding]
        @factory = "box-embedding"
        dropout_p = "ok"
        """
        ).resolve(deep=True)
    assert str(exc_info.value) == (
        "2 validation errors for BoxEmbedding\n"
        "embedding -> size\n"
        "  field required (type=value_error.missing)\n"
        "embedding -> dropout_p\n"
        "  value is not a valid float (type=type_error.float)"
    )


def test_absolute_dump_path():
    config_str = Config(
        value=dict(
            moved=Config(
                test="ok",
                __path__=("my", "deep", "path"),
            ),
        )
    ).to_str()
    assert config_str == (
        "[value]\n"
        "moved = ${my.deep.path}\n"
        "\n"
        "[my]\n"
        "\n"
        "[my.deep]\n"
        "\n"
        "[my.deep.path]\n"
        "test = ok\n"
        "\n"
    )


def test_merge():
    config = Config().from_str(pipeline_config)
    other = Config().from_str(
        """\
[components]

[components.extractor]
@factory = "pdfminer-extractor"
extract_style = true

[components.extra]
size = 128
"""
    )
    merged = config.merge(other, remove_extra=True)
    merged = merged.merge(
        Config(components=Config(new_component=Config(test="ok"))), remove_extra=False
    )
    assert merged["components"]["extractor"]["extract_style"] is True
    assert "new_component" in merged["components"]
    merged = merged.merge(
        {
            "components.extractor.extract_style": True,
            "components.other_extra": {"key": "val"},
        },
        remove_extra=False,
    )
    merged = merged.merge(
        {
            "components.missing_subsection.size": 96,
            "components.extractor.missing_key": 96,
        },
        remove_extra=True,
    )
    resolved = merged.resolve()
    assert merged["components"]["extractor"]["extract_style"] is True
    assert "extra" not in resolved["components"]
    assert "other_extra" in resolved["components"]
