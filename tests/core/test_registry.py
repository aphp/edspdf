import pytest

from edspdf.pipeline import Pipeline
from edspdf.registry import CurriedFactory, registry


def test_misc_register_decorator():
    @registry.misc.register("test-1")
    def test_function(param: int = 3):
        pass

    assert test_function is registry.misc.get("test-1")


def test_misc_register_call():
    def test_function(param: int = 3):
        pass

    test_function_2 = registry.misc.register("test", func=test_function)
    assert test_function_2 is registry.misc.get("test")


def test_factory_default_config():
    @registry.factory.register("custom-test-component-1", default_config={"value": 5})
    class CustomComponent:
        def __init__(self, pipeline: "Pipeline", name: str, value: int = 3):
            self.name = name
            self.value = value

        def __call__(self, *args, **kwargs):
            return self.value

    registry_result = registry.factory.get("custom-test-component-1")()
    assert isinstance(registry_result, CurriedFactory)

    pipeline = Pipeline()
    pipeline.add_pipe("custom-test-component-1")

    assert pipeline.get_pipe("custom-test-component-1").value == 5


def test_factory_required_arguments():
    with pytest.raises(ValueError) as exc_info:

        @registry.factory.register("custom-test-component-2")
        class CustomComponent:
            def __init__(self, value: int = 3):
                self.value = value

            def __call__(self, *args, **kwargs):
                return self.value

    assert "Factory functions must accept pipeline and name as arguments." in str(
        exc_info.value
    )


def test_missing_component():
    pipeline = Pipeline()

    with pytest.raises(ValueError) as exc_info:
        pipeline.add_pipe("missing_custom_test_component")

    assert (
        "Can't find 'missing_custom_test_component' in registry edspdf -> factories."
        in str(exc_info.value)
    )
