from edspdf import registry


def test_register_decorator():
    @registry.misc.register("test-1")
    def test_function(param: int = 3):
        pass

    assert test_function is registry.misc.get("test-1")


def test_register_call():
    def test_function(param: int = 3):
        pass

    test_function_2 = registry.misc.register("test", func=test_function)
    assert test_function_2 is registry.misc.get("test")
