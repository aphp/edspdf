class compose:
    def __init__(self, *functions):
        self.functions = functions

    def __call__(self, arg):
        for fn in self.functions:
            arg = fn(arg)
        return arg
