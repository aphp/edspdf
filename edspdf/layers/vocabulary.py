import contextlib
from typing import Generic, Sequence, TypeVar

import torch

T = TypeVar("T")


class Vocabulary(torch.nn.Module, Generic[T]):
    """
    Vocabulary layer.
    This is not meant to be used as a `torch.nn.Module` but subclassing
    `torch.nn.Module` makes the instances appear when printing a model, which is nice.
    """

    def __init__(self, items: Sequence[T] = None, default: int = -100):
        """
        Parameters
        ----------
        items: Sequence[InputT]
            Initial vocabulary elements if any.
            Specific elements such as padding and unk can be set here to enforce their
            index in the vocabulary.
        default: int
            Default index to use for out of vocabulary elements
            Defaults to -100
        """
        super().__init__()
        self.indices = {} if items is None else {v: i for i, v in enumerate(items)}
        self.initialized = True
        self.default = default

    def __len__(self):
        return len(self.indices)

    @contextlib.contextmanager
    def initialization(self):
        """
        Enters the initialization mode.
        Out of vocabulary elements will be assigned an index.
        """
        self.initialized = False
        yield
        self.initialized = True

    def encode(self, item):
        """
        Converts an element into its vocabulary index
        If the layer is in its initialization mode (`with vocab.initialization(): ...`),
        and the element is out of vocabulary, a new index will be created and returned.
        Otherwise, any oov element will be encoded with the `default` index.

        Parameters
        ----------
        item: InputT

        Returns
        -------
        int
        """
        if self.initialized:
            return self.indices.get(
                item, self.default
            )  # .setdefault(item, len(self.indices))
        else:
            return self.indices.setdefault(
                item, len(self.indices)
            )  # .setdefault(item, len(self.indices))

    def decode(self, idx):
        """
        Converts an index into its original value

        Parameters
        ----------
        idx: int

        Returns
        -------
        InputT
        """
        return list(self.indices.keys())[idx] if idx >= 0 else None

    def extra_repr(self):
        return "n={}".format(len(self.indices))
