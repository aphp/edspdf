from enum import Enum


class WriteMode(str, Enum):
    overwrite = "overwrite"
    append = "append"


class Master(str, Enum):
    yarn = "yarn"


class Queue(str, Enum):
    default = "default"


class DeployMode(str, Enum):
    cluster = "cluster"
    client = "client"
    local = "local"
