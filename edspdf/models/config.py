from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from edspdf.reading import PdfReader

from .enums import DeployMode, Master, Queue


class SparkConf(BaseModel):

    spark_home: Path
    master: Master
    queue: Queue
    deploy_mode: DeployMode
    driver_memory: str
    executor_memory: str
    num_executor: int
    executor_cores: int
    memoryOverhead: str
    timeZone: str
    archives: str


class ExtractConf(BaseModel):

    pdfs: str
    output: str
    documents: str
    reader: PdfReader
    limit: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True


class ConfigSchema(BaseModel):

    spark: SparkConf
    extract: ExtractConf
