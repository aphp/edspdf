from thinc.config import Config

from edspdf import registry
from edspdf.models.config import ConfigSchema

configuration = """
[spark]
spark_home = /usr/hdp/current/spark2.4.3-client/
master = yarn
queue = default
deploy_mode = cluster
driver_memory = 15g
executor_memory = 15g
num_executor = 5
executor_cores = 5
memoryOverhead = 6g
timeZone = Europe/Paris
archives = env/venv.tar.gz
jars = "/export/home/opt/lib/EdsTools-0.0.1-SNAPSHOT.jar"


[tables]

documents = edsprod.orbis_document
output = edsprod.orbis_document_text

[tables.hdfs]
subset = /tmp/pdfs-subset
extracted = /tmp/extracted-subset

[extract]

pdfs = ${tables.hdfs.subset}
documents = ${tables.documents}
output = ${tables.hdfs.extracted}
limit = null

[extract.reader]
@readers = "pdf-reader.v1"
new_line_threshold = 0.2
new_paragraph_threshold = 1.2

[extract.reader.extractor]
@extractors = "line-extractor.v1"
style = true
laparams = { "@params": "laparams.v1" }

[extract.reader.classifier]
@classifiers = "dummy.v1"

[extract.reader.transform]
@transforms = "chain.v1"

[extract.reader.transform.*.dates]
@transforms = "dates.v1"

[extract.reader.transform.*.orbis]
@transforms = "orbis.v1"

[extract.reader.transform.*.telephone]
@transforms = "telephone.v1"

[extract.reader.transform.*.dimensions]
@transforms = "dimensions.v1"

"""


def test_configuration():
    cfg = Config().from_str(configuration)
    resolved = registry.resolve(cfg)

    ConfigSchema.parse_obj(resolved)
