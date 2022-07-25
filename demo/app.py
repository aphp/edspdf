import base64

import pandas as pd
import pdf2image  # noqa
import streamlit as st
from thinc.config import Config

from edspdf import registry
from edspdf.readers.reader import PdfReader
from edspdf.visualization.annotations import show_annotations
from edspdf.visualization.merge import merge_lines

CONFIG = """\
[reader]
@readers = "pdf-reader.v1"

[reader.extractor]
@extractors = "pdfminer.v1"

[reader.classifier]
@classifiers = "mask.v1"
x0 = 0.1
x1 = 0.9
y0 = 0.4
y1 = 0.9
threshold = 0.1

[reader.aggregator]
@aggregators = "styled.v1"\
"""


st.set_page_config(
    page_title="EDS-PDF Demo",
    page_icon="📄",
)

st.title("EDS-PDF")

st.warning(
    "You should **not** put sensitive data in the example, as this application "
    "**is not secure**."
)

st.sidebar.header("About")
st.sidebar.markdown(
    "EDS-PDF is a contributive effort maintained by AP-HP's Data Science team. "
    "Have a look at the "
    "[documentation](https://aphp.github.io/edspdf/) for more information."
)


st.header("Extract a PDF")

st.subheader("Configuration")
config = st.text_area(label="Change the config", value=CONFIG, height=200)


def load_model(cfg) -> PdfReader:
    config = Config().from_str(cfg)
    resolved = registry.resolve(config)
    return resolved["reader"]


model_load_state = st.info("Loading model...")

reader = load_model(config)

model_load_state.empty()

st.subheader("Input")
upload = st.file_uploader("PDF to analyse", accept_multiple_files=False)

if upload:

    pdf = upload.getvalue()

    base64_pdf = base64.b64encode(pdf).decode("utf-8")

    text, styles = reader(pdf)

    body = text.get("body")
    body_styles = styles.get("body")

    pdf_display = f"""\
    <iframe
        src="data:application/pdf;base64,{base64_pdf}"
        width="700"
        height="1000"
        type="application/pdf">
    </iframe>"""

    st.subheader("Output")

    with st.expander("Visualisation"):

        lines = reader.prepare_and_predict(pdf)  # noqa
        merged = merge_lines(lines)

        imgs = show_annotations(pdf=pdf, annotations=merged)

        page = st.selectbox("Pages", options=[i + 1 for i in range(len(imgs))]) - 1

        st.image(imgs[page])

    with st.expander("PDF"):
        st.markdown(pdf_display, unsafe_allow_html=True)

    with st.expander("Text"):
        if body is None:
            st.warning(
                "No text detected... Are you sure this is a text-based PDF?\n\n"
                "There is no support for OCR within EDSPDF (for now?)."
            )
        else:
            st.markdown("```\n" + body + "\n```")

    with st.expander("Styles"):
        if body_styles is None:
            st.warning(
                "No text detected... Are you sure this is a text-based PDF?\n\n"
                "There is no support for OCR within EDSPDF (for now?)."
            )
        else:
            st.dataframe(pd.DataFrame.from_records(body_styles))
