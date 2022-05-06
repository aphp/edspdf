import base64

import pdf2image
import streamlit as st
from thinc.config import Config

from edspdf import registry
from edspdf.reading.reader import PdfReader
from edspdf.visualization.merge import merge_lines

CATEGORY20 = [
    "#1f77b4",
    # "#aec7e8",
    "#ff7f0e",
    # "#ffbb78",
    "#2ca02c",
    "#98df8a",
    "#d62728",
    "#ff9896",
    "#9467bd",
    "#c5b0d5",
    "#8c564b",
    "#c49c94",
    "#e377c2",
    "#f7b6d2",
    "#7f7f7f",
    "#c7c7c7",
    "#bcbd22",
    "#dbdb8d",
    "#17becf",
    "#9edae5",
]

CONFIG = """\
[reader]
@readers = "pdf-reader.v1"
new_line_threshold = 0.2
new_paragraph_threshold = 1.2

[reader.extractor]
@extractors = "line-extractor.v1"
style = true
laparams = { "@params": "laparams.v1" }

[reader.transform]
@transforms = "chain.v1"

[reader.transform.*.dates]
@transforms = "dates.v1"

[reader.transform.*.orbis]
@transforms = "orbis.v1"

[reader.transform.*.telephone]
@transforms = "telephone.v1"

[reader.transform.*.dimensions]
@transforms = "dimensions.v1"

[reader.classifier]
@classifiers = "mask.v1"
x0 = 0.1
x1 = 0.9
y0 = 0.4
y1 = 0.9
threshold = 0.1\
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
    "[documentation](https://aphp.github.io/edsnlp/) for more information."
)


st.header("Extract a PDF")

st.subheader("Configuration")
config = st.text_area(label="Change the config", value=CONFIG, height=200)


def load_model(cfg) -> PdfReader:
    config = Config().from_str(cfg)
    resolved = registry.resolve(config)
    return resolved["reader"]


model_load_state = st.info("Loading model...")

model = load_model(config)

model_load_state.empty()

st.subheader("Input")
upload = st.file_uploader("PDF to analyse", accept_multiple_files=False)

if upload:

    pdf = upload.getvalue()

    base64_pdf = base64.b64encode(pdf).decode("utf-8")

    body = model(pdf, orbis=True)

    pdf_display = f"""\
    <iframe
        src="data:application/pdf;base64,{base64_pdf}"
        width="700"
        height="1000"
        type="application/pdf">
    </iframe>"""

    st.subheader("Output")

    with st.expander("Visualisation"):
        from PIL import ImageDraw

        lines = model.extractor(pdf)
        lines["label"] = model.classifier.predict(lines)  # noqa
        merged = merge_lines(lines)

        colors = {
            label: f"{color}01"
            for label, color in zip(lines.label.unique(), CATEGORY20)
        }

        page = st.selectbox("Pages", options=list(merged.page.unique() + 1)) - 1

        img = pdf2image.convert_from_bytes(pdf)[page]
        w, h = img.size
        draw = ImageDraw.Draw(img)
        for _, bloc in merged.query("page == @page").iterrows():
            draw.rectangle(
                [(bloc.x0 * w, bloc.y0 * h), (bloc.x1 * w, bloc.y1 * h)],
                outline=colors[bloc.label],
                width=4,
            )
        st.image(img)

    with st.expander("PDF"):
        st.markdown(pdf_display, unsafe_allow_html=True)

    with st.expander("Text"):
        st.markdown("```\n" + body + "\n```")
