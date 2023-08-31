def test_repr(styles_pdf):
    from edspdf.pipes.extractors.pdfminer import PdfMinerExtractor

    doc = PdfMinerExtractor(extract_style=True)(styles_pdf)
    doc.id = "test"

    for b in doc.content_boxes:
        b.x0 = round(b.x0, 2)
        b.y0 = round(b.y0, 2)
        b.x1 = round(b.x1, 2)
        b.y1 = round(b.y1, 2)

    assert repr(doc) == (
        "PDFDoc(content=39476 bytes, id='test', num_pages=0, pages=[Page(page_num=0, "
        "width=612, height=792, image=None)], error=False, "
        "content_boxes=[TextBox(x0=0.12, x1=0.65, y0=0.09, y1=0.11, label=None, "
        "page_num=0, text='This is a test to check EDS-PDF’s ability to detect "
        "changing styles.', props=[TextProperties(italic=False, bold=False, begin=0, "
        "end=9, fontname='AAAAAA+ArialMT'), TextProperties(italic=False, bold=True, "
        "begin=10, end=14, fontname='BAAAAA+Arial-BoldMT'), "
        "TextProperties(italic=False, bold=False, begin=15, end=33, "
        "fontname='AAAAAA+ArialMT'), TextProperties(italic=True, bold=False, "
        "begin=34, end=41, fontname='CAAAAA+Arial-ItalicMT'), "
        "TextProperties(italic=False, bold=False, begin=42, end=68, "
        "fontname='AAAAAA+ArialMT')]), TextBox(x0=0.12, x1=0.73, y0=0.11, y1=0.13, "
        "label=None, page_num=0, text='Let’s up the stakes, with intra-word change. "
        "Or better yet, this might be hard.', props=[TextProperties(italic=False, "
        "bold=False, begin=0, end=25, fontname='AAAAAA+ArialMT'), "
        "TextProperties(italic=True, bold=False, begin=26, end=31, "
        "fontname='CAAAAA+Arial-ItalicMT'), TextProperties(italic=False, bold=False, "
        "begin=31, end=59, fontname='AAAAAA+ArialMT'), TextProperties(italic=False, "
        "bold=True, begin=60, end=67, fontname='BAAAAA+Arial-BoldMT'), "
        "TextProperties(italic=False, bold=False, begin=67, end=79, "
        "fontname='AAAAAA+ArialMT')])], aggregated_texts={})"
    )
