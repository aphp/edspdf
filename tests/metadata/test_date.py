from edspdf.metadata.extract_date import ExtractDate, find_date


def test_find_date(date_text):

    list_dates = find_date(date_text)

    dates = [
        ExtractDate(
            extract_date="24/06/2014",
            start_span=10,
            end_span=20,
            date_type="Examen du ",
        ),
        ExtractDate(
            extract_date="25 JUIN 2014",
            start_span=22,
            end_span=34,
            date_type="Examen du ",
        ),
        ExtractDate(
            extract_date="2-9-2014", start_span=47, end_span=55, date_type="imprimé le "
        ),
        ExtractDate(
            extract_date="28.10.2014", start_span=70, end_span=80, date_type="signé le "
        ),
    ]

    assert list_dates == dates
