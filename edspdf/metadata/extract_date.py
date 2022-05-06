import re
from typing import List, Optional

from pydantic import BaseModel

from .regex import date_pattern, type_date_pattern


class ExtractDate(BaseModel):
    extract_date: Optional[str] = None
    start_span: Optional[int] = None
    end_span: Optional[int] = None
    date_type: Optional[str] = None


def find_date(text: str) -> List[ExtractDate]:
    """
    Extracts dates from document.

    Arguments
    ---------
    text:
        String text if body or List of Strings in header, title, footer.

    Returns
    -------
    match_list:
        list, containing the extracted date and info (of the date).
    """
    match_list = []

    if text:
        if type(text) is list:
            text = " ".join([elem.text for elem in text])
        text = text.replace("\xa0", " ")
        date_list = list(re.finditer(date_pattern, text))
        if len(date_list) > 0:
            for elem in date_list:
                found_date = elem.group()
                start_span = int(elem.span()[0])
                end_span = int(elem.span()[1])
                # check telephone
                text_to_check = text[max(0, start_span - 8) : end_span + 8]
                if re.search(r"(?:\d\d[-\. ]?){5,}", text_to_check):
                    continue
                # if date is at the beginning of the text
                if start_span >= 30:
                    start_check_text = start_span - 30
                else:
                    start_check_text = 0
                text_before = text[start_check_text:start_span]
                found_type_list = re.findall(type_date_pattern, text_before)
                if len(found_type_list) > 0:
                    found_regex = found_type_list[0]
                else:
                    found_regex = "UNK"

                extract_date_info = ExtractDate(
                    extract_date=found_date,
                    start_span=start_span,
                    end_span=end_span,
                    date_type=found_regex,
                )
                match_list.append(extract_date_info)

    return match_list
