import re

from edspdf.metadata.regex import date_pattern, type_date_pattern

to_check = [
    "VUE LE 01/02/2010",
    "Date de l'examen : 2 juin 2006",
    "Prochaine consultation le 1-10-2014",
    "Née le 24.10.98",
    "Signé électroniquement le 18 02 2001",
    "Boulogne CEDEX, le 19:12:2018",
    "ISSY-LES-MOULINEAUX le, 26/3/12",
]


def test_date_regex():
    results = [re.findall(date_pattern, x) for x in to_check]
    wanted_results = [
        ["01/02/2010"],
        ["2 juin 2006"],
        ["1-10-2014"],
        ["24.10.98"],
        ["18 02 2001"],
        ["19:12:2018"],
        ["26/3/12"],
    ]
    assert results == wanted_results


def test_before_date_regex():
    results = [re.findall(type_date_pattern, x) for x in to_check]
    wanted_results = [
        ["VUE LE "],
        ["Date de l'examen "],
        ["Prochaine consultation le "],
        ["Née le "],
        ["Signé électroniquement le "],
        ["Boulogne CEDEX, le "],
        ["ISSY-LES-MOULINEAUX le"],
    ]
    assert results == wanted_results
