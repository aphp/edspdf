import re

FLAGS_TO_USE = re.IGNORECASE

day = r"(?:[1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])"
name_month = (
    r"janvier|janv|jan|février|fév|mars|avril|avr|mai|juin|juillet"
    r"|juil|août|septembre|sept|octobre|oct|novembre|nov|décembre|déc"
)
nb_month = r"[1-9]|0[1-9]|1[0-2]"
month = r"(?:" + nb_month + "|" + name_month + ")"
year = r"(?:19[0-9][0-9]|20[0-9][0-9]|[0-9][0-9])"
sep = r"(?:\/|\-|\.|\:|\s)"

list_date_pattern = [day, sep, month, sep, year]

date_pattern = re.compile(
    r"(" + "".join(x for x in list_date_pattern) + ")", FLAGS_TO_USE
)

loc_pattern = [
    r"Paris",
    r"Clichy",
    r"(?:|Kremlin)\s*\-?Bicêtre",
    r"Créteil",
    r"Boulogne\s*\-?(?:Billancourt|)",
    r"Clamart",
    r"Bobigny",
    r"Ivry\s*\-?sur\s*\-?Seine",
    r"Issy\s*\-?les\s*\-?Moulineaux",
    r"Draveil",
    r"Limeil",
    r"Champcueil",
    r"Bondy",
    r"Colombes",
    r"Hendaye",
    r"Berck\s*\-?sur\s*\-?mer",
    r"Villejuif",
    r"Labruyere",
    r"Garches",
    r"Sevran",
    r"Hyères",
    r"Gennevilliers",
    r"BERCK S/MER",
]

loc_pattern = r"(?:" + "|".join(x for x in loc_pattern) + ")"

end_pattern = r"\s*(?:|\:)\s*"

regexes = [
    r"Date de compte rendu",
    r"Compte rendu fait le",
    r"Date (?:d|de l)(?:\'|’)examen",
    r"Examen (?:réalisé le|du)",
    r"courrier patient du",
    r"(?:Prochaine)\s*(?:Consultation|CS)\s*(?:du|le)",
    r"Date\s*(?:du bilan|du jour)",
    r"Ordonnance du",
    r"Heure de prise en charge IAO",
    r"Date d(?:\'|’)entrée au SAU",
    r"Heure de décroché",
    r"Admission du",
    r"Vu(?:|e|\(e\))\s*le",
    r"Entré(?:|e|\(e\))\s*le",
    r"Sorti(?:|e|\(e\))\s*le",
    r"Date d(?:\'|´)intervention",
    r"Intervention du",
    r"Posé(?:|e)\s*le",
    r"Réalisé(?:|e)\s*le",
    r"Prévu(?:|e)\s*le",
    r"Hospitalisé\s*(?:le)",
    r"Hopistalisation\s*(?:du)",
    r"Hospitalisation de jour\s*(?:le|du)",
    r"Compte rendu d\'hospitalisation du",
    r"Compte rendu d\'hospitalisation du .* au",
    r"Document créé le",
    r"Saisi(?:|e) le",
    r"Edité(?:|e) le",
    r"Enregistré(?:|e) le",
    r"Dicté(?:|e) le",
    r"Cr tapé le",
    r"imprimé(?i:|e) le",
    r"Edition\s*(?:|sécurisée)\s*du",
    r"Signé\s*(?:|électroniquement)\s*le",
    r"Compte-rendu signé le",
    r"validé(?:|e)\s*le",
    r"RCP du",
    r"Date RCP",
    r"Date du test",
    r"test\s*(?:|du)",
    r"Prélevé(?:|e)\s*le",
    r"Fait le",
    r"Effectué(?:|e)\s*le",
    r"Date de réception",
    r"Reçu(?:|e)\s*le",
    loc_pattern + r"\s*(?:cedex|)(?:,|)\s*le",
    r"Date de Naissance|D.D.N|DDN|Né(?:|e|\(e\))\s*le",
]

type_date_pattern = re.compile(
    r"(" + "|".join(x + end_pattern for x in regexes) + ")", FLAGS_TO_USE
)
