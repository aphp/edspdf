# EDS-PDF

EDS-PDF est un outil développé à l'Entrepôt de données de santé (EDS) de l'AP-HP pour extraire le texte à partir de documents médicaux sous forme de PDF.

EDS-PDF utilise la [bibliothèque Python `pdfminer.six`](https://pdfminersix.readthedocs.io/en/latest/topic/converting_pdf_to_text.html). Celle-ci permet d'extraire le texte de PDF, en conservant toutes les informations jusqu'au plus bas niveau. Par exemple :

- position et dimensions des _bounding boxes_
- police, taille graisse des éléments de texte

`pdfminer` conserve une vision hiérarchique des éléments du PDF.

Un pipeline d'extraction du corps du texte a été testée avec cette librairie.

## Projets connexes

La solution utilisée aujourd'hui à l'EDS pour l'extraction de texte repose sur la bibliothèque Java [PDFBox](https://pdfbox.apache.org/). Très rapide, elle ne permet cependant pas de sélection fine du corps du texte. En effet, la méthode utilise un masque fixe qui empêche de retirer toutes les pollutions, et peut à l'inverse cacher des informations médicalement pertinentes.

Nous avons également expérimenté avec la solution [GROBID](https://grobid.readthedocs.io/en/latest/){footcite:p}`GROBID`, qui n'a finalement pas eu les résultats escomptés. En particulier, il n'était pas possible de conserver l'ordre des lignes (les titres de sections étaient typiquement regroupés).

| Solution | Langage | Rapidité (qualitative) | Classification   | Déploiement   | Ordre des lignes respecté |
| -------- | ------- | ---------------------- | ---------------- | ------------- | ------------------------- |
| PDFBox   | Java    | Rapide                 | Masque statique  | En production | Partiellement             |
| GROBID   | Java    | Lent                   | CRF en cascade   | Difficile     | Non                       |
| EDS-PDF  | Python  | Lent                   | ML sur les blocs | Facile        | Oui                       |

La rapidité est présentée de façon qualitative dans le tableau qui précède. GROBID et EDS-PDF, en retraitant le texte à l'aide d'algorithmes d'apprentissage automatique, sont naturellement mois performants mais réalisent une tâche plus importante, à savoir la classification des zones de texte.

## Fonctionnement de l'algorithme

1. PDFMiner extrait le texte, sous forme de blocs détectés par l'outil.
2. Un algorithme de machine learning utilise la position et l'étalement du bloc, en plus de son contenu, pour déterminer s'il appartient au corps du texte ou à une catégorie de méta-données.
3. Les blocs du corps du texte sont fusionnés, en respectant l'ordre des lignes, afin de reconstruire le texte final.

## Documentation

Pour plus de détail, voir la [documentation](https://bigdata-pages.eds.aphp.fr/algorithms/edspdf/).
