# REGULATORY_RAG_ASSISTANT
Assistant RAG dédié à l’analyse des textes réglementaires bancaires (BCE, AMF, ACPR, CRR/CRD IV, guidelines EBA), permettant d’interroger la réglementation en langage naturel et d’obtenir des réponses sourcées pour les équipes risque, conformité et conseil réglementaire.

## Data


Le projet s’appuie sur un corpus de documents réglementaires et de publications institutionnelles couvrant quatre domaines : **Bâle / IRB, IFRS 9, stress tests et ICAAP**.

| Domaine          | Types de documents                                                | Principaux sujets                                                            |
| ---------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **Bâle / IRB**   | Règlements, directives, guidelines, guides prudentiels            | Définition du défaut, PD/LGD, modèles internes, classes de risque            |
| **IFRS 9**       | Normes comptables, règlements européens, guidelines, rapports     | Pertes de crédit attendues (ECL), dépréciation, classification et évaluation |
| **Stress tests** | Méthodologies, guidelines, rapports institutionnels, publications | Scénarios macroéconomiques, risques climatiques, tests de résistance         |
| **ICAAP**        | Guides prudentiels, guidelines, documents de supervision          | Adéquation du capital interne, évaluation des risques, processus SREP        |

Les sources proviennent notamment de l’**EBA, de la BCE, du Comité de Bâle, de la Banque de France et de la Fondation IFRS**.

Les documents sont conservés dans leur format brut, puis préparés pour l’indexation. Un fichier de métadonnées (`documents.jsonl`) centralise les informations utiles à leur identification, leur classement et leur traçabilité : source, date, langue, domaine, statut et URL officielle.

L’objectif est de permettre à l’assistant de retrouver les passages pertinents et de **fournir des réponses sourcées, vérifiables et contextualisées**.
