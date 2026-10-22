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

Les documents sont conservés dans leur format brut, puis préparés pour l’indexation. Un fichier de métadonnées (`corpus_metadonnees.jsonl`) centralise les informations utiles à leur identification, leur classement et leur traçabilité : source, date, langue, domaine, statut et URL officielle.

L’objectif est de permettre à l’assistant de retrouver les passages pertinents et de **fournir des réponses sourcées, vérifiables et contextualisées**.


## Construction de la pipeline d'ingestion des pdf

### 1. Création du fichier json avec les métadonnées
Les **métadonnées** sont des informations qui décrivent et permettent de contextualiser chaque document du corpus, comme sa source, sa date, son domaine, sa langue ou son statut réglementaire. Elles servent à **organiser les documents, améliorer la pertinence de la recherche dans le RAG et assurer la traçabilité des réponses** en permettant notamment de retrouver le document et la page à l’origine d’une information. Je les crée à partir des informations disponibles sur chaque document (titre, référence, organisme émetteur, date, type, domaine, etc.), puis je les centralise dans un fichier `documents.jsonl`. Lors de l’ingestion, ces métadonnées sont ensuite associées aux chunks correspondants afin d’être conservées dans ChromaDB.

### 2. Chargement des documents

Cette étape parcourt automatiquement les différents dossiers, charge chaque PDF page par page avec `PyPDFLoader` et regroupe l’ensemble des pages dans une liste de documents LangChain. Les métadonnées associées aux pages sont conservées afin de pouvoir retracer leur document source. Cette étape constitue la base de la pipeline avant le **découpage en chunks**, la création des **embeddings** et l’indexation dans **ChromaDB**.

### 3. Découpage en chunks

Les pages extraites sont ensuite découpées en **chunks** à l’aide de `RecursiveCharacterTextSplitter`. Chaque chunk contient jusqu’à **1 200 caractères**, avec un chevauchement de **200 caractères** entre deux chunks consécutifs afin de préserver le contexte. Le découpage privilégie les frontières naturelles du texte (`paragraphes`, `lignes`, `phrases`, puis espaces). Les métadonnées des pages sont conservées et associées aux chunks, permettant de retrouver leur document et leur page d’origine avant la création des embeddings. 

Le découpage utilise une taille de **1 200 caractères** (environ 250–300 tokens), choisie comme compromis entre la conservation du contexte réglementaire et la précision de la recherche sémantique. Un **chevauchement de 200 caractères** est utilisé afin de limiter la perte de contexte lorsque les phrases ou dispositions sont réparties entre deux chunks. Ces paramètres pourront être ajustés ou calibrés selon les besoin a la phase de test

### 4. Enrichissement des chunks avec les métadonnées

Chaque chunk est enrichi à partir du fichier `data/corpus_metadata.json`. Le script identifie le fichier d’origine à partir de ses métadonnées de chargement, construit une clé correspondant à son chemin relatif dans `data/processed/`, puis recherche les informations associées dans le registre. Il ajoute ainsi à chaque chunk son identifiant documentaire, son titre, son émetteur, sa référence réglementaire, son domaine et son URL source. Une vérification permet enfin de repérer les documents dont les métadonnées n’ont pas été retrouvées. Cet enrichissement facilite le filtrage des résultats et la traçabilité des réponses du RAG.

### 5. Indexation dans ChromaDB

Les chunks enrichis sont ensuite indexés dans une **base vectorielle ChromaDB persistante**. Un modèle `all-MiniLM-L6-v2` est utilisé pour générer automatiquement les embeddings des textes. Pour chaque chunk, le texte, son embedding et ses métadonnées sont stockés dans la collection `regulatory_docs`. L’indexation est réalisée par **lots de 500 chunks** afin de limiter la charge mémoire. ChromaDB permet ainsi de conserver les documents et leurs représentations vectorielles pour effectuer ensuite une **recherche sémantique** à partir des questions de l’utilisateur.

## Construction de la RAG chain

La dernière étape consiste à construire la chaîne RAG permettant de répondre aux questions de l’utilisateur à partir du corpus réglementaire indexé dans ChromaDB. Le fonctionnement repose sur quatre étapes principales : **retrieval, construction du contexte, génération de la réponse et restitution des sources**.

### 1. Recherche des documents pertinents

La collection `regulatory_docs` est chargée depuis la base ChromaDB persistante. Lorsqu’une question est posée, le retriever utilise le modèle `all-MiniLM-L6-v2` pour représenter la requête sous forme vectorielle et rechercher les chunks sémantiquement les plus proches dans la base.

Par défaut, les **5 chunks les plus pertinents** sont récupérés. Pour chacun d’eux, le système conserve à la fois le contenu textuel et les métadonnées associées au document (`title`, `reference`, `source_url`, etc.).

### 2. Construction du contexte

Les chunks récupérés sont ensuite regroupés pour constituer le contexte transmis au LLM. Chaque extrait est précédé des informations permettant d’identifier sa source :

```text
[Titre du document – Référence]

Contenu de l’extrait
```

Cette étape permet de conserver le lien entre le contenu utilisé pour répondre à la question et le document réglementaire dont il provient.

### 3. Génération de la réponse

Le contexte récupéré et la question de l’utilisateur sont injectés dans un prompt destiné au LLM `openai/gpt-oss-120b`, utilisé via Groq.

Le prompt impose plusieurs règles afin de rendre les réponses adaptées à un usage réglementaire :

- répondre uniquement à partir des extraits fournis ;
- signaler explicitement lorsqu’une information n’est pas présente dans le contexte ;
- citer les documents utilisés sous la forme `[Titre – Référence]`.

La température du modèle est fixée à `0` afin de privilégier des réponses déterministes et reproductibles.

### 4. Restitution des sources

La fonction `ask()` retourne finalement deux éléments : la **réponse générée** et la **liste des sources utilisées**. Pour chaque document, le système restitue son titre, sa référence et, lorsqu’elle est disponible, son URL officielle.

La réponse peut ainsi être vérifiée et replacée dans son contexte documentaire, ce qui constitue un élément essentiel pour une application d’intelligence réglementaire.

## Interface utilisateur avec Streamlit

L'application est déployée sous la forme d'une interface web développée avec **Streamlit**.

L'interface propose une zone de saisie pour la question de l'utilisateur. Lorsqu'une question est renseignée, elle est transmise à la fonction `ask()` de `src/rag_chain.py`, qui exécute la chaîne RAG.

L'interface affiche ensuite séparément la **réponse générée** et les **documents utilisés**. Pour chaque source, le titre et la référence réglementaire sont affichés et, lorsqu'une URL officielle est disponible, le titre devient un lien cliquable vers le document source.


```text
Question utilisateur
        ↓
Interface Streamlit
        ↓
      ask()
        ↓
Recherche ChromaDB
        ↓
Construction du contexte
        ↓
LLM
        ↓
Réponse + sources
        ↓
Affichage Streamlit
```

## Tests et validation

Des tests automatisés avec **Pytest** permettent de vérifier le bon fonctionnement de la pipeline RAG et la qualité de l'index documentaire.

Une fixture commune établit une connexion à la collection `regulatory_docs` de ChromaDB en utilisant le même modèle d'embeddings `all-MiniLM-L6-v2` que lors de l'indexation.

Plusieurs contrôles sont réalisés :

- **Volume de l'index** : vérification que la collection contient plus de 10 000 chunks ;
- **Métadonnées** : vérification de la présence d'informations essentielles telles que `document_id` et `issuer` ;
- **Classification documentaire** : vérification que les résultats associés aux requêtes Stress Test et IFRS 9 possèdent bien les scopes correspondants ;
- **Pipeline de bout en bout** : lorsqu'une clé API Groq est disponible, une question est envoyée à la RAG complète et le test vérifie qu'une réponse ainsi que des sources sont retournées.

Ces tests permettent de détecter les erreurs d'ingestion, d'enrichissement des métadonnées, d'indexation ou de fonctionnement de la chaîne RAG avant son utilisation dans l'interface utilisateur.

## Conclusion, limites et perspectives

Ce projet met en œuvre un assistant d’intelligence réglementaire basé sur une architecture **RAG**, combinant recherche sémantique et génération de réponses sourcées sur Bâle/IRB, IFRS 9, stress tests et ICAAP.

Le prototype présente certaines limites, notamment la couverture du corpus, la pertinence des extraits récupérés et la nécessité de vérifier les réponses générées. Des améliorations pourraient porter sur l’évaluation de la qualité du retrieval, le reranking des résultats et la gestion des versions réglementaires.

À terme, cette solution pourrait être adaptée aux besoins des banques en intégrant leurs **livres blancs, procédures internes, documentations de modèles et supports de formation**. Elle faciliterait ainsi la recherche documentaire, le partage des connaissances et la montée en compétence des équipes risque et conformité, tout en centralisant l’accès à des informations internes et réglementaires.
