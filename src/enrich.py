"""Enrichissement des chunks ChromaDB depuis corpus_metadata.json.

Ajoute à chaque chunk : document_id, title, reference, regulator (page est déjà présente).
Les citations doivent s'appuyer sur ces champs, jamais sur le nom de fichier :
`source` est donc remplacé par le document_id.

Usage :
    python src/enrich.py --check   # affiche l'appariement PDF <-> JSON, ne modifie rien
    python src/enrich.py           # applique l'enrichissement à la collection (sans ré-embedder)

Le script est idempotent. Appariement PDF <-> JSON : voir match_all (forcer un cas via OVERRIDES).
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

import chromadb

ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "corpus_metadata.json"
PDF_DIR = ROOT / "data" / "processed"
CHROMA_DIR = ROOT / "data" / "chromadb"
COLLECTION = "regulatory_docs"
BATCH = 2000

# stem du PDF -> index (0-based) de l'entrée dans le JSON, pour les cas non résolus automatiquement
OVERRIDES: dict[str, int] = {}

REGULATOR_ALIASES = {
    "european banking authority": "EBA",
    "european central bank": "ECB",
    "european union": "EU",
    "european commission": "European Commission",
}
MISSING = {"", "to_verify", "not_applicable", "none", "null"}


# ---------------------------------------------------------------- JSON tolérant
def load_records(path: Path = JSON_PATH) -> list[dict]:
    """Le fichier n'est pas du JSON valide : commentaires '# ...', None, objets concaténés."""
    text = "\n".join(l for l in path.read_text(encoding="utf-8").splitlines()
                     if not l.lstrip().startswith("#"))
    text = re.sub(r"\bNone\b", "null", text)
    text = re.sub(r",(\s*[}\]])", r"\1", text)  # virgules finales
    dec, pos, out = json.JSONDecoder(), 0, []
    while True:
        m = re.compile(r"\S").search(text, pos)
        if not m:
            return out
        obj, pos = dec.raw_decode(text, m.start())
        out.append(obj)


def clean(v):
    if v is None or str(v).strip().lower() in MISSING:
        return None
    return str(v).strip()


# ---------------------------------------------------------------- appariement
def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def stem_of(pdf: Path) -> str:
    return pdf.name.removesuffix(".pdf").removesuffix(".pdf")


def score(stem: str, rec: dict) -> float:
    st_tokens = {t[:6] for t in norm(stem).split()}
    st_norm = "_".join(norm(stem).split())
    lang = stem.rsplit("_", 1)[-1]
    s = 0.0
    if rec.get("language") and lang in ("FR", "EN"):
        s += 5 if rec["language"] == lang else -100
    ref = clean(rec.get("reference"))
    if ref:
        nums = re.findall(r"\d+", ref)
        if len(nums) >= 2 and "_".join(nums[-2:]) in st_norm.replace("_gl_", "_"):
            s += 20
        elif len(nums) >= 2 and "_".join(nums[-2:][::-1]) in st_norm:
            s += 20
    did = clean(rec.get("document_id"))
    ref_toks = set(norm(did.replace("_", " ")).split()) if did else set()
    title_toks = {t[:6] for t in set(norm(rec.get("title") or "").split()) | ref_toks}
    if st_tokens and title_toks:
        s += 10 * len(st_tokens & title_toks) / len(st_tokens)
    pub = clean(rec.get("publication_date"))
    if pub and pub[:4] in norm(stem).split():
        s += 3
    if clean(rec.get("short_name")) and norm(rec["short_name"]) in norm(stem).split():
        s += 5
    return s


MIN_SCORE = 7.0


def match_all(records: list[dict], stems: list[str]) -> dict[str, tuple[int, float] | None]:
    """Affectation gloutonne par score décroissant : une entrée JSON = au plus un PDF.
    stem -> (index, score), ou None si aucune entrée ne convient (document absent du JSON)."""
    result: dict[str, tuple[int, float] | None] = {}
    pairs = sorted(((score(st, r), st, i) for st in stems if st not in OVERRIDES
                    for i, r in enumerate(records)), reverse=True)
    taken = set(OVERRIDES.values())
    for st, i in OVERRIDES.items():
        result[st] = (i, 99.0)
    for sc, st, i in pairs:
        if st in result or i in taken or sc < MIN_SCORE:
            continue
        result[st] = (i, sc)
        taken.add(i)
    return {st: result.get(st) for st in stems}


def infer_regulator(stem: str) -> str | None:
    """Dernier recours quand le JSON n'indique pas d'émetteur (préfixe du nom de fichier)."""
    for prefix, reg in [("EBA_", "EBA"), ("ECB_", "ECB"), ("BCBS_", "BCBS"), ("ACPR_", "ACPR"),
                        ("EU_", "EU"), ("BANQUE_DE_FRANCE", "Banque de France"),
                        ("IFRS_FOUNDATION", "IFRS Foundation"), ("IASB_", "IASB")]:
        if stem.startswith(prefix):
            return reg
    return None


def slug(s: str) -> str:
    return "_".join(norm(s).split())


def build_registry() -> tuple[dict[str, dict], list[str]]:
    """stem -> {document_id, title, reference, regulator, date}; + liste d'avertissements."""
    records = load_records()
    stems = sorted(stem_of(p) for p in PDF_DIR.rglob("*.pdf"))
    registry, problems = {}, []
    for stem, hit in match_all(records, stems).items():
        if hit is None:
            problems.append(f"AUCUNE entrée JSON pour {stem} -> métadonnées minimales (à ajouter au JSON)")
            rec = {}
        else:
            rec = records[hit[0]]
        regulator = clean(rec.get("issuer"))
        if regulator:
            regulator = REGULATOR_ALIASES.get(regulator.lower(), regulator)
        else:
            regulator = infer_regulator(stem)
            problems.append(f"regulator absent du JSON, déduit du nom de fichier: {stem} -> {regulator}")
        registry[stem] = {
            "document_id": clean(rec.get("document_id")) or slug(stem),
            "title": clean(rec.get("title")) or stem.replace("_", " "),
            "reference": clean(rec.get("reference")) or "",
            "regulator": regulator or "",
            "date": clean(rec.get("publication_date")) or clean(rec.get("version_date")) or "",
        }
    return registry, problems


# ---------------------------------------------------------------- enrichissement
def enrich(registry: dict[str, dict], collection: str = COLLECTION) -> int:
    col = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(collection)
    n, offset = 0, 0
    while True:
        got = col.get(limit=BATCH, offset=offset, include=["metadatas"])
        if not got["ids"]:
            return n
        ids, metas = [], []
        for cid, m in zip(got["ids"], got["metadatas"]):
            info = registry.get(m.get("file_stem") or m["source"])
            if info is None:
                raise SystemExit(f"Chunk sans entrée de registre: {m['source']}")
            new = {**m, **info, "source": info["document_id"], "file_stem": m.get("file_stem") or m["source"]}
            # la date du JSON prime sur la date de création du PDF quand elle existe
            if not info["date"]:
                new["date"] = m.get("date", "")
            ids.append(cid)
            metas.append(new)
        col.update(ids=ids, metadatas=metas)
        n += len(ids)
        offset += BATCH


def citation(meta: dict) -> str:
    """Libellé de citation construit uniquement depuis les champs enrichis."""
    ref = f" ({meta['reference']})" if meta.get("reference") else ""
    art = f", {meta['article']}" if meta.get("article") else ""
    return f"{meta['regulator']} — {meta['title']}{ref}{art}, p. {meta['page']} [{meta['document_id']}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="affiche l'appariement sans modifier")
    ap.add_argument("--collection", default=COLLECTION)
    args = ap.parse_args()

    registry, problems = build_registry()
    for stem, info in registry.items():
        print(f"{stem}\n   -> {info['document_id']} | {info['regulator']} | {info['reference']} | {info['title'][:60]}")
    if problems:
        print("\nÀ VÉRIFIER :")
        print("\n".join(" - " + p for p in problems))
    if args.check:
        return
    print(f"\n{enrich(registry, args.collection)} chunks enrichis.")


if __name__ == "__main__":
    main()
