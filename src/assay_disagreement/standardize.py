"""Deterministic chemical identity and scaffold helpers."""

from __future__ import annotations

import hashlib
import re

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


KNOWN_CONTEXT_RE = re.compile(r"\b(cell[- ]based|whole[- ]cell|organism|tissue|in vivo|animal)\b", re.I)
VARIANT_RE = re.compile(r"\b(mutant|mutation|variant|truncated|fusion|chimeric|recombinant)\b", re.I)


def _hash(value: str, prefix: str) -> str:
    return prefix + hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def normalize_mol(smiles: str):
    if not isinstance(smiles, str) or not smiles.strip():
        return None, "missing_smiles"
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "invalid_smiles"
        Chem.SanitizeMol(mol)
    except Exception:
        return None, "invalid_smiles"
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    organic = [frag for frag in fragments if any(atom.GetAtomicNum() == 6 for atom in frag.GetAtoms())]
    if len(organic) == 0:
        return None, "no_organic_fragment"
    if len(organic) > 1:
        return None, "multiple_organic_fragments"
    kept = organic[0]
    if any(atom.GetAtomicNum() > 20 and atom.GetSymbol() not in {"Cl", "Br", "I"}
           for atom in kept.GetAtoms()):
        return None, "metal_or_unhandled_element"
    return kept, "ok" if len(fragments) == 1 else "counterion_removed"


def identity_fields(smiles: str) -> dict[str, str | None]:
    mol, reason = normalize_mol(smiles)
    if mol is None:
        return {"canonical_isomeric_smiles": None, "chemical_identity": None,
                "chemical_family": None, "scaffold": None, "standardization_status": reason}
    isomeric = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    family_mol = Chem.Mol(mol)
    Chem.RemoveStereochemistry(family_mol)
    family_smiles = Chem.MolToSmiles(family_mol, canonical=True, isomericSmiles=False)
    scaffold_mol = MurckoScaffold.GetScaffoldForMol(family_mol)
    scaffold = Chem.MolToSmiles(scaffold_mol, canonical=True, isomericSmiles=False)
    # Unique fallback prevents all acyclic molecules sharing one empty scaffold.
    scaffold_key = _hash(scaffold if scaffold else family_smiles, "scf_")
    return {"canonical_isomeric_smiles": isomeric,
            "chemical_identity": _hash(isomeric, "id_"),
            "chemical_family": _hash(family_smiles, "fam_"),
            "scaffold": scaffold_key, "standardization_status": reason}


def context_flags(description: str) -> dict[str, bool | str]:
    text = description if isinstance(description, str) else ""
    return {"known_cell_or_organism_context": bool(KNOWN_CONTEXT_RE.search(text)),
            "variant_signal": bool(VARIANT_RE.search(text)),
            "context_status": "known" if text else "unknown"}
