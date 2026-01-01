# src/loader.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

from .models import (
    Check,
    CheckOp,
    ConstraintDef,
    RoleDef,
    SenseDef,
    UnitFamilyDef,
    ValidatorSpec,
    WhenAtom,
    WhenClause,
    Value,
    Literal,
    FactArgRef,
    TypeDef,
    ProfileDef,
    RelaxationRuleDef,
)


# -----------------------------
# Public API
# -----------------------------

def load_validator_yaml(path: str | Path) -> ValidatorSpec:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("validator.yaml must be a mapping at top-level")

    units = _load_units(raw.get("units", {}))
    types = _load_types(raw.get("types", {}))
    senses = _load_senses(raw.get("senses", []), units=units)
    profile_constraints = _load_profile_constraints(raw.get("profile_constraints", {}), units=units)
    profiles = _load_profiles(raw.get("profiles", {}))
    relax = _load_relaxation_rules(raw.get("relaxation_rules", {}))

    return ValidatorSpec(
        unit_families=units,
        types=types,
        senses=senses,
        profile_constraints_by_profile=profile_constraints,
        profiles=profiles,
        relaxation_rules=relax
    )


def load_scenario_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("scenario JSON must be a top-level object")
    return data


# -----------------------------
# Units
# -----------------------------

def _load_units(raw: Any) -> Mapping[str, UnitFamilyDef]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("units must be a mapping: {unit_family: {...}}")

    out: Dict[str, UnitFamilyDef] = {}
    for uname, body in raw.items():
        name = str(uname)
        if not isinstance(body, dict):
            raise ValueError(f"units.{name} must be mapping")

        normalize_to = body.get("normalize_to")
        if not isinstance(normalize_to, str) or not normalize_to:
            raise ValueError(f"units.{name}.normalize_to must be non-empty string")

        allowed = body.get("allowed", []) or []
        if not isinstance(allowed, list) or not all(isinstance(x, str) for x in allowed):
            raise ValueError(f"units.{name}.allowed must be list[str]")

        conv_raw = body.get("conversions", {}) or {}
        if not isinstance(conv_raw, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in conv_raw.items()):
            raise ValueError(f"units.{name}.conversions must be mapping[str,str]")

        # engine expects "from->to" keys; yaml may use "percent_to_unit_interval"
        conv: Dict[str, str] = {}
        for k, expr in conv_raw.items():
            kk = k.strip()
            if "_to_" in kk:
                a, b = kk.split("_to_", 1)
                kk = f"{a}->{b}"
            conv[kk] = expr

        out[name] = UnitFamilyDef(
            name=name,
            normalize_to=normalize_to,
            allowed=tuple(allowed),
            conversions=conv,
        )
    return out


# -----------------------------
# Senses
# -----------------------------

def _load_senses(raw: Any, *, units: Mapping[str, UnitFamilyDef]) -> Mapping[str, SenseDef]:
    if raw is None:
        return {}
    if not isinstance(raw, list):
        raise ValueError("senses must be a list")

    out: Dict[str, SenseDef] = {}
    for i, body in enumerate(raw):
        if not isinstance(body, dict):
            raise ValueError(f"senses[{i}] must be mapping")

        sid = body.get("sense_id") or body.get("id") or f"SENSE_{i:03d}"
        if not isinstance(sid, str) or not sid.strip():
            raise ValueError(f"senses[{i}].sense_id must be non-empty string")
        sid = sid.strip()

        lemma = body.get("lemma", "")
        pos = body.get("pos", body.get("POS", ""))
        if not isinstance(lemma, str) or not isinstance(pos, str):
            raise ValueError(f"senses[{i}].lemma/pos must be strings")

        surface_forms = body.get("surface_forms", []) or []
        if not isinstance(surface_forms, list) or not all(isinstance(x, str) for x in surface_forms):
            raise ValueError(f"senses[{i}].surface_forms must be list[str]")

        predicate = body.get("predicate", "") or ""
        returns = body.get("returns", None)

        # roles
        roles_raw = body.get("roles", {}) or {}
        roles: List[RoleDef] = []

        if isinstance(roles_raw, dict):
            for rname, rdef in roles_raw.items():
                if not isinstance(rname, str) or not rname:
                    raise ValueError(f"senses[{i}].roles has invalid role name")
                if not isinstance(rdef, dict):
                    raise ValueError(f"senses[{i}].roles.{rname} must be mapping")
                rtype = rdef.get("type", "ANY")
                optional = bool(rdef.get("optional", False))
                roles.append(RoleDef(name=rname, type=str(rtype), optional=optional))
        elif isinstance(roles_raw, list):
            for rj, r in enumerate(roles_raw):
                if not isinstance(r, dict):
                    raise ValueError(f"senses[{i}].roles[{rj}] must be mapping")
                rname = r.get("name")
                if not isinstance(rname, str) or not rname:
                    raise ValueError(f"senses[{i}].roles[{rj}].name must be non-empty string")
                rtype = r.get("type", "ANY")
                optional = bool(r.get("optional", False))
                roles.append(RoleDef(name=rname, type=str(rtype), optional=optional))

        else:
            raise ValueError(f"senses[{i}].roles must be dict or list")


        # inline constraints (same schema as profile_constraints entries)
        constraints_raw = body.get("constraints", []) or []
        if not isinstance(constraints_raw, list):
            raise ValueError(f"senses[{i}].constraints must be list")
        constraints: List[ConstraintDef] = []
        for cj, cbody in enumerate(constraints_raw):
            if not isinstance(cbody, dict):
                raise ValueError(f"senses[{i}].constraints[{cj}] must be mapping")
            constraints.append(_load_constraint_inline(f"sense::{sid}", cj, cbody, units=units))

        tags_raw = body.get("tags", []) or []
        if isinstance(tags_raw, str):
            tags_raw = [tags_raw]
        if not isinstance(tags_raw, list) or not all(isinstance(x, str) for x in tags_raw):
            raise ValueError(f"senses[{i}].tags must be list[str]")

        note = body.get("note", None)
        if note is not None and not isinstance(note, str):
            raise ValueError(f"senses[{i}].note must be string")

        out[sid] = SenseDef(
            sense_id=sid,
            lemma=lemma,
            pos=pos,
            surface_forms=tuple(surface_forms),
            predicate=str(predicate),
            returns=str(returns) if returns is not None else None,
            roles=tuple(roles),
            constraints=tuple(constraints),
            tags=tuple(tags_raw),
            note=note,
        )
    return out


# -----------------------------
# Profile constraints
# -----------------------------

def _load_profile_constraints(raw: Any, units: Mapping[str, UnitFamilyDef]) -> Mapping[str, Tuple[ConstraintDef, ...]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("profile_constraints must be mapping[str, list[constraint]]")

    out: Dict[str, Tuple[ConstraintDef, ...]] = {}
    for pname, lst in raw.items():
        if not isinstance(lst, list):
            raise ValueError(f"profile_constraints.{pname} must be a list")
        constraints: List[ConstraintDef] = []
        for i, cbody in enumerate(lst):
            if not isinstance(cbody, dict):
                raise ValueError(f"profile_constraints.{pname}[{i}] must be mapping")
            constraints.append(_load_constraint_inline(f"profile::{pname}", i, cbody, units=units))
        out[str(pname)] = tuple(constraints)

    return out


# -----------------------------
# Constraint parsing
# -----------------------------

def _load_constraint_inline(
    owner: str,
    idx: int,
    body: Mapping[str, Any],
    *,
    units: Optional[Mapping[str, UnitFamilyDef]] = None,
) -> ConstraintDef:
    cid = str(body.get("id", f"{owner}::C_{idx:03d}"))

    when_raw = body.get("when", None)
    when = _load_when(when_raw)

    must_hold_raw = body.get("must_hold", [])
    must_hold = _load_must_hold(must_hold_raw, units=units)

    tags_raw = body.get("tags", ()) or ()
    if isinstance(tags_raw, str):
        tags_raw = (tags_raw,)
    if not isinstance(tags_raw, (list, tuple)) or not all(isinstance(x, str) for x in tags_raw):
        raise ValueError(f"{cid}.tags must be list[str] or tuple[str]")

    note = body.get("note", None)
    if note is not None and not isinstance(note, str):
        raise ValueError(f"{cid}.note must be string")

    return ConstraintDef(
        constraint_id=cid,
        when=when,
        must_hold=tuple(must_hold),
        tags=tuple(tags_raw),
        note=note,
    )


def _load_when(raw: Any) -> Tuple[WhenClause, ...]:
    """
    Accepts the validator.yaml schema:

      when:
        all:
          - predicate: "construct"
            args: {theme: "?X", time: "?T_construct"}
          - predicate: "commemorate"
            args: {carrier: "?X", event: "?E", event_time: "?T_event"}

    Also accepts list-form directly (treat as "all").
    """
    if raw is None or raw == {}:
        return ()

    atoms_raw = None
    if isinstance(raw, dict):
        atoms_raw = raw.get("all", None)
        if atoms_raw is None:
            # permissive: allow single atom dict {predicate,args}
            if "predicate" in raw:
                atoms_raw = [raw]
            else:
                return ()
    elif isinstance(raw, list):
        atoms_raw = raw
    else:
        raise ValueError("when must be mapping or list")

    if not isinstance(atoms_raw, list):
        raise ValueError("when.all must be a list")

    atoms: List[WhenAtom] = []
    for i, a in enumerate(atoms_raw):
        if not isinstance(a, dict):
            raise ValueError(f"when.all[{i}] must be mapping")
        pred = a.get("predicate")
        if not isinstance(pred, str) or not pred:
            raise ValueError(f"when.all[{i}].predicate must be non-empty string")
        args = a.get("args", {}) or {}
        if not isinstance(args, dict) or not all(isinstance(k, str) and isinstance(v, (str, int, float, bool)) for k, v in args.items()):
            raise ValueError(f"when.all[{i}].args must be mapping[str, scalar]")
        # engine expects Mapping[str,str] (pattern vars are strings like "?X")
        atoms.append(WhenAtom(predicate=pred, args={str(k): str(v) for k, v in args.items()}))

    return (WhenClause(all_of=tuple(atoms)),)


def _load_must_hold(raw: Any, *, units: Optional[Mapping[str, UnitFamilyDef]] = None) -> List[Check]:
    checks: List[Check] = []
    if raw is None:
        return checks

    # list form: [{op: "...", args: [...]}, ...]
    if isinstance(raw, list):
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ValueError(f"must_hold[{i}] must be mapping")
            op = item.get("op")
            args = item.get("args", [])
            if not isinstance(op, str) or not op:
                raise ValueError(f"must_hold[{i}].op must be non-empty string")
            if not isinstance(args, list):
                raise ValueError(f"must_hold[{i}].args must be list")
            checks.append(Check(op=CheckOp(str(op)), args=tuple(_parse_value(a, units=units) for a in args)))
        return checks

    # dict shorthand: {ge: ["?a","?b"], in_range: ["?x",0,1]}
    if isinstance(raw, dict):
        for op, args in raw.items():
            if not isinstance(op, str) or not op:
                raise ValueError("must_hold dict keys must be op strings")
            if not isinstance(args, list):
                raise ValueError(f"must_hold.{op} must be list")
            checks.append(Check(op=CheckOp(op), args=tuple(_parse_value(a, units=units) for a in args)))
        return checks

    raise ValueError("must_hold must be list[check] or dict[op -> args]")


# -----------------------------
# Value parsing for must_hold args
# -----------------------------

def _parse_value(spec: Any, *, units: Optional[Mapping[str, UnitFamilyDef]] = None) -> Value:
    # numbers / bool / null
    if isinstance(spec, (int, float, bool)) or spec is None:
        return Literal(spec)

    # string tokens:
    if isinstance(spec, str):
        s = spec.strip()
        # Catelingo paper style: bare "value"/"unit" refers to current fact argument
        if s in ("value", "unit"):
            # Reserved variable "?$" will be bound to the current fact by the engine.
            # Interpret as "?$.value" / "?$.unit"
            return FactArgRef(var="?$", arg=s)

        # pattern vars: "?X" or "?X.field"
        if s.startswith("?"):
            if "." in s:
                var, field = s.split(".", 1)
                return FactArgRef(var=var, arg=field)
            return FactArgRef(var=s, arg="__self__")

        # unit-family refs used in validator.yaml:
        # "probability.allowed" / "probability.normalize_to"  -> family name "probability"
        if "." in s and not s.startswith("$"):
            head, tail = s.split(".", 1)
            if units is None or head in units:
                if tail in ("allowed", "normalize_to") or tail.startswith("conversions"):
                    return Literal(head)

        # "$units.probability.allowed" -> family name "probability"
        if s.startswith("$units."):
            parts = s.split(".")
            if len(parts) >= 3:
                fam = parts[1]
                return Literal(fam)

        # fallback literal string
        return Literal(spec)

    # dict form is reserved for future; treat as stringified literal now
    if isinstance(spec, dict):
        return Literal(str(spec))

    return Literal(str(spec))

def _load_types(raw: Any) -> Mapping[str, TypeDef]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("types must be a mapping")

    out: Dict[str, TypeDef] = {}
    for name, body in raw.items():
        if not isinstance(body, dict):
            raise ValueError(f"types.{name} must be mapping")
        
        isa_raw = body.get("isa", [])
        if not isinstance(isa_raw, list):
            raise ValueError(f"types.{name}.isa must be list")
            
        out[str(name)] = TypeDef(
            name=str(name),
            isa=tuple(str(x) for x in isa_raw)
        )
    return out

def _load_profiles(raw: Any) -> Mapping[str, ProfileDef]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("profiles must be a mapping")

    out: Dict[str, ProfileDef] = {}
    for name, body in raw.items():
        if not isinstance(body, dict):
            raise ValueError(f"profiles.{name} must be mapping")
        inherits = body.get("inherits", []) or []
        enable = body.get("enable_relaxation_rules", []) or []
        strict = bool(body.get("strict_metaphor", False))
        if not isinstance(inherits, list) or not all(isinstance(x, str) for x in inherits):
            raise ValueError(f"profiles.{name}.inherits must be list[str]")
        if not isinstance(enable, list) or not all(isinstance(x, str) for x in enable):
            raise ValueError(f"profiles.{name}.enable_relaxation_rules must be list[str]")
        out[str(name)] = ProfileDef(
            name=str(name),
            inherits=tuple(inherits),
            enable_relaxation_rules=tuple(enable),
            strict_metaphor=strict,
        )
    return out

def _load_relaxation_rules(raw: Any) -> Mapping[str, RelaxationRuleDef]:
    if raw is None:
        return {}
    if isinstance(raw, list):
        # list[{id, from, to, only_lemmas?}]
        items = raw
    elif isinstance(raw, dict):
        # mapping[id -> {...}]
        items = []
        for rid, body in raw.items():
            if isinstance(body, dict):
                b = dict(body)
                b.setdefault("id", rid)
                items.append(b)
            else:
                raise ValueError(f"relaxation_rules.{rid} must be mapping")
    else:
        raise ValueError("relaxation_rules must be list or mapping")

    out: Dict[str, RelaxationRuleDef] = {}
    for i, body in enumerate(items):
        if not isinstance(body, dict):
            raise ValueError(f"relaxation_rules[{i}] must be mapping")
        rid = body.get("id")
        frm = body.get("from") or body.get("from_type")
        to = body.get("to") or body.get("to_type")
        if not isinstance(rid, str) or not rid:
            raise ValueError(f"relaxation_rules[{i}].id must be non-empty string")
        if not isinstance(frm, str) or not frm:
            raise ValueError(f"{rid}: from_type must be non-empty string")
        if not isinstance(to, str) or not to:
            raise ValueError(f"{rid}: to_type must be non-empty string")

        only = body.get("only_lemmas", []) or []
        if isinstance(only, str):
            only = [only]
        if not isinstance(only, list) or not all(isinstance(x, str) for x in only):
            raise ValueError(f"{rid}: only_lemmas must be list[str]")

        note = body.get("note", None)
        if note is not None and not isinstance(note, str):
            raise ValueError(f"{rid}: note must be string")

        out[rid] = RelaxationRuleDef(
            rule_id=rid,
            from_type=frm,
            to_type=to,
            only_lemmas=tuple(only),
            note=note,
        )
    return out
