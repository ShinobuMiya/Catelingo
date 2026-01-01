# src/models.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union


# ============================================================
# Core input representation
# ============================================================

@dataclass(frozen=True)
class Fact:
    predicate: str
    args: Mapping[str, Any] = ()

@dataclass(frozen=True)
class InputStructure:
    facts: Tuple[Fact, ...]
    profile: Optional[str] = None


# ============================================================
# Values used inside constraints (must_hold args)
# ============================================================

@dataclass(frozen=True)
class Literal:
    value: Union[str, int, float, bool, None]

@dataclass(frozen=True)
class RoleRef:
    role: str

@dataclass(frozen=True)
class TypeRef:
    type_name: str

@dataclass(frozen=True)
class FactArgRef:
    # engine.py expects fields: var, arg
    var: str
    arg: str = "__self__"

@dataclass(frozen=True)
class AttrRef:
    base: "Value"
    attr: str

Value = Union[Literal, RoleRef, TypeRef, FactArgRef, AttrRef]

@dataclass(frozen=True)
class TypeDef:
    name: str
    isa: Tuple[str, ...] = ()

# ============================================================
# must_hold checks
# ============================================================

class CheckOp(str, Enum):
    # comparisons
    PRESENT = "present"
    EQ = "eq"
    NE = "ne"
    GE = "ge"
    GT = "gt"
    LE = "le"
    LT = "lt"

    # range / typing
    IN_RANGE = "in_range"
    TYPE_IS = "type_is"

    # units
    UNIT_ALLOWED = "unit_allowed"
    NORMALIZE = "normalize"


@dataclass(frozen=True)
class Check:
    op: CheckOp
    args: Tuple[Value, ...]
    note: Optional[str] = None


# ============================================================
# when clause
# ============================================================

@dataclass(frozen=True)
class WhenAtom:
    predicate: str
    args: Mapping[str, str] = ()

@dataclass(frozen=True)
class WhenClause:
    # engine.py expects "when.all_of"
    all_of: Tuple[WhenAtom, ...] = ()


# ============================================================
# constraints + spec
# ============================================================

@dataclass(frozen=True)
class ConstraintDef:
    constraint_id: str
    when: Tuple[WhenClause, ...] = ()
    must_hold: Tuple[Check, ...] = ()
    tags: Tuple[str, ...] = ()
    note: Optional[str] = None


@dataclass(frozen=True)
class UnitFamilyDef:
    name: str
    normalize_to: str
    allowed: Tuple[str, ...] = ()
    # engine expects keys like "percent->unit_interval"
    conversions: Mapping[str, str] = ()


# ============================================================
# Lexical senses / profiles (minimal shape needed by engine)
# ============================================================

@dataclass(frozen=True)
class RoleDef:
    name: str
    type: str = "ANY"
    optional: bool = False

@dataclass(frozen=True)
class SenseDef:
    sense_id: str
    lemma: str
    pos: str
    surface_forms: Tuple[str, ...] = ()
    predicate: str = ""
    returns: Optional[str] = None
    roles: Tuple[RoleDef, ...] = ()
    constraints: Tuple[ConstraintDef, ...] = ()
    tags: Tuple[str, ...] = ()
    note: Optional[str] = None

@dataclass(frozen=True)
class ProfileDef:
    name: str
    inherits: Tuple[str, ...] = ()
    enable_relaxation_rules: Tuple[str, ...] = ()
    strict_metaphor: bool = False

@dataclass(frozen=True)
class RelaxationRuleDef:
    rule_id: str
    from_type: str
    to_type: str
    only_lemmas: Tuple[str, ...] = ()
    note: Optional[str] = None

@dataclass(frozen=True)
class ValidatorSpec:
    unit_families: Mapping[str, UnitFamilyDef] = ()
    senses: Mapping[str, SenseDef] = ()
    profile_constraints_by_profile: Mapping[str, Tuple[ConstraintDef, ...]] = ()
    profiles: Mapping[str, ProfileDef] = ()
    types: Mapping[str, TypeDef] = ()
    relaxation_rules: Mapping[str, RelaxationRuleDef] = ()

    @property
    def profile_constraints(self) -> Tuple[ConstraintDef, ...]:
        # Flatten all profile constraints into one list for the toy engine
        out = []
        for _, cs in self.profile_constraints_by_profile.items():
            out.extend(list(cs))
        return tuple(out)

