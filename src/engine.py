from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .models import (
    AttrRef,
    Check,
    CheckOp,
    ConstraintDef,
    Fact,
    FactArgRef,
    InputStructure,
    Literal,
    ProfileDef,
    RoleRef,
    TypeRef,
    UnitFamilyDef,
    ValidatorSpec,
    Value,
)

# ============================================================
# 3-valued outcome
# ============================================================


class Tri(Enum):
    SAT = "SAT"
    UNSAT = "UNSAT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class CheckResult:
    tri: Tri
    op: str
    args_repr: Tuple[str, ...]
    note: Optional[str] = None
    debug: Optional[str] = None


@dataclass(frozen=True)
class ConstraintResult:
    constraint_id: str
    tri: Tri
    checks: Tuple[CheckResult, ...]
    note: Optional[str] = None
    # Explicitly mark this as a local constraint evaluation
    locality: str = "local"


@dataclass(frozen=True)
class VerificationResult:
    tri: Tri  # overall verdict
    profile: str
    applied: Tuple[ConstraintResult, ...]
    ignored: Tuple[str, ...]  # constraint_ids not triggered by when-clause
    penalty_score: float  # toy scalar: 0 if SAT else >0
    summary: str


# ============================================================
# Engine
# ============================================================


class CatelingoEngine:
    """
    Core engine:
      - Takes extracted facts (InputStructure.facts)
      - Evaluates constraints (profile_constraints + sense constraints if matched)
      - Uses 3-valued semantics:
          UNSAT: definite violation
          UNKNOWN: insufficient / upstream parsing-normalization ambiguity
          SAT: satisfied

    Design alignment with Catelingo paper / validator.yaml:
      - `when` clauses are *not* boolean triggers. They produce a set of
        variable bindings (environments) via pattern matching / unification.
      - `must_hold` is evaluated *under* each environment.
      - No hidden KB: facts like WWI=1914 must be provided by the input layer.
      - This toy engine does NOT perform global CSP across senses; it assumes
        the scenario already provides canonical facts.

    NOTE:
      - This implementation treats pattern variables as strings starting with "?".
        Example: "?X", "?T_construct", etc.
      - These variables are bound into `env` as scalars (or dicts) and referenced
        by `FactArgRef(var="?X", arg="__self__")` style Values produced by loader.
    """

    def __init__(self, spec: ValidatorSpec):
        self.spec = spec
        self._active_profile: ProfileDef = ProfileDef(name="general1")
        self._applied_relaxations: set[str] = set()

    def verify(self, inp: InputStructure | Mapping[str, Any], profile: Optional[str] = None) -> VerificationResult:
        # Accept both InputStructure and raw dict (scenario JSON).
        if isinstance(inp, dict):
            facts_raw = inp.get("facts", []) or []
            facts: List[Fact] = []
            for fr in facts_raw:
                if isinstance(fr, Fact):
                    facts.append(fr)
                elif isinstance(fr, dict):
                    facts.append(Fact(predicate=str(fr.get("predicate", "")), args=fr.get("args", {}) or {}))
            inp = InputStructure(
                facts=tuple(facts),
                profile=str(inp.get("profile")) if inp.get("profile") is not None else None,
            )

        prof_name = profile or (inp.profile or "general1")
        self._active_profile = self._resolve_profile(prof_name)
        self._applied_relaxations = set()

        fact_index: Dict[str, List[Fact]] = {}
        for f in inp.facts:
            fact_index.setdefault(f.predicate, []).append(f)
        # Collect constraints:
        #   (a) global profile_constraints
        #   (b) sense constraints: if a fact predicate matches a sense.predicate, apply those constraints too
        all_constraints: List[ConstraintDef] = []
        all_constraints.extend(self.spec.profile_constraints)

        pred_to_senses = self._predicate_index()
        for f in inp.facts:
            for sid in pred_to_senses.get(f.predicate, ()):
                sdef = self.spec.senses[sid]
                all_constraints.extend(list(sdef.constraints))

        applied_results: List[ConstraintResult] = []
        ignored: List[str] = []

        for c in all_constraints:
            envs = self._when_match_envs(c.when[0], fact_index) if c.when else [dict()]
            if not envs:
                ignored.append(c.constraint_id)
                continue
            cres = self._eval_constraint_for_envs(c, inp.facts, envs)
            applied_results.append(cres)

        for rid in sorted(self._applied_relaxations):
            applied_results.append(
                ConstraintResult(
                    constraint_id=rid,
                    tri=Tri.SAT,
                    checks=(),
                    note="relaxation rule applied",
                    locality="relaxation",
                )
            )

        overall = self._aggregate_overall(applied_results)
        penalty_score = self._toy_penalty_score(applied_results)
        summary = self._summary(overall, applied_results)

        return VerificationResult(
            tri=overall,
            profile=prof_name,
            applied=tuple(applied_results),
            ignored=tuple(ignored),
            penalty_score=penalty_score,
            summary=summary,
        )

    # ============================================================
    # When: pattern matching with variable binding
    # ============================================================

# ============================================================
    # When: pattern matching with variable binding (Optimized)
    # ============================================================

    def _when_match_envs(self, when, fact_index: Mapping[str, List[Fact]]) -> List[Dict[str, Any]]:
        """
        Executes pattern matching using the fact index to avoid O(n^2) behavior.
        Complexity: O(atoms * candidates), effectively linear to relevant edges.
        """
        envs: List[Dict[str, Any]] = [dict()]

        for atom in when.all_of:
            new_envs: List[Dict[str, Any]] = []
            
            candidates = fact_index.get(atom.predicate, [])
            
            for env in envs:
                for f in candidates:
                    bound = self._unify_atom_with_fact(atom.args, f, env)
                    if bound is not None:
                        new_envs.append(bound)
            envs = new_envs
            if not envs:
                break

        return envs

    def _unify_atom_with_fact(self, pattern, fact, env):
        out = dict(env)

        for k, pv in pattern.items():
            has_key = k in fact.args
            av = fact.args.get(k, None)

            if isinstance(pv, str) and pv.startswith("?"):
                if not has_key:
                    if pv in out and out[pv] is not None:
                        return None
                    out.setdefault(pv, None)
                    continue

                if pv in out:
                    if out[pv] != av:
                        return None
                else:
                    out[pv] = av
            else:
                if not has_key:
                    return None
                if str(av) != str(pv):
                    return None

        return out


    # ============================================================
    # Constraint evaluation
    # ============================================================

    def _eval_constraint_for_envs(
        self,
        c: ConstraintDef,
        facts: Tuple[Fact, ...],
        envs: List[Dict[str, Any]],
    ) -> ConstraintResult:
        final_tri = Tri.SAT
        all_checks: List[CheckResult] = []

        for base_env in envs:
            env = dict(base_env)

            local_tri = Tri.SAT
            for chk in c.must_hold:
                r = self._eval_check(chk, facts, env)
                all_checks.append(r)

                if r.tri == Tri.UNSAT:
                    local_tri = Tri.UNSAT
                    break
                if r.tri == Tri.UNKNOWN and local_tri != Tri.UNSAT:
                    local_tri = Tri.UNKNOWN

            if local_tri == Tri.UNSAT:
                final_tri = Tri.UNSAT
                break
            if local_tri == Tri.UNKNOWN and final_tri != Tri.UNSAT:
                final_tri = Tri.UNKNOWN

        return ConstraintResult(
            constraint_id=c.constraint_id,
            tri=final_tri,
            checks=tuple(all_checks),
            note=c.note,
        )

    def _eval_check(self, chk: Check, facts: Tuple[Fact, ...], env: Dict[str, Any]) -> CheckResult:
        op = chk.op.value
        args = chk.args

        resolved: List[Any] = []
        args_repr: List[str] = []
        for idx, a in enumerate(args):
            args_repr.append(self._repr_value(a))

            if chk.op == CheckOp.NORMALIZE and idx == 3:
                if isinstance(a, FactArgRef) and a.arg == "__self__" and a.var.startswith("?"):
                    resolved.append(a.var)
                    continue
                if isinstance(a, Literal) and isinstance(a.value, str) and a.value.startswith("?"):
                    resolved.append(a.value)
                    continue
                return CheckResult(
                    tri=Tri.UNKNOWN,
                    op=op,
                    args_repr=tuple(args_repr),
                    note=chk.note,
                    debug="normalize: out var must be like '?p_norm'",
                )

            v, tri = self._resolve_value(a, facts, env)
            if tri == Tri.UNKNOWN:
                return CheckResult(
                    tri=Tri.UNKNOWN,
                    op=op,
                    args_repr=tuple(args_repr),
                    note=chk.note,
                    debug=f"arg '{self._repr_value(a)}' unresolved",
                )
            resolved.append(v)

        tri, debug = self._eval_op(chk.op, resolved, env)
        return CheckResult(
            tri=tri,
            op=op,
            args_repr=tuple(args_repr),
            note=chk.note,
            debug=debug,
        )

    # ============================================================
    # Operators (CheckOp)
    # ============================================================

    def _eval_op(self, op: CheckOp, args: List[Any], env: Dict[str, Any]) -> Tuple[Tri, Optional[str]]:
        # ---- presence
        if op == CheckOp.PRESENT:
            for a in args:
                if a is None or a == "":
                    return (Tri.UNKNOWN, "missing required value")
            return (Tri.SAT, None)

        # ---- comparisons
        if op == CheckOp.EQ:
            return (Tri.SAT if args[0] == args[1] else Tri.UNSAT, None)
        if op == CheckOp.NE:
            return (Tri.SAT if args[0] != args[1] else Tri.UNSAT, None)
        if op == CheckOp.GE:
            return self._cmp(args, lambda a, b: a >= b, ">=")
        if op == CheckOp.GT:
            return self._cmp(args, lambda a, b: a > b, ">")
        if op == CheckOp.LE:
            return self._cmp(args, lambda a, b: a <= b, "<=")
        if op == CheckOp.LT:
            return self._cmp(args, lambda a, b: a < b, "<")

        # ---- range
        if op == CheckOp.IN_RANGE:
            x, lo, hi = args[0], args[1], args[2]
            try:
                ok = (x >= lo) and (x <= hi)
                return (Tri.SAT if ok else Tri.UNSAT, None)
            except Exception as e:
                return (Tri.UNKNOWN, f"in_range type error: {e!r}")

        # ---- units
        if op == CheckOp.UNIT_ALLOWED:
            unit, fam = args[0], args[1]
            if unit is None or unit == "":
                return (Tri.UNKNOWN, "unit missing")
            if not isinstance(unit, str):
                return (Tri.UNKNOWN, f"unit not a string: {unit!r}")

            fam_key = str(fam).split(".", 1)[0]     # ★追加
            fam_def = self.spec.unit_families.get(fam_key)
            if fam_def is None:
                return (Tri.UNKNOWN, f"unknown unit family: {fam!r}")

            return (Tri.SAT if unit in fam_def.allowed else Tri.UNKNOWN, f"unit={unit!r}")
        if op == CheckOp.NORMALIZE:
            val, unit, fam, out_name = args[0], args[1], args[2], args[3]

            if unit is None or unit == "":
                return (Tri.UNKNOWN, "normalize: unit missing")
            if not isinstance(unit, str):
                return (Tri.UNKNOWN, "normalize: unit not string")

            fam_key = str(fam).split(".", 1)[0]
            fam_def = self.spec.unit_families.get(fam_key)
            if fam_def is None:
                return (Tri.UNKNOWN, f"normalize: unknown family {fam!r}")

            if unit not in fam_def.allowed:
                return (Tri.UNKNOWN, f"normalize: unit {unit!r} not allowed")

            try:
                normalized = self._convert_unit(val, unit, fam_def)
            except Exception as e:
                return (Tri.UNKNOWN, f"normalize: conversion failed: {e!r}")

            if not isinstance(out_name, str) or not out_name.startswith("?"):
                return (Tri.UNKNOWN, "normalize: out var must be like '?p_norm'")

            env[out_name] = normalized
            return (Tri.SAT, f"{out_name}={normalized!r}")

        if op == CheckOp.TYPE_IS:
            obj, expected = args[0], args[1]

            actual_type = None
            lexeme = None

            if isinstance(obj, dict) and "type" in obj:
                actual_type = obj.get("type")
                lexeme = obj.get("lexeme") or obj.get("lemma") or obj.get("surface")
            elif isinstance(obj, str):
                actual_type = obj

            if not isinstance(actual_type, str) or not actual_type:
                return (Tri.UNKNOWN, "type_is: object has no type info")

            ok, used_rule = self._is_subtype_or_relaxed(actual_type, expected, lexeme=lexeme)
            if used_rule is not None:
                self._applied_relaxations.add(used_rule)

            return (Tri.SAT if ok else Tri.UNSAT, f"type={actual_type!r} (expected {expected!r})")

        return (Tri.UNKNOWN, f"unhandled op: {op.value}")

    def _cmp(self, args: List[Any], fn, sym: str) -> Tuple[Tri, Optional[str]]:
        try:
            ok = fn(args[0], args[1])
            return (Tri.SAT if ok else Tri.UNSAT, None)
        except Exception as e:
            return (Tri.UNKNOWN, f"cmp {sym} type error: {e!r}")

    def _convert_unit(self, val, unit, fam):
        if val is None:
            raise ValueError("value missing")
        x = float(val)

        if unit == fam.normalize_to:
            return x

        key1 = f"{unit}->{fam.normalize_to}"
        key2 = f"{unit}_to_{fam.normalize_to}"
        expr = fam.conversions.get(key1) or fam.conversions.get(key2)

        if expr is None:
            raise ValueError(f"no conversion {unit} -> {fam.normalize_to}")

        return float(eval(expr, {"__builtins__": {}}, {"x": x}))


    # ============================================================
    # Value resolution
    # ============================================================

    def _resolve_value(self, v: Value, facts: Tuple[Fact, ...], env: Dict[str, Any]) -> Tuple[Any, Tri]:
        if isinstance(v, Literal):
            return v.value, Tri.SAT

        if isinstance(v, RoleRef):
            return None, Tri.UNKNOWN

        if isinstance(v, TypeRef):
            return v.type_name, Tri.SAT

        if isinstance(v, FactArgRef):
            if v.var in env:
                bound = env[v.var]
                if v.arg == "__self__":
                    return bound, Tri.SAT
                if isinstance(bound, Fact):
                    if v.arg in bound.args:
                        return bound.args[v.arg], Tri.SAT
                    return None, Tri.UNKNOWN
                if isinstance(bound, dict):
                    if v.arg in bound:
                        return bound[v.arg], Tri.SAT
                    return None, Tri.UNKNOWN
                if v.arg == "value":
                        return bound, Tri.SAT
                return None, Tri.UNKNOWN

            return None, Tri.UNKNOWN

        if isinstance(v, AttrRef):
            base_val, tri = self._resolve_value(v.base, facts, env)
            if tri != Tri.SAT:
                return None, Tri.UNKNOWN
            if isinstance(base_val, dict) and v.attr in base_val:
                return base_val[v.attr], Tri.SAT
            return None, Tri.UNKNOWN

        return None, Tri.UNKNOWN

    def _repr_value(self, v: Value) -> str:
        if isinstance(v, Literal):
            return repr(v.value)
        if isinstance(v, RoleRef):
            return f"${v.role}"
        if isinstance(v, TypeRef):
            return f"@TYPE:{v.type_name}"
        if isinstance(v, FactArgRef):
            return f"{v.var}.{v.arg}"
        if isinstance(v, AttrRef):
            return f"attr({self._repr_value(v.base)},{v.attr})"
        return str(v)

    # ============================================================
    # Profile resolution
    # ============================================================

    def _resolve_profile(self, name: str) -> ProfileDef:
        p = self.spec.profiles.get(name)
        if p is None:
            return ProfileDef(
                name=name,
                inherits=(),
                enable_relaxation_rules=(),
                strict_metaphor=False,
            )

        visited = set()
        order: List[str] = []

        def dfs(x: str) -> None:
            if x in visited:
                return
            visited.add(x)
            px = self.spec.profiles.get(x)
            if px is None:
                return
            for parent in px.inherits:
                dfs(parent)
            order.append(x)

        dfs(name)

        strict = False
        enabled: List[str] = []
        for pn in order:
            px = self.spec.profiles.get(pn)
            if px is None:
                continue
            strict = strict or px.strict_metaphor
            enabled.extend(list(px.enable_relaxation_rules))

        return ProfileDef(
            name=name,
            inherits=tuple(self.spec.profiles[name].inherits),
            enable_relaxation_rules=tuple(dict.fromkeys(enabled).keys()),
            strict_metaphor=strict,
        )

    # ============================================================
    # Indexes
    # ============================================================

    def _predicate_index(self) -> Mapping[str, Tuple[str, ...]]:
        out: Dict[str, List[str]] = {}
        for sid, s in self.spec.senses.items():
            if not s.predicate:
                continue
            out.setdefault(s.predicate, []).append(sid)
        return {k: tuple(v) for k, v in out.items()}

    # ============================================================
    # Aggregation & penalty_score (toy)
    # ============================================================

    def _aggregate_overall(self, results: Sequence[ConstraintResult]) -> Tri:
        seen_unknown = False
        for r in results:
            if r.tri == Tri.UNSAT:
                return Tri.UNSAT
            if r.tri == Tri.UNKNOWN:
                seen_unknown = True
        return Tri.UNKNOWN if seen_unknown else Tri.SAT

    def _toy_penalty_score(self, results: Sequence[ConstraintResult]) -> float:
        penalty_score = 0.0
        for r in results:
            if r.tri == Tri.UNKNOWN:
                penalty_score += 1.0
            elif r.tri == Tri.UNSAT:
                penalty_score += 100.0
        return penalty_score

    def _summary(self, overall: Tri, results: Sequence[ConstraintResult]) -> str:
            unsat = sum(1 for r in results if r.tri == Tri.UNSAT)
            unk = sum(1 for r in results if r.tri == Tri.UNKNOWN)
            sat = sum(1 for r in results if r.tri == Tri.SAT)

            if overall == Tri.UNSAT:
                status_msg = "local violations detected; global coherence unattainable"
            elif overall == Tri.SAT:
                status_msg = "all constraints satisfied; global coherence established"
            else:
                status_msg = "insufficient information; global coherence undetermined"

            return f"{overall.value} ({status_msg}) [SAT={sat}, UNKNOWN={unk}, UNSAT={unsat}]"

    def _is_subtype(self, actual: str, expected: str) -> bool:
        if actual == expected:
            return True
        
        if actual not in self.spec.types:
            return False
            
        queue = [actual]
        visited = {actual}
        
        while queue:
            curr_name = queue.pop(0)
            curr_def = self.spec.types.get(curr_name)
            
            if not curr_def:
                continue
                
            for parent in curr_def.isa:
                if parent == expected:
                    return True
                if parent not in visited:
                    visited.add(parent)
                    queue.append(parent)
                    
        return False
    
    def _is_subtype_or_relaxed(self, actual: str, expected: str, *, lexeme: str | None = None) -> tuple[bool, str | None]:
        if self._is_subtype(actual, expected):
            return True, None

        if self._active_profile.strict_metaphor:
            return False, None

        for rid in self._active_profile.enable_relaxation_rules:
            rule = self.spec.relaxation_rules.get(rid)
            if rule is None:
                continue

            if rule.from_type != actual:
                continue

            if rule.only_lemmas:
                if lexeme is None or lexeme not in rule.only_lemmas:
                    continue

            if self._is_subtype(rule.to_type, expected) or rule.to_type == expected:
                return True, rid

        return False, None

