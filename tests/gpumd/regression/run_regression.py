#!/usr/bin/env python3
"""Run strict, two-executable GPUMD regression tests.

Each selected case is run exactly once with the baseline executable and once
with the candidate executable. The runner uses the Python standard library
and NumPy for declared semantic post-checks.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import difflib
import fnmatch
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import post_checks


PACKAGE_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = PACKAGE_ROOT / "manifest.json"
WORK_ROOT = PACKAGE_ROOT / ".work"
REPORT_DIR = PACKAGE_ROOT / "reports"

CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
TAG_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]*$")
NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?(?![A-Za-z0-9_])"
)
NONFINITE_RE = re.compile(r"(?<![A-Za-z_])[-+]?(?:nan|inf(?:inity)?)(?![A-Za-z_])", re.I)
IGNORED_STDOUT_LINES = (
    re.compile(rb"^\s*Time used(?: for this run)?\s*=.*$"),
    re.compile(rb"^\s*Speed of this run\s*=.*$"),
)
IGNORED_STDERR_LINES = (
    re.compile(rb"^\s*File:\s+.*$"),
    re.compile(rb"^\s*Line:\s+\d+\s*$"),
)


class ConfigurationError(RuntimeError):
    """The test package, manifest, or command-line invocation is invalid."""


class ComparisonError(RuntimeError):
    """A run violates its contract or differs from the other executable."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def require_inside(path: Path, parent: Path, description: str) -> None:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError as exc:
        raise ConfigurationError(f"{description} escapes {parent}: {path}") from exc


def validate_relative_target(value: str, description: str) -> None:
    target = Path(value)
    if target.is_absolute() or ".." in target.parts or value in ("", "."):
        raise ConfigurationError(f"{description} must be a safe relative path: {value!r}")


def resolve_source(specification: str, repo_root: Path, description: str) -> Path:
    if not isinstance(specification, str):
        raise ConfigurationError(f"{description} source must be a string")
    prefix, separator, value = specification.partition(":")
    if not separator or prefix not in ("repo", "package"):
        raise ConfigurationError(
            f"{description} source must start with 'repo:' or 'package:'"
        )
    validate_relative_target(value, f"{description} source")
    root = repo_root if prefix == "repo" else PACKAGE_ROOT
    path = (root / value).resolve()
    require_inside(path, root, f"{description} source")
    if not path.exists():
        raise ConfigurationError(f"{description} source does not exist: {specification}")
    return path


def read_manifest_data() -> Dict[str, Any]:
    def reject_duplicate_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ConfigurationError(f"Duplicate JSON key in manifest: {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            MANIFEST_PATH.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Cannot read {MANIFEST_PATH}: {exc}") from exc


def validate_stage(
    stage: Any, owner: str, repo_root: Path, allow_empty: bool = False
) -> None:
    if not isinstance(stage, list) or (not stage and not allow_empty):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise ConfigurationError(f"{owner} stage must be {qualifier}")
    targets: Set[str] = set()
    for item in stage:
        if not isinstance(item, dict) or set(item) != {"source", "target"}:
            raise ConfigurationError(f"{owner} stage entries require exactly source and target")
        source = item["source"]
        target = item["target"]
        if not isinstance(target, str):
            raise ConfigurationError(f"{owner} stage target must be a string")
        resolve_source(source, repo_root, owner)
        validate_relative_target(target, f"{owner} target")
        if target in targets:
            raise ConfigurationError(f"{owner} has duplicate target: {target}")
        targets.add(target)


def effective_commands(input_path: Path, case_id: str) -> List[List[str]]:
    try:
        lines = input_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigurationError(f"Cannot read input for case {case_id}: {exc}") from exc
    commands: List[List[str]] = []
    for raw_line in lines:
        fields = raw_line.split()
        for index, field in enumerate(fields):
            if field.startswith("#"):
                fields = fields[:index]
                break
        if fields:
            commands.append(fields)
    return commands


def validate_input_command_order(input_path: Path, case_id: str, style: str) -> None:
    commands = effective_commands(input_path, case_id)
    keywords = [fields[0] for fields in commands]
    if "potential" not in keywords:
        raise ConfigurationError(
            f"Case {case_id} must contain at least one effective potential command"
        )

    # Negative behavior contracts may intentionally violate one of these
    # static rules so the executable can reject the input with its public
    # diagnostic. All other manifest checks still apply to those cases.
    if style == "intentional_invalid":
        return

    replicate_indices = [i for i, keyword in enumerate(keywords) if keyword == "replicate"]
    if len(replicate_indices) > 1:
        raise ConfigurationError(f"Case {case_id} contains more than one replicate command")
    if replicate_indices and replicate_indices[0] != 0:
        raise ConfigurationError(
            f"Case {case_id} must place replicate at the first effective command"
        )

    # Compatibility cases relax placement of run settings and Actions around
    # ensemble and may append another potential between completed runs. The
    # replicate and one-global-command rules still apply.

    for keyword in ("dftd3", "kspace"):
        if keywords.count(keyword) > 1:
            raise ConfigurationError(
                f"Case {case_id} contains more than one {keyword} command"
            )

    first_run = keywords.index("run") if "run" in keywords else len(keywords)
    potential_indices = [i for i, keyword in enumerate(keywords) if keyword == "potential"]
    if potential_indices and style == "canonical":
        if max(potential_indices) >= first_run:
            raise ConfigurationError(
                f"Case {case_id} must place all potential commands before the first run"
            )
        for index in range(min(potential_indices), max(potential_indices) + 1):
            if keywords[index] != "potential":
                raise ConfigurationError(
                    f"Case {case_id} potential commands must form one contiguous block"
                )
    elif potential_indices and min(potential_indices) >= first_run:
        raise ConfigurationError(
            f"Case {case_id} must place its first potential before the first run"
        )
    if any(i >= first_run for i, keyword in enumerate(keywords) if keyword == "velocity"):
        raise ConfigurationError(f"Case {case_id} must place velocity before the first run")


def ensemble_keywords(path: Path, case_id: str) -> List[str]:
    return [
        fields[1]
        for fields in effective_commands(path, case_id)
        if len(fields) >= 2 and fields[0] == "ensemble"
    ]


def validate_expectation(expectation: Any, owner: str) -> None:
    if not isinstance(expectation, dict):
        raise ConfigurationError(f"{owner} must be an object")
    allowed = {
        "expect",
        "expected_returncode",
        "stdout_contains",
        "stderr_contains",
        "outputs",
    }
    unknown = set(expectation) - allowed
    if unknown:
        raise ConfigurationError(f"{owner} has unknown key(s): {sorted(unknown)}")
    expect = expectation.get("expect")
    if expect not in ("success", "failure"):
        raise ConfigurationError(f"{owner}.expect must be success or failure")
    default_code = 0 if expect == "success" else None
    expected_returncode = expectation.get("expected_returncode", default_code)
    if not isinstance(expected_returncode, int) or isinstance(expected_returncode, bool):
        raise ConfigurationError(f"{owner}.expected_returncode must be an integer")
    if expect == "success" and expected_returncode != 0:
        raise ConfigurationError(f"{owner} success return code must be 0")
    if expect == "failure" and expected_returncode == 0:
        raise ConfigurationError(f"{owner} failure return code must be nonzero")
    for stream in ("stdout_contains", "stderr_contains"):
        if stream in expectation and (
            not isinstance(expectation[stream], str) or not expectation[stream]
        ):
            raise ConfigurationError(f"{owner}.{stream} must be a non-empty string")
    if expect == "failure" and not any(
        stream in expectation for stream in ("stdout_contains", "stderr_contains")
    ):
        raise ConfigurationError(
            f"{owner} failure must declare stdout_contains or stderr_contains"
        )
    if "outputs" in expectation:
        outputs = expectation["outputs"]
        if (
            not isinstance(outputs, list)
            or not all(isinstance(output, str) for output in outputs)
            or len(outputs) != len(set(outputs))
        ):
            raise ConfigurationError(f"{owner}.outputs must be a unique string list")
        for output in outputs:
            validate_relative_target(output, f"{owner} output")


def validate_comparisons(comparisons: Any, outputs: Set[str], case_id: str) -> None:
    if not isinstance(comparisons, dict):
        raise ConfigurationError(f"Case {case_id} comparisons must be an object")
    unknown_outputs = set(comparisons) - outputs
    if unknown_outputs:
        raise ConfigurationError(
            f"Case {case_id} comparisons names undeclared output(s): {sorted(unknown_outputs)}"
        )
    required_keys = {"mode", "rtol", "atol", "reason"}
    for output, comparison in comparisons.items():
        if not isinstance(comparison, dict) or set(comparison) != required_keys:
            raise ConfigurationError(
                f"Case {case_id} comparison for {output} must contain "
                "mode, rtol, atol, and reason"
            )
        if comparison["mode"] != "numeric":
            raise ConfigurationError(
                f"Case {case_id} comparison for {output} only supports "
                "mode 'numeric'"
            )
        for tolerance in ("rtol", "atol"):
            value = comparison[tolerance]
            if (
                not is_finite_number(value)
                or value < 0
            ):
                raise ConfigurationError(
                    f"Case {case_id} comparison {output}.{tolerance} must be non-negative"
                )
        if not isinstance(comparison["reason"], str) or not comparison["reason"].strip():
            raise ConfigurationError(
                f"Case {case_id} comparison {output}.reason must be non-empty"
            )


def validate_relation_reference(
    reference: Any,
    owner: str,
    cases_by_id: Mapping[str, Mapping[str, Any]],
) -> Tuple[str, str]:
    if not isinstance(reference, dict) or set(reference) != {"case", "output"}:
        raise ConfigurationError(f"{owner} must contain exactly case and output")
    case_id = reference["case"]
    output = reference["output"]
    if not isinstance(case_id, str) or case_id not in cases_by_id:
        raise ConfigurationError(f"{owner} references unknown case {case_id!r}")
    if not isinstance(output, str) or output not in cases_by_id[case_id]["outputs"]:
        raise ConfigurationError(
            f"{owner} references undeclared output {output!r} in case {case_id}"
        )
    return case_id, output


def relation_references(relation: Mapping[str, Any]) -> List[Mapping[str, str]]:
    if relation["operation"] == "equal":
        return list(relation["members"])
    return list(relation["parts"]) + [relation["result"]]


def validate_relations(
    relations: Any,
    cases_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    if not isinstance(relations, list):
        raise ConfigurationError("relations must be a list")
    relation_ids: Set[str] = set()
    for relation in relations:
        if not isinstance(relation, dict):
            raise ConfigurationError("Each relation must be an object")
        relation_id = relation.get("id")
        if not isinstance(relation_id, str) or not CASE_ID_RE.fullmatch(relation_id):
            raise ConfigurationError(f"Invalid relation id: {relation_id!r}")
        if relation_id in relation_ids:
            raise ConfigurationError(f"Duplicate relation id: {relation_id}")
        relation_ids.add(relation_id)
        description = relation.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ConfigurationError(
                f"Relation {relation_id} description must be non-empty"
            )
        roles = relation.get("roles")
        if (
            not isinstance(roles, list)
            or not roles
            or not all(role in ("baseline", "candidate") for role in roles)
            or len(roles) != len(set(roles))
        ):
            raise ConfigurationError(
                f"Relation {relation_id} roles must be a unique non-empty role list"
            )
        operation = relation.get("operation")
        if operation == "equal":
            allowed = {"id", "description", "roles", "operation", "members"}
            if set(relation) != allowed:
                raise ConfigurationError(
                    f"Relation {relation_id} equal relation must contain exactly "
                    f"{sorted(allowed)}"
                )
            members = relation["members"]
            if not isinstance(members, list) or len(members) < 2:
                raise ConfigurationError(
                    f"Relation {relation_id} equal relation needs at least two members"
                )
            references = members
        elif operation == "concat":
            allowed = {
                "id",
                "description",
                "roles",
                "operation",
                "parts",
                "result",
            }
            if set(relation) != allowed:
                raise ConfigurationError(
                    f"Relation {relation_id} concat relation must contain exactly "
                    f"{sorted(allowed)}"
                )
            parts = relation["parts"]
            if not isinstance(parts, list) or not parts:
                raise ConfigurationError(
                    f"Relation {relation_id} concat relation needs at least one part"
                )
            references = parts + [relation["result"]]
        else:
            raise ConfigurationError(
                f"Relation {relation_id} operation must be equal or concat"
            )

        normalized_references = [
            validate_relation_reference(
                reference, f"Relation {relation_id} reference", cases_by_id
            )
            for reference in references
        ]
        if len(normalized_references) != len(set(normalized_references)):
            raise ConfigurationError(
                f"Relation {relation_id} contains duplicate output references"
            )
        referenced_case_ids = {case_id for case_id, _ in normalized_references}
        candidate_only_cases = sorted(
            case_id
            for case_id in referenced_case_ids
            if cases_by_id[case_id].get("candidate_only", False)
        )
        if candidate_only_cases:
            raise ConfigurationError(
                f"Relation {relation_id} cannot reference candidate-only case(s): "
                f"{candidate_only_cases}"
            )
        shared_suites = set.intersection(
            *(set(cases_by_id[case_id]["suites"]) for case_id in referenced_case_ids)
        )
        if not shared_suites:
            raise ConfigurationError(
                f"Relation {relation_id} references cases with no shared suite"
            )
        for role in roles:
            for case_id, output in normalized_references:
                if expectation_for_role(cases_by_id[case_id], role)["expect"] != "success":
                    raise ConfigurationError(
                        f"Relation {relation_id} references {case_id} for {role}, "
                        "but that role is not expected to succeed"
                    )
                if output not in expected_outputs_for_role(cases_by_id[case_id], role):
                    raise ConfigurationError(
                        f"Relation {relation_id} references {case_id}:{output} for {role}, "
                        "but that output is not expected for the role"
                    )


def validate_manifest(manifest: Mapping[str, Any], repo_root: Path) -> None:
    allowed_top = {
        "schema_version",
        "defaults",
        "required_ensemble_keywords",
        "fixtures",
        "cases",
        "known_manual_gaps",
        "relations",
    }
    unknown_top = set(manifest) - allowed_top
    if unknown_top:
        raise ConfigurationError(f"Unknown manifest key(s): {sorted(unknown_top)}")
    if manifest.get("schema_version") != 1:
        raise ConfigurationError("manifest schema_version must be 1")

    defaults = manifest.get("defaults")
    if not isinstance(defaults, dict) or set(defaults) != {"timeout_s"}:
        raise ConfigurationError("defaults must contain exactly timeout_s")
    timeout_s = defaults["timeout_s"]
    if (
        not is_finite_number(timeout_s)
        or timeout_s <= 0
    ):
        raise ConfigurationError("defaults.timeout_s must be positive")

    required = manifest.get("required_ensemble_keywords")
    if (
        not isinstance(required, list)
        or not required
        or not all(isinstance(value, str) and value for value in required)
        or len(required) != len(set(required))
    ):
        raise ConfigurationError(
            "required_ensemble_keywords must be a non-empty unique string list"
        )

    gaps = manifest.get("known_manual_gaps")
    if not isinstance(gaps, list) or not all(
        isinstance(gap, str) and gap for gap in gaps
    ):
        raise ConfigurationError("known_manual_gaps must be a list of non-empty strings")

    fixtures = manifest.get("fixtures")
    if not isinstance(fixtures, dict) or not fixtures:
        raise ConfigurationError("fixtures must be a non-empty object")
    for name, fixture in fixtures.items():
        if not CASE_ID_RE.fullmatch(name):
            raise ConfigurationError(f"Invalid fixture name: {name!r}")
        if not isinstance(fixture, dict) or set(fixture) != {"stage"}:
            raise ConfigurationError(f"Fixture {name} must contain exactly a stage list")
        validate_stage(fixture["stage"], f"fixture {name}", repo_root)

    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ConfigurationError("cases must be a non-empty list")
    allowed_case = {
        "id",
        "description",
        "suites",
        "fixture",
        "stage",
        "input",
        "input_style",
        "covers",
        "expect",
        "outputs",
        "timeout_s",
        "expected_returncode",
        "stdout_contains",
        "stderr_contains",
        "mutable_inputs",
        "comparisons",
        "role_expectations",
        "compare_cross_version",
        "candidate_only",
        "allow_empty_outputs",
        "post_checks",
    }
    ids: Set[str] = set()
    full_coverage: Set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ConfigurationError("Each case must be an object")
        case_id = case.get("id")
        unknown = set(case) - allowed_case
        if unknown:
            raise ConfigurationError(
                f"Case {case_id or '<unknown>'} has unknown key(s): {sorted(unknown)}"
            )
        if not isinstance(case_id, str) or not CASE_ID_RE.fullmatch(case_id):
            raise ConfigurationError(f"Invalid case id: {case_id!r}")
        if case_id in ids:
            raise ConfigurationError(f"Duplicate case id: {case_id}")
        ids.add(case_id)

        description = case.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ConfigurationError(f"Case {case_id} description must be non-empty")
        fixture_name = case.get("fixture")
        if fixture_name not in fixtures:
            raise ConfigurationError(
                f"Case {case_id} references unknown fixture {fixture_name!r}"
            )

        suites = case.get("suites")
        if (
            not isinstance(suites, list)
            or not suites
            or not all(isinstance(value, str) and CASE_ID_RE.fullmatch(value) for value in suites)
            or len(suites) != len(set(suites))
        ):
            raise ConfigurationError(f"Case {case_id} suites is invalid")
        has_role_expectations = "role_expectations" in case
        if "full" not in suites:
            raise ConfigurationError(
                f"Case {case_id} must belong to the full suite"
            )

        covers = case.get("covers", [])
        if (
            not isinstance(covers, list)
            or not all(isinstance(value, str) and TAG_RE.fullmatch(value) for value in covers)
            or len(covers) != len(set(covers))
        ):
            raise ConfigurationError(f"Case {case_id} covers must be unique valid tags")

        style = case.get("input_style", "canonical")
        if style not in ("canonical", "compatibility", "intentional_invalid"):
            raise ConfigurationError(
                f"Case {case_id} input_style must be canonical, compatibility, "
                "or intentional_invalid"
            )
        if style == "intentional_invalid" and case.get("expect") != "failure":
            raise ConfigurationError(
                f"Case {case_id} intentional_invalid input must expect failure"
            )
        candidate_only = case.get("candidate_only", False)
        if "candidate_only" in case and candidate_only is not True:
            raise ConfigurationError(
                f"Case {case_id} candidate_only, when present, must be true"
            )
        if candidate_only:
            if style != "intentional_invalid":
                raise ConfigurationError(
                    f"Case {case_id} candidate_only requires intentional_invalid input"
                )
            if "full" not in suites:
                raise ConfigurationError(
                    f"Case {case_id} candidate_only must belong to the full suite"
                )
            if case.get("expect") != "failure":
                raise ConfigurationError(
                    f"Case {case_id} candidate_only must expect candidate failure"
                )
            if case.get("expected_returncode") != 1:
                raise ConfigurationError(
                    f"Case {case_id} candidate_only must require candidate exit 1"
                )
            if not any(
                diagnostic in case
                for diagnostic in ("stdout_contains", "stderr_contains")
            ):
                raise ConfigurationError(
                    f"Case {case_id} candidate_only must require a candidate diagnostic"
                )
            if has_role_expectations or "compare_cross_version" in case:
                raise ConfigurationError(
                    f"Case {case_id} candidate_only cannot use role expectations"
                )
        input_path = resolve_source(case.get("input"), repo_root, f"case {case_id} input")
        if not input_path.is_file():
            raise ConfigurationError(f"Case {case_id} input is not a file: {input_path}")
        validate_input_command_order(input_path, case_id, style)
        if "full" in suites:
            full_coverage.update(ensemble_keywords(input_path, case_id))

        expectation_keys = (
            "expect",
            "expected_returncode",
            "stdout_contains",
            "stderr_contains",
        )
        case_expectation = {key: case[key] for key in expectation_keys if key in case}
        validate_expectation(case_expectation, f"Case {case_id}")
        role_expectations = case.get("role_expectations")
        if has_role_expectations and role_expectations is None:
            raise ConfigurationError(
                f"Case {case_id} role_expectations cannot be null"
            )
        if role_expectations is not None:
            if not isinstance(role_expectations, dict) or set(role_expectations) != {
                "baseline",
                "candidate",
            }:
                raise ConfigurationError(
                    f"Case {case_id} role_expectations must contain baseline and candidate"
                )
            for role in ("baseline", "candidate"):
                validate_expectation(
                    role_expectations[role], f"Case {case_id} role_expectations.{role}"
                )
            candidate_expectation = role_expectations["candidate"]
            if case_expectation["expect"] != candidate_expectation["expect"]:
                raise ConfigurationError(
                    f"Case {case_id} top-level expect must summarize candidate expect"
                )
            top_returncode = case_expectation.get(
                "expected_returncode",
                0 if case_expectation["expect"] == "success" else None,
            )
            candidate_returncode = candidate_expectation.get(
                "expected_returncode",
                0 if candidate_expectation["expect"] == "success" else None,
            )
            if top_returncode != candidate_returncode:
                raise ConfigurationError(
                    f"Case {case_id} top-level return code must summarize candidate"
                )
            for diagnostic in ("stdout_contains", "stderr_contains"):
                if (
                    diagnostic in case_expectation
                    and case_expectation[diagnostic]
                    != candidate_expectation.get(diagnostic)
                ):
                    raise ConfigurationError(
                        f"Case {case_id} top-level {diagnostic} must summarize candidate"
                    )
        if "compare_cross_version" in case and not isinstance(
            case["compare_cross_version"], bool
        ):
            raise ConfigurationError(f"Case {case_id} compare_cross_version must be boolean")
        if "compare_cross_version" in case and role_expectations is None:
            raise ConfigurationError(
                f"Case {case_id} compare_cross_version is only valid with role_expectations"
            )
        if role_expectations is not None and case.get("compare_cross_version", False):
            baseline_expectation = role_expectations["baseline"]
            candidate_expectation = role_expectations["candidate"]
            baseline_returncode = baseline_expectation.get(
                "expected_returncode",
                0 if baseline_expectation["expect"] == "success" else None,
            )
            candidate_returncode = candidate_expectation.get(
                "expected_returncode",
                0 if candidate_expectation["expect"] == "success" else None,
            )
            if baseline_returncode != candidate_returncode:
                raise ConfigurationError(
                    f"Case {case_id} cannot compare roles with different return codes"
                )

        outputs = case.get("outputs", [])
        if (
            not isinstance(outputs, list)
            or not all(isinstance(output, str) for output in outputs)
            or len(outputs) != len(set(outputs))
        ):
            raise ConfigurationError(f"Case {case_id} outputs must be a unique string list")
        for output in outputs:
            validate_relative_target(output, f"Case {case_id} output")
        if candidate_only and outputs:
            raise ConfigurationError(
                f"Case {case_id} candidate_only cannot declare outputs"
            )
        for index, left in enumerate(outputs):
            left_parts = Path(left).parts
            for right in outputs[index + 1 :]:
                right_parts = Path(right).parts
                common = min(len(left_parts), len(right_parts))
                if left_parts[:common] == right_parts[:common]:
                    raise ConfigurationError(
                        f"Case {case_id} has overlapping outputs: {left!r}, {right!r}"
                    )
        if role_expectations is not None:
            for role, expectation in role_expectations.items():
                role_outputs = expectation.get("outputs", outputs)
                unknown_role_outputs = set(role_outputs) - set(outputs)
                if unknown_role_outputs:
                    raise ConfigurationError(
                        f"Case {case_id} role_expectations.{role}.outputs contains "
                        f"output(s) absent from case outputs: {sorted(unknown_role_outputs)}"
                    )
            if case.get("compare_cross_version", False) and (
                sorted(expected_outputs_for_role(case, "baseline"))
                != sorted(expected_outputs_for_role(case, "candidate"))
            ):
                raise ConfigurationError(
                    f"Case {case_id} cannot compare roles with different output inventories"
                )
        effective_expectations = (
            role_expectations.values() if role_expectations is not None else (case_expectation,)
        )
        success_for_any_role = any(
            expectation["expect"] == "success" for expectation in effective_expectations
        )
        if success_for_any_role and not outputs:
            raise ConfigurationError(f"Successful case {case_id} must declare outputs")
        if "comparisons" in case:
            validate_comparisons(case["comparisons"], set(outputs), case_id)

        allow_empty_outputs = case.get("allow_empty_outputs", [])
        if (
            not isinstance(allow_empty_outputs, list)
            or not all(isinstance(output, str) for output in allow_empty_outputs)
            or len(allow_empty_outputs) != len(set(allow_empty_outputs))
        ):
            raise ConfigurationError(
                f"Case {case_id} allow_empty_outputs must be a unique string list"
            )
        unknown_empty_outputs = set(allow_empty_outputs) - set(outputs)
        if unknown_empty_outputs:
            raise ConfigurationError(
                f"Case {case_id} allow_empty_outputs names undeclared output(s): "
                f"{sorted(unknown_empty_outputs)}"
            )

        post_check_specs = case.get("post_checks", [])
        if not isinstance(post_check_specs, list):
            raise ConfigurationError(f"Case {case_id} post_checks must be a list")
        for index, spec in enumerate(post_check_specs):
            try:
                post_checks.validate_spec(spec, f"Case {case_id} post_checks[{index}]")
            except ValueError as exc:
                raise ConfigurationError(str(exc)) from exc

        if "timeout_s" in case:
            value = case["timeout_s"]
            if (
                not is_finite_number(value)
                or value <= 0
            ):
                raise ConfigurationError(f"Case {case_id} timeout_s must be positive")

        if "stage" in case:
            validate_stage(case["stage"], f"case {case_id}", repo_root)
        fixture_targets = [item["target"] for item in fixtures[fixture_name]["stage"]]
        case_targets = [item["target"] for item in case.get("stage", [])]
        combined_targets = fixture_targets + case_targets
        combined_stage_items = list(fixtures[fixture_name]["stage"]) + list(
            case.get("stage", [])
        )
        if len(combined_targets) != len(set(combined_targets)):
            raise ConfigurationError(f"Case {case_id} has a stage target collision")
        for index, left in enumerate(combined_targets):
            left_parts = Path(left).parts
            for right in combined_targets[index + 1 :]:
                right_parts = Path(right).parts
                common = min(len(left_parts), len(right_parts))
                if left_parts[:common] == right_parts[:common]:
                    raise ConfigurationError(
                        f"Case {case_id} has overlapping stage targets: {left!r}, {right!r}"
                    )
        staged_directory_targets = {
            item["target"]
            for item in combined_stage_items
            if resolve_source(
                item["source"], repo_root, f"case {case_id} potential source"
            ).is_dir()
        }
        for fields in effective_commands(input_path, case_id):
            if fields[0] != "potential":
                continue
            if len(fields) < 2:
                raise ConfigurationError(
                    f"Case {case_id} potential command must name a staged file"
                )
            potential_path = fields[1]
            validate_relative_target(
                potential_path, f"Case {case_id} potential file"
            )
            is_staged = potential_path in combined_targets or any(
                Path(potential_path).is_relative_to(Path(directory_target))
                for directory_target in staged_directory_targets
            )
            if not is_staged:
                raise ConfigurationError(
                    f"Case {case_id} potential file is not staged: {potential_path!r}"
                )
        reserved = {"run.in", "stdout.txt", "stderr.txt"}
        if set(combined_targets) & reserved:
            raise ConfigurationError(f"Case {case_id} stages a runner-reserved filename")
        if set(outputs) & reserved:
            raise ConfigurationError(f"Case {case_id} output uses a runner-reserved filename")
        for output in outputs:
            output_parts = Path(output).parts
            for target in combined_targets:
                target_parts = Path(target).parts
                common = min(len(output_parts), len(target_parts))
                if output_parts[:common] == target_parts[:common]:
                    raise ConfigurationError(
                        f"Case {case_id} output {output!r} overlaps staged target {target!r}"
                    )

        mutable_inputs = case.get("mutable_inputs", [])
        if (
            not isinstance(mutable_inputs, list)
            or not all(isinstance(path, str) for path in mutable_inputs)
            or len(mutable_inputs) != len(set(mutable_inputs))
        ):
            raise ConfigurationError(
                f"Case {case_id} mutable_inputs must be a unique string list"
            )
        for path in mutable_inputs:
            validate_relative_target(path, f"Case {case_id} mutable input")
        if candidate_only and mutable_inputs:
            raise ConfigurationError(
                f"Case {case_id} candidate_only cannot declare mutable inputs"
            )
        if mutable_inputs and "command:deposit" not in covers:
            raise ConfigurationError(
                f"Case {case_id} mutable_inputs is only allowed for command:deposit"
            )
        if mutable_inputs and not any(
            fields[0] == "deposit" for fields in effective_commands(input_path, case_id)
        ):
            raise ConfigurationError(
                f"Case {case_id} mutable_inputs requires an actual deposit command"
            )
        available_inputs = {"run.in"} | set(combined_targets)
        undeclared_inputs = set(mutable_inputs) - available_inputs
        if undeclared_inputs:
            raise ConfigurationError(
                f"Case {case_id} mutable_inputs names unstaged path(s): "
                f"{sorted(undeclared_inputs)}"
            )

    missing = set(required) - full_coverage
    if missing:
        raise ConfigurationError(
            f"Full suite inputs miss ensemble keyword(s): {sorted(missing)}"
        )
    validate_relations(
        manifest.get("relations", []),
        {case["id"]: case for case in cases},
    )


def available_suites(manifest: Mapping[str, Any]) -> List[str]:
    cases = manifest.get("cases", [])
    if not isinstance(cases, list):
        return []
    return sorted(
        {
            suite
            for case in cases
            if isinstance(case, dict)
            for suite in case.get("suites", [])
            if isinstance(suite, str)
        }
    )


def parse_arguments(manifest: Mapping[str, Any]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        required=True,
        help="Path to the GPUMD source repository used by repo: stage sources",
    )
    parser.add_argument("--suite", default="quick", help="Manifest suite to run")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        metavar="GLOB",
        help="Run matching case id(s); may be repeated",
    )
    parser.add_argument("--baseline", help="Baseline executable path")
    parser.add_argument("--candidate", help="Candidate executable path")
    parser.add_argument(
        "--device",
        help="CUDA/HIP visible device id; defaults to the first already-visible device",
    )
    parser.add_argument("--timeout-scale", type=float, default=1.0)
    parser.add_argument("--keep-all", action="store_true", help="Retain passing workdirs")
    parser.add_argument("--list", action="store_true", help="List selected cases without running")
    parser.add_argument(
        "--check-manifest", action="store_true", help="Validate package structure without a GPU"
    )
    args = parser.parse_args()
    suites = available_suites(manifest)
    if args.suite not in suites:
        parser.error(f"unknown suite {args.suite!r}; choose from {', '.join(suites)}")
    if not args.check_manifest and not args.list:
        if args.baseline is None or args.candidate is None:
            parser.error("--baseline and --candidate are required when running cases")
    return args


def resolve_command_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def executable_metadata(path: Path) -> Dict[str, Any]:
    return {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)}


def copy_stage_item(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)


def inspect_workdir(directory: Path) -> Tuple[Set[str], List[str]]:
    files: Set[str] = set()
    unsafe: List[str] = []
    resolved_directory = directory.resolve()
    for path in directory.rglob("*"):
        relative_path = path.relative_to(directory).as_posix()
        if path.is_symlink():
            unsafe.append(f"symlink:{relative_path}")
            continue
        if path.is_dir():
            continue
        if not path.is_file():
            unsafe.append(f"nonregular:{relative_path}")
            continue
        try:
            path.resolve(strict=True).relative_to(resolved_directory)
        except (OSError, ValueError):
            unsafe.append(f"escape:{relative_path}")
            continue
        files.add(relative_path)
    return files, sorted(unsafe)


def prepare_workdir(
    case: Mapping[str, Any],
    manifest: Mapping[str, Any],
    role: str,
    invocation_root: Path,
    repo_root: Path,
) -> Tuple[Path, Dict[str, str]]:
    workdir = (invocation_root / "cases" / case["id"] / role).resolve()
    require_inside(workdir, invocation_root, "work directory")
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    stage_items: List[Mapping[str, str]] = []
    stage_items.extend(manifest["fixtures"][case["fixture"]]["stage"])
    stage_items.extend(case.get("stage", []))
    for item in stage_items:
        source = resolve_source(item["source"], repo_root, f"case {case['id']}")
        target = workdir / item["target"]
        require_inside(target, workdir, "staged target")
        if target.exists():
            raise ConfigurationError(f"Stage collision in {case['id']}: {item['target']}")
        copy_stage_item(source, target)

    input_path = resolve_source(case["input"], repo_root, f"case {case['id']} input")
    shutil.copy2(input_path, workdir / "run.in")
    staged_files, unsafe = inspect_workdir(workdir)
    if unsafe:
        raise ConfigurationError(
            f"Case {case['id']} staging created unsafe path(s): {unsafe}"
        )
    hashes = {path: sha256_file(workdir / path) for path in staged_files}
    return workdir, hashes


def validate_xyz_output(text: str, label: str) -> None:
    lines = text.splitlines()
    cursor = 0
    frames = 0
    while cursor < len(lines):
        if not lines[cursor].strip():
            raise ComparisonError(f"{label}: blank line where an XYZ atom count was expected")
        try:
            number_of_atoms = int(lines[cursor].strip())
        except ValueError as exc:
            raise ComparisonError(f"{label}: invalid XYZ atom count at line {cursor + 1}") from exc
        if number_of_atoms <= 0:
            raise ComparisonError(f"{label}: XYZ atom count must be positive")
        frame_end = cursor + number_of_atoms + 2
        if frame_end > len(lines):
            raise ComparisonError(f"{label}: truncated XYZ frame {frames + 1}")
        for line_number in range(cursor + 2, frame_end):
            fields = lines[line_number].split()
            if len(fields) < 4:
                raise ComparisonError(
                    f"{label}: malformed XYZ atom record at line {line_number + 1}"
                )
            try:
                coordinates = [float(value) for value in fields[1:4]]
            except ValueError as exc:
                raise ComparisonError(
                    f"{label}: non-numeric XYZ coordinate at line {line_number + 1}"
                ) from exc
            if not all(math.isfinite(value) for value in coordinates):
                raise ComparisonError(
                    f"{label}: non-finite XYZ coordinate at line {line_number + 1}"
                )
        cursor = frame_end
        frames += 1
    if frames == 0:
        raise ComparisonError(f"{label}: no XYZ frames")


def validate_csv_output(text: str, label: str) -> None:
    rows = [row for row in csv.reader(text.splitlines()) if row]
    if len(rows) < 2:
        raise ComparisonError(f"{label}: CSV must contain a header and at least one data row")
    width = len(rows[0])
    if width < 2 or any(len(row) != width for row in rows[1:]):
        raise ComparisonError(f"{label}: inconsistent CSV column count")
    if any(not cell.strip() for row in rows for cell in row):
        raise ComparisonError(f"{label}: CSV contains an empty cell")
    for row_number, row in enumerate(rows[1:], start=2):
        try:
            values = [float(cell) for cell in row]
        except ValueError as exc:
            raise ComparisonError(
                f"{label}: non-numeric CSV data at row {row_number}"
            ) from exc
        if not all(math.isfinite(value) for value in values):
            raise ComparisonError(
                f"{label}: non-finite CSV data at row {row_number}"
            )


def validate_yaml_output(text: str, label: str) -> None:
    content_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not content_lines or any(":" not in line for line in content_lines):
        raise ComparisonError(f"{label}: malformed flat YAML output")
    for line_number, line in enumerate(content_lines, start=1):
        key, value_text = line.split(":", 1)
        if not key.strip() or not value_text.strip():
            raise ComparisonError(
                f"{label}: empty YAML key or value at content line {line_number}"
            )
        try:
            value = float(value_text)
        except ValueError as exc:
            raise ComparisonError(
                f"{label}: non-numeric YAML value at content line {line_number}"
            ) from exc
        if not math.isfinite(value):
            raise ComparisonError(
                f"{label}: non-finite YAML value at content line {line_number}"
            )


def validate_generated_output(
    path: Path, relative_path: str, label: str, allow_empty: bool = False
) -> None:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ComparisonError(f"{label}: cannot read generated output: {exc}") from exc
    if not data:
        if allow_empty:
            return
        raise ComparisonError(f"{label}: generated output is empty")
    suffix = Path(relative_path).suffix.lower()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        if suffix in (".xyz", ".csv", ".yaml", ".yml"):
            raise ComparisonError(f"{label}: expected UTF-8 text output")
        return
    if NONFINITE_RE.search(text):
        raise ComparisonError(f"{label}: generated text contains NaN or Inf")
    numeric_tokens = [float(match.group(0)) for match in NUMBER_RE.finditer(text)]
    if not all(math.isfinite(value) for value in numeric_tokens):
        raise ComparisonError(f"{label}: generated numeric token overflows to NaN or Inf")
    if suffix == ".xyz":
        validate_xyz_output(text, label)
    elif suffix == ".csv":
        validate_csv_output(text, label)
    elif suffix in (".yaml", ".yml"):
        validate_yaml_output(text, label)


def descendant_process_ids(root_pid: int) -> List[int]:
    if os.name == "nt" or not Path("/proc").is_dir():
        return []
    children: Dict[int, List[int]] = {}
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            stat = stat_path.read_text(encoding="utf-8")
            closing_parenthesis = stat.rfind(")")
            fields = stat[closing_parenthesis + 1 :].split()
            pid = int(stat_path.parent.name)
            parent_pid = int(fields[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(parent_pid, []).append(pid)
    descendants: List[int] = []
    pending = list(children.get(root_pid, []))
    while pending:
        pid = pending.pop()
        descendants.append(pid)
        pending.extend(children.get(pid, []))
    return descendants


def terminate_process(process: subprocess.Popen[Any]) -> int:
    descendants = descendant_process_ids(process.pid)
    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for pid in reversed(descendants):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        process.kill()
    except ProcessLookupError:
        pass
    try:
        return process.wait(timeout=0.25)
    except subprocess.TimeoutExpired:
        return process.poll() if process.poll() is not None else -signal.SIGKILL


def run_once(
    executable: Path,
    case: Mapping[str, Any],
    manifest: Mapping[str, Any],
    role: str,
    invocation_root: Path,
    repo_root: Path,
    device_environment: Mapping[str, str],
    timeout_scale: float,
) -> Dict[str, Any]:
    workdir, staged_hashes = prepare_workdir(
        case, manifest, role, invocation_root, repo_root
    )
    timeout_s = float(case.get("timeout_s", manifest["defaults"]["timeout_s"])) * timeout_scale
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ConfigurationError(f"Case {case['id']} has a non-finite effective timeout")
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    environment["LANG"] = "C"
    environment["OMP_NUM_THREADS"] = "1"
    environment.update(device_environment)

    capture_dir = invocation_root / "captures" / case["id"] / role
    capture_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = capture_dir / "stdout.txt"
    stderr_path = capture_dir / "stderr.txt"
    started = time.monotonic()
    timed_out = False
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        try:
            process = subprocess.Popen(
                [str(executable)],
                cwd=str(workdir),
                env=environment,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=(os.name != "nt"),
            )
        except OSError as exc:
            raise ConfigurationError(f"Cannot execute {executable}: {exc}") from exc
        try:
            returncode = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = terminate_process(process)
        except BaseException:
            terminate_process(process)
            raise
    duration_s = time.monotonic() - started

    files_after, unsafe_files = inspect_workdir(workdir)
    changes: List[str] = []
    after_hashes: Dict[str, Optional[str]] = {}
    for relative_path, before_hash in sorted(staged_hashes.items()):
        staged_path = workdir / relative_path
        if relative_path not in files_after:
            changes.append(f"missing:{relative_path}")
            after_hashes[relative_path] = None
        else:
            after_hash = sha256_file(staged_path)
            after_hashes[relative_path] = after_hash
            if after_hash != before_hash:
                changes.append(f"modified:{relative_path}")
    mutable_inputs = set(case.get("mutable_inputs", []))
    unauthorized = [
        change
        for change in changes
        if change.startswith("missing:")
        or change.split(":", 1)[1] not in mutable_inputs
    ]
    generated = files_after - set(staged_hashes)
    output_hashes = {
        path: sha256_file(workdir / path)
        for path in sorted(generated)
        if (workdir / path).is_file()
    }
    return {
        "role": role,
        "workdir": str(workdir),
        "returncode": returncode,
        "timed_out": timed_out,
        "duration_s": duration_s,
        "generated_files": sorted(generated),
        "output_sha256": output_hashes,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_sha256": sha256_file(stderr_path),
        "unsafe_workdir_entries": unsafe_files,
        "staged_input_sha256_before": staged_hashes,
        "staged_input_sha256_after": after_hashes,
        "staged_input_changes": changes,
        "unauthorized_input_changes": unauthorized,
    }


def filter_stream_lines(
    data: bytes, patterns: Sequence[re.Pattern[bytes]]
) -> bytes:
    kept: List[bytes] = []
    for line in data.splitlines(keepends=True):
        body = line.rstrip(b"\r\n")
        if not any(pattern.match(body) for pattern in patterns):
            kept.append(line)
    return b"".join(kept)


def normalize_stdout(data: bytes) -> bytes:
    """Remove only GPUMD timing and speed lines."""
    return filter_stream_lines(data, IGNORED_STDOUT_LINES)


def normalize_stderr(data: bytes) -> bytes:
    """Remove only internal source-file and source-line location fields."""
    return filter_stream_lines(data, IGNORED_STDERR_LINES)


class DifferenceRecorder:
    def __init__(self, invocation_root: Path) -> None:
        self.root = invocation_root / "diffs"
        self.counter = 0

    def text_diff(
        self,
        label: str,
        left: str,
        right: str,
        left_name: str = "baseline",
        right_name: str = "candidate",
    ) -> str:
        self.counter += 1
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "difference"
        path = self.root / f"{self.counter:03d}-{safe_label}.diff"
        path.parent.mkdir(parents=True, exist_ok=True)
        diff = "".join(
            difflib.unified_diff(
                left.splitlines(keepends=True),
                right.splitlines(keepends=True),
                fromfile=f"{left_name}/{label}",
                tofile=f"{right_name}/{label}",
            )
        )
        if not diff:
            diff = (
                "Text differs but split-line rendering is identical.\n"
                f"baseline bytes: {left.encode('utf-8').hex()}\n"
                f"candidate bytes: {right.encode('utf-8').hex()}\n"
            )
        path.write_text(diff, encoding="utf-8")
        return str(path.relative_to(PACKAGE_ROOT))


def first_different_byte(left: bytes, right: bytes) -> Tuple[int, Optional[int], Optional[int]]:
    for index, (left_byte, right_byte) in enumerate(zip(left, right)):
        if left_byte != right_byte:
            return index, left_byte, right_byte
    index = min(len(left), len(right))
    return (
        index,
        left[index] if index < len(left) else None,
        right[index] if index < len(right) else None,
    )


def byte_label(value: Optional[int]) -> str:
    return "EOF" if value is None else f"0x{value:02x}"


def compare_exact_bytes(
    left: bytes,
    right: bytes,
    label: str,
    recorder: DifferenceRecorder,
    left_name: str = "baseline",
    right_name: str = "candidate",
) -> Dict[str, Any]:
    if left == right:
        return {"mode": "exact", "bytes": len(left), "sha256": sha256_bytes(left)}
    try:
        left_text = left.decode("utf-8")
        right_text = right.decode("utf-8")
        if "\x00" in left_text or "\x00" in right_text:
            raise ValueError("NUL marks binary data")
    except (UnicodeDecodeError, ValueError):
        offset, left_byte, right_byte = first_different_byte(left, right)
        raise ComparisonError(
            f"{label}: binary output differs at byte offset {offset}: "
            f"{byte_label(left_byte)} != {byte_label(right_byte)} "
            f"({left_name} size={len(left)}, {right_name} size={len(right)})"
        )
    diff_path = recorder.text_diff(
        label, left_text, right_text, left_name, right_name
    )
    raise ComparisonError(f"{label}: text differs; unified diff: {diff_path}")


def compare_numeric_text(
    left_text: str,
    right_text: str,
    label: str,
    rtol: float,
    atol: float,
    recorder: DifferenceRecorder,
) -> Dict[str, Any]:
    if not math.isfinite(rtol) or not math.isfinite(atol) or rtol < 0 or atol < 0:
        raise ComparisonError(f"{label}: numeric tolerances must be finite and non-negative")
    if NONFINITE_RE.search(left_text) or NONFINITE_RE.search(right_text):
        raise ComparisonError(f"{label}: NaN or Inf is not allowed")
    left_numbers = [float(match.group(0)) for match in NUMBER_RE.finditer(left_text)]
    right_numbers = [float(match.group(0)) for match in NUMBER_RE.finditer(right_text)]
    if not all(math.isfinite(value) for value in left_numbers + right_numbers):
        raise ComparisonError(f"{label}: parsed numeric field overflows to NaN or Inf")
    left_skeleton = NUMBER_RE.sub("<N>", left_text)
    right_skeleton = NUMBER_RE.sub("<N>", right_text)
    if left_skeleton != right_skeleton:
        diff_path = recorder.text_diff(label, left_skeleton, right_skeleton)
        raise ComparisonError(
            f"{label}: non-numeric text differs; unified diff: {diff_path}"
        )
    if len(left_numbers) != len(right_numbers):
        raise ComparisonError(
            f"{label}: numeric field count differs ({len(left_numbers)} != {len(right_numbers)})"
        )
    max_abs = 0.0
    max_rel = 0.0
    for index, (left_value, right_value) in enumerate(
        zip(left_numbers, right_numbers), start=1
    ):
        difference = abs(left_value - right_value)
        scale = max(abs(left_value), abs(right_value))
        relative = difference / scale if scale else 0.0
        max_abs = max(max_abs, difference)
        max_rel = max(max_rel, relative)
        if not math.isclose(left_value, right_value, rel_tol=rtol, abs_tol=atol):
            raise ComparisonError(
                f"{label}: numeric field {index} differs: "
                f"{left_value:.17g} vs {right_value:.17g} "
                f"(abs={difference:.3e}, rel={relative:.3e}, "
                f"rtol={rtol:.3e}, atol={atol:.3e})"
            )
    return {
        "mode": "numeric",
        "rtol": rtol,
        "atol": atol,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "numeric_fields": len(left_numbers),
    }


def read_stream(path: Path) -> bytes:
    return path.read_bytes()


def compare_run_pair(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    case: Mapping[str, Any],
    recorder: DifferenceRecorder,
) -> Dict[str, Any]:
    baseline_dir = Path(baseline["workdir"])
    candidate_dir = Path(candidate["workdir"])
    label = case["id"]
    metrics: Dict[str, Any] = {}

    if baseline["returncode"] != candidate["returncode"]:
        raise ComparisonError(
            f"{label}: return code differs "
            f"({baseline['returncode']} != {candidate['returncode']})"
        )
    if baseline["generated_files"] != candidate["generated_files"]:
        baseline_only = sorted(
            set(baseline["generated_files"]) - set(candidate["generated_files"])
        )
        candidate_only_files = sorted(
            set(candidate["generated_files"]) - set(baseline["generated_files"])
        )
        raise ComparisonError(
            f"{label}: generated-file inventory differs; "
            f"baseline-only={baseline_only}, candidate-only={candidate_only_files}"
        )

    comparisons = case.get("comparisons", {})
    comparison_errors: List[str] = []
    for output in baseline["generated_files"]:
        baseline_path = baseline_dir / output
        candidate_path = candidate_dir / output
        comparison = comparisons.get(output)
        try:
            if comparison is None:
                metrics[output] = compare_exact_bytes(
                    baseline_path.read_bytes(),
                    candidate_path.read_bytes(),
                    f"{label}:{output}",
                    recorder,
                )
            else:
                try:
                    baseline_text = baseline_path.read_bytes().decode("utf-8")
                    candidate_text = candidate_path.read_bytes().decode("utf-8")
                except (OSError, UnicodeDecodeError) as exc:
                    raise ComparisonError(
                        f"{label}:{output}: numeric comparison requires UTF-8 text: {exc}"
                    ) from exc
                metrics[output] = compare_numeric_text(
                    baseline_text,
                    candidate_text,
                    f"{label}:{output}",
                    float(comparison["rtol"]),
                    float(comparison["atol"]),
                    recorder,
                )
                metrics[output]["reason"] = comparison["reason"]
        except (ComparisonError, OSError) as exc:
            comparison_errors.append(str(exc))

    for mutable_path in case.get("mutable_inputs", []):
        try:
            metrics[f"mutable:{mutable_path}"] = compare_exact_bytes(
                (baseline_dir / mutable_path).read_bytes(),
                (candidate_dir / mutable_path).read_bytes(),
                f"{label}:mutable:{mutable_path}",
                recorder,
            )
        except (ComparisonError, OSError) as exc:
            comparison_errors.append(str(exc))

    for stream_name, normalize in (
        ("stdout", normalize_stdout),
        ("stderr", normalize_stderr),
    ):
        try:
            baseline_stream = normalize(
                read_stream(Path(baseline[f"{stream_name}_path"]))
            )
            candidate_stream = normalize(
                read_stream(Path(candidate[f"{stream_name}_path"]))
            )
            metrics[stream_name] = compare_exact_bytes(
                baseline_stream,
                candidate_stream,
                f"{label}:{stream_name}",
                recorder,
            )
        except (ComparisonError, OSError) as exc:
            comparison_errors.append(str(exc))

    if comparison_errors:
        raise ComparisonError("\n".join(comparison_errors))
    return metrics


def expectation_for_role(case: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    role_expectations = case.get("role_expectations")
    if role_expectations is not None:
        return role_expectations[role]
    return {
        key: case[key]
        for key in (
            "expect",
            "expected_returncode",
            "stdout_contains",
            "stderr_contains",
        )
        if key in case
    }


def expected_outputs_for_role(case: Mapping[str, Any], role: str) -> List[str]:
    expectation = expectation_for_role(case, role)
    if "role_expectations" not in case:
        return list(case["outputs"])
    if "outputs" in expectation:
        return list(expectation["outputs"])
    if expectation["expect"] == "failure":
        return []
    return list(case["outputs"])


def validate_run_result(result: Mapping[str, Any], case: Mapping[str, Any]) -> None:
    role = result["role"]
    label = f"{case['id']}:{role}"
    if result["timed_out"]:
        raise ComparisonError(f"{label}: timed out")
    if result["unsafe_workdir_entries"]:
        raise ComparisonError(
            f"{label}: generated symlink or non-regular path: "
            f"{result['unsafe_workdir_entries']}"
        )
    if result["unauthorized_input_changes"]:
        raise ComparisonError(
            f"{label}: staged input changed without mutable_inputs authorization: "
            f"{result['unauthorized_input_changes']}"
        )

    expectation = expectation_for_role(case, role)
    expect = expectation["expect"]
    expected_returncode = expectation.get(
        "expected_returncode", 0 if expect == "success" else None
    )
    if result["returncode"] != expected_returncode:
        raise ComparisonError(
            f"{label}: expected exit {expected_returncode}, got {result['returncode']}"
        )
    workdir = Path(result["workdir"])
    normalized_streams = {
        "stdout_contains": normalize_stdout(read_stream(Path(result["stdout_path"]))),
        "stderr_contains": normalize_stderr(read_stream(Path(result["stderr_path"]))),
    }
    for field, stream in normalized_streams.items():
        expected_diagnostic = expectation.get(field)
        if expected_diagnostic is not None and expected_diagnostic.encode("utf-8") not in stream:
            raise ComparisonError(
                f"{label}: {field.removesuffix('_contains')} does not contain "
                f"expected diagnostic {expected_diagnostic!r}"
            )

    expected_outputs = sorted(expected_outputs_for_role(case, role))
    if result["generated_files"] != expected_outputs:
        missing = sorted(set(expected_outputs) - set(result["generated_files"]))
        unexpected = sorted(set(result["generated_files"]) - set(expected_outputs))
        raise ComparisonError(
            f"{label}: generated-file inventory differs from manifest; "
            f"missing={missing}, unexpected={unexpected}"
        )
    allow_empty_outputs = set(case.get("allow_empty_outputs", []))
    for output in result["generated_files"]:
        validate_generated_output(
            workdir / output,
            output,
            f"{label}:{output}",
            allow_empty=output in allow_empty_outputs,
        )


def cross_version_comparison_enabled(case: Mapping[str, Any]) -> bool:
    if case.get("candidate_only", False):
        return False
    if "compare_cross_version" in case:
        return bool(case["compare_cross_version"])
    return "role_expectations" not in case


def execute_case(
    case: Mapping[str, Any],
    manifest: Mapping[str, Any],
    baseline: Path,
    candidate: Path,
    invocation_root: Path,
    repo_root: Path,
    device_environment: Mapping[str, str],
    timeout_scale: float,
    recorder: DifferenceRecorder,
) -> Dict[str, Any]:
    print(f"[{case['id']}] {case['description']}")
    runs: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []
    metrics: Dict[str, Any] = {}
    post_check_metrics: Dict[str, Any] = {}
    cross_version_compared = False
    candidate_only = bool(case.get("candidate_only", False))
    roles = (("candidate", candidate),) if candidate_only else (
        ("baseline", baseline),
        ("candidate", candidate),
    )
    not_run: Dict[str, str] = {}
    if candidate_only:
        reason = "not run by candidate_only validation contract"
        not_run["baseline"] = reason
        print(f"  baseline {reason}")
    for role, executable in roles:
        print(f"  run {role}", flush=True)
        result = run_once(
            executable,
            case,
            manifest,
            role,
            invocation_root,
            repo_root,
            device_environment,
            timeout_scale,
        )
        runs[role] = result
        try:
            validate_run_result(result, case)
            if case.get("post_checks"):
                try:
                    post_check_metrics[role] = post_checks.run(
                        case["post_checks"], Path(result["workdir"])
                    )
                except post_checks.PostCheckError as exc:
                    raise ComparisonError(f"{case['id']}:{role}: post-check failed: {exc}") from exc
        except ComparisonError as exc:
            errors.append(str(exc))

    if not errors and cross_version_comparison_enabled(case):
        cross_version_compared = True
        try:
            metrics = compare_run_pair(runs["baseline"], runs["candidate"], case, recorder)
        except ComparisonError as exc:
            errors.append(str(exc))

    if post_check_metrics:
        metrics["post_checks"] = post_check_metrics

    status = "PASS" if not errors else "FAIL"
    if status == "PASS":
        print("  PASS")
    else:
        print(f"  FAIL: {errors[0]}")
    return {
        "id": case["id"],
        "status": status,
        "errors": errors,
        "runs": runs,
        "not_run": not_run,
        "execution_contract": "candidate_only" if candidate_only else "two_sided",
        "cross_version_comparison_enabled": cross_version_comparison_enabled(case),
        "cross_version_compared": cross_version_compared,
        "metrics": metrics,
        "workdirs_retained": True,
    }


def relation_case_ids(relation: Mapping[str, Any]) -> Set[str]:
    return {reference["case"] for reference in relation_references(relation)}


def relation_output_path(
    reference: Mapping[str, str],
    role: str,
    case_results: Mapping[str, Mapping[str, Any]],
) -> Path:
    workdir = Path(case_results[reference["case"]]["runs"][role]["workdir"])
    return workdir / reference["output"]


def relation_reference_name(reference: Mapping[str, str], role: str) -> str:
    return f"{role}/{reference['case']}/{reference['output']}"


def evaluate_relation(
    relation: Mapping[str, Any],
    selected_case_ids: Set[str],
    case_results: Mapping[str, Mapping[str, Any]],
    recorder: DifferenceRecorder,
) -> Dict[str, Any]:
    referenced_case_ids = relation_case_ids(relation)
    base_result: Dict[str, Any] = {
        "id": relation["id"],
        "description": relation["description"],
        "operation": relation["operation"],
        "roles": relation["roles"],
        "case_ids": sorted(referenced_case_ids),
        "errors": [],
        "metrics": {},
    }
    missing_selection = sorted(referenced_case_ids - selected_case_ids)
    if missing_selection:
        base_result.update(
            {
                "status": "SKIP",
                "reason": "not all referenced cases were selected",
                "missing_case_ids": missing_selection,
            }
        )
        return base_result

    failed_cases = sorted(
        case_id
        for case_id in referenced_case_ids
        if case_results[case_id]["status"] != "PASS"
    )
    if failed_cases:
        base_result.update(
            {
                "status": "SKIP",
                "reason": "referenced case execution failed",
                "failed_case_ids": failed_cases,
            }
        )
        return base_result

    for role in relation["roles"]:
        try:
            if relation["operation"] == "equal":
                reference = relation["members"][0]
                reference_data = relation_output_path(
                    reference, role, case_results
                ).read_bytes()
                role_metrics: List[Dict[str, Any]] = []
                for member in relation["members"][1:]:
                    member_data = relation_output_path(
                        member, role, case_results
                    ).read_bytes()
                    metric = compare_exact_bytes(
                        reference_data,
                        member_data,
                        f"relation:{relation['id']}:{role}",
                        recorder,
                        relation_reference_name(reference, role),
                        relation_reference_name(member, role),
                    )
                    role_metrics.append(
                        {
                            "left": reference,
                            "right": member,
                            "comparison": metric,
                        }
                    )
                base_result["metrics"][role] = role_metrics
            else:
                part_data = [
                    relation_output_path(part, role, case_results).read_bytes()
                    for part in relation["parts"]
                ]
                result_reference = relation["result"]
                result_data = relation_output_path(
                    result_reference, role, case_results
                ).read_bytes()
                metric = compare_exact_bytes(
                    b"".join(part_data),
                    result_data,
                    f"relation:{relation['id']}:{role}",
                    recorder,
                    " + ".join(
                        relation_reference_name(part, role)
                        for part in relation["parts"]
                    ),
                    relation_reference_name(result_reference, role),
                )
                base_result["metrics"][role] = {
                    "parts": relation["parts"],
                    "result": result_reference,
                    "comparison": metric,
                }
        except (ComparisonError, OSError) as exc:
            base_result["errors"].append(f"{relation['id']}:{role}: {exc}")

    base_result["status"] = "PASS" if not base_result["errors"] else "FAIL"
    return base_result


def evaluate_relations(
    relations: Sequence[Mapping[str, Any]],
    selected_case_ids: Set[str],
    case_results: Mapping[str, Mapping[str, Any]],
    recorder: DifferenceRecorder,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for relation in relations:
        result = evaluate_relation(
            relation, selected_case_ids, case_results, recorder
        )
        results.append(result)
        if result["status"] == "PASS":
            print(f"[relation:{relation['id']}] PASS")
        elif result["status"] == "FAIL":
            print(f"[relation:{relation['id']}] FAIL: {result['errors'][0]}")
    return results


def cleanup_case_workdirs(
    case_results: Sequence[MutableMapping[str, Any]],
    relation_results: Sequence[Mapping[str, Any]],
    invocation_root: Path,
    keep_all: bool,
) -> None:
    retained_case_ids = {
        case_result["id"]
        for case_result in case_results
        if case_result["status"] != "PASS"
    }
    for relation_result in relation_results:
        if relation_result["status"] == "FAIL":
            retained_case_ids.update(relation_result["case_ids"])

    for case_result in case_results:
        retain = keep_all or case_result["id"] in retained_case_ids
        case_result["workdirs_retained"] = retain
        if not retain:
            case_root = invocation_root / "cases" / case_result["id"]
            if case_root.exists():
                shutil.rmtree(case_root)
            capture_root = invocation_root / "captures" / case_result["id"]
            if capture_root.exists():
                shutil.rmtree(capture_root)


def select_cases(
    manifest: Mapping[str, Any], suite: str, patterns: Sequence[str]
) -> List[Mapping[str, Any]]:
    selected = [case for case in manifest["cases"] if suite in case["suites"]]
    if patterns:
        selected = [
            case
            for case in selected
            if any(fnmatch.fnmatchcase(case["id"], pattern) for pattern in patterns)
        ]
    if not selected:
        raise ConfigurationError("No cases selected")
    return selected


def write_report(report: Mapping[str, Any]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = str(
        report.get(
            "run_id", dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    report_path = REPORT_DIR / f"report-{stamp}.json"
    report_text = json.dumps(report, indent=2, sort_keys=False) + "\n"
    report_path.write_text(report_text, encoding="utf-8")
    temporary_latest = REPORT_DIR / f".latest-{stamp}.tmp"
    temporary_latest.write_text(report_text, encoding="utf-8")
    os.replace(temporary_latest, REPORT_DIR / "latest.json")
    return report_path


def first_visible_device(variable: str) -> str:
    value = os.environ.get(variable)
    if value is None:
        return "0"
    return value.split(",", 1)[0].strip()


def resolve_device_environment(device: Optional[str]) -> Dict[str, str]:
    if device is not None:
        return {"CUDA_VISIBLE_DEVICES": device, "HIP_VISIBLE_DEVICES": device}
    return {
        "CUDA_VISIBLE_DEVICES": first_visible_device("CUDA_VISIBLE_DEVICES"),
        "HIP_VISIBLE_DEVICES": first_visible_device("HIP_VISIBLE_DEVICES"),
    }


def main() -> int:
    try:
        manifest = read_manifest_data()
        args = parse_arguments(manifest)
        if not math.isfinite(args.timeout_scale) or args.timeout_scale <= 0:
            raise ConfigurationError("--timeout-scale must be positive")
        repo_root = Path(args.repo_root).resolve()
        if not repo_root.is_dir():
            raise ConfigurationError(f"Repository root does not exist: {repo_root}")
        validate_manifest(manifest, repo_root)
        selected = select_cases(manifest, args.suite, args.case)

        if args.check_manifest:
            print(
                f"Manifest OK: {len(manifest['cases'])} cases, "
                f"{len(available_suites(manifest))} suites"
            )
            return 0
        if args.list:
            for case in selected:
                print(f"{case['id']:<40} {case['description']}")
            print(f"{len(selected)} case(s)")
            return 0

        baseline = resolve_command_path(args.baseline, repo_root)
        candidate = resolve_command_path(args.candidate, repo_root)
        for role, executable in (("baseline", baseline), ("candidate", candidate)):
            if not executable.is_file():
                raise ConfigurationError(f"{role} executable does not exist: {executable}")
            if not os.access(executable, os.X_OK):
                raise ConfigurationError(f"{role} executable is not executable: {executable}")

        run_id = (
            dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            + f"-p{os.getpid()}"
        )
        invocation_root = (WORK_ROOT / run_id).resolve()
        require_inside(invocation_root, WORK_ROOT, "invocation work directory")
        invocation_root.mkdir(parents=True)
        device_environment = resolve_device_environment(args.device)
        report: MutableMapping[str, Any] = {
            "schema_version": 2,
            "started_utc": utc_now(),
            "suite": args.suite,
            "case_patterns": args.case,
            "requested_device": args.device,
            "device_environment": device_environment,
            "keep_all": args.keep_all,
            "run_id": run_id,
            "work_root": str(invocation_root),
            "repo_root": str(repo_root),
            "baseline": executable_metadata(baseline),
            "candidate": executable_metadata(candidate),
            "manifest_sha256": sha256_file(MANIFEST_PATH),
            "cases": [],
        }
        recorder = DifferenceRecorder(invocation_root)
        for case in selected:
            report["cases"].append(
                execute_case(
                    case,
                    manifest,
                    baseline,
                    candidate,
                    invocation_root,
                    repo_root,
                    device_environment,
                    args.timeout_scale,
                    recorder,
                )
            )
        case_results_by_id = {
            case_result["id"]: case_result for case_result in report["cases"]
        }
        report["relations"] = evaluate_relations(
            manifest.get("relations", []),
            set(case_results_by_id),
            case_results_by_id,
            recorder,
        )
        cleanup_case_workdirs(
            report["cases"],
            report["relations"],
            invocation_root,
            args.keep_all,
        )
        report["finished_utc"] = utc_now()
        failed = [case for case in report["cases"] if case["status"] != "PASS"]
        failed_relations = [
            relation
            for relation in report["relations"]
            if relation["status"] == "FAIL"
        ]
        passed_relations = [
            relation
            for relation in report["relations"]
            if relation["status"] == "PASS"
        ]
        skipped_relations = [
            relation
            for relation in report["relations"]
            if relation["status"] == "SKIP"
        ]
        report["summary"] = {
            "passed": len(report["cases"]) - len(failed),
            "failed": len(failed),
            "total": len(report["cases"]),
            "relations_passed": len(passed_relations),
            "relations_failed": len(failed_relations),
            "relations_skipped": len(skipped_relations),
            "relations_total": len(report["relations"]),
        }
        report_path = write_report(report)
        if not failed and not failed_relations and not args.keep_all:
            shutil.rmtree(invocation_root)
        print(
            f"Summary: {report['summary']['passed']} passed, "
            f"{report['summary']['failed']} failed; "
            f"relations: {report['summary']['relations_passed']} passed, "
            f"{report['summary']['relations_failed']} failed, "
            f"{report['summary']['relations_skipped']} skipped; "
            f"report: {report_path.relative_to(PACKAGE_ROOT)}"
        )
        return 1 if failed or failed_relations else 0
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
