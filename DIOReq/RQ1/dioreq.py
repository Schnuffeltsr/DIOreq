# dioreq.py
"""
DIOReq: Multi-View Dependency-Guided Requirement-Gap Diagnosis and
Evidence-Grounded Completion.

Core implementation and experiment runner for the proposed method only.
Comparison baselines and evaluation procedures are kept outside this module.

Command line:
    python dioreq.py --dataset data/documents.json \
                     --output results/dioreq_raw.json \
                     --runs 3 --dependency-budget 20
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
from openai import OpenAI


# ============================================================
# 1. Configuration
# ============================================================

@dataclass(frozen=True)
class ModelConfig:
    extraction_model: str
    validation_model: str
    generation_model: str

    # None means "do not send a temperature field", for models that accept
    # only their built-in sampling temperature.
    extraction_temperature: float | None = 0.0
    validation_temperature: float | None = 0.0
    generation_temperature: float | None = 0.2


@dataclass(frozen=True)
class PipelineConfig:
    dependency_view: bool = True
    isolation_view: bool = True
    operation_view: bool = True

    cross_requirement_extraction: bool = True
    use_dps_ranking: bool = True

    evidence_validation: bool = True
    coverage_validation: bool = True
    boundary_validation: bool = True

    consolidation: bool = True
    refinement: bool = True
    deduplication: bool = True
    final_filtering: bool = True

    dependency_budget: int = 20
    max_parents: int = 12


def env_temperature(name: str, default: float) -> float | None:
    """
    Read a temperature override from the environment.

    The literal value ``none`` (or ``default``) disables the field, which is
    required by reasoning models that accept only their built-in
    temperature. An unparseable value falls back to ``default`` rather than
    aborting the run.
    """
    raw = os.getenv(name)

    if raw is None or not raw.strip():
        return default

    raw = raw.strip().lower()

    if raw in {"none", "default", "off"}:
        return None

    try:
        return float(raw)
    except ValueError:
        return default


def default_model_config() -> ModelConfig:
    """
    Build the model configuration from the current environment.

    This is evaluated on each call rather than once at import time, so
    environment variables set after this module has been imported still
    take effect. Importing a module-level constant would silently freeze
    the environment as it was at import.
    """
    return ModelConfig(
        extraction_model=os.getenv(
            "DIOREQ_EXTRACTION_MODEL", "gpt-5.5"
        ),
        validation_model=os.getenv(
            "DIOREQ_VALIDATION_MODEL", "gpt-5.5"
        ),
        generation_model=os.getenv(
            "DIOREQ_GENERATION_MODEL", "gpt-5.5"
        ),
        extraction_temperature=env_temperature(
            "DIOREQ_EXTRACTION_TEMPERATURE", 0.0
        ),
        validation_temperature=env_temperature(
            "DIOREQ_VALIDATION_TEMPERATURE", 0.0
        ),
        generation_temperature=env_temperature(
            "DIOREQ_GENERATION_TEMPERATURE", 0.2
        ),
    )


VALID_ELEMENT_TYPES = {
    "FUNCTION",
    "DATA",
    "ACTOR",
    "STATE",
    "EVENT",
    "CONSTRAINT",
}

VALID_RELATION_TYPES = {
    "ENABLES",
    "TRIGGERS",
    "PREVENTS",
    "MODIFIES",
    "CONTROLS",
    "PRODUCES",
    "DEPENDS_ON",
}

RELATION_REVIEW_PATTERNS = {
    "ENABLES": [
        "MISSING_PRECONDITION",
        "MISSING_UNAVAILABILITY_RESPONSE",
        "MISSING_FALLBACK",
    ],
    "DEPENDS_ON": [
        "MISSING_PRECONDITION",
        "MISSING_DEPENDENCY_FAILURE_RESPONSE",
        "MISSING_FALLBACK",
    ],
    "TRIGGERS": [
        "MISSING_TRIGGERED_RESPONSE",
        "MISSING_NOTIFICATION",
    ],
    "PREVENTS": [
        "MISSING_ENFORCEMENT",
        "MISSING_DENIAL_RESPONSE",
    ],
    "CONTROLS": [
        "MISSING_AUTHORIZATION",
        "MISSING_ACCESS_DENIAL",
        "MISSING_AUDIT",
    ],
    "MODIFIES": [
        "MISSING_STATE_PROPAGATION",
        "MISSING_CONSISTENCY_HANDLING",
        "MISSING_ROLLBACK",
    ],
    "PRODUCES": [
        "MISSING_CONSUMPTION",
        "MISSING_STORAGE",
        "MISSING_VALIDATION",
        "MISSING_PRODUCTION_FAILURE_HANDLING",
    ],
}


# ============================================================
# 2. Data models
# ============================================================

@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    api_calls: int = 0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.api_calls += other.api_calls


@dataclass
class RequirementElement:
    name: str
    element_type: str
    confidence: float
    source_frs: list[str]
    supporting_excerpts: list[str]
    aliases: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class DependencyRelation:
    source: str
    target: str
    relation_type: str
    confidence: float
    source_frs: list[str]
    supporting_excerpts: list[str]
    justification: str = ""
    cross_requirement: bool = False


@dataclass
class RankingRecord:
    record_id: str
    source: str
    target: str

    # Ranking signals
    dps: float
    extraction_support: float
    shortest_path_length: int

    # Common diagnostic context shared by all ranking strategies.
    # The nomination stage always receives this deterministic shortest path.
    path: list[str]
    relation_types: list[str]

    # Audit metadata only. This path is used to calculate Extraction Support
    # but is not passed to the nomination prompt.
    strongest_support_path: list[str] = field(default_factory=list)


@dataclass
class DiagnosticFinding:
    finding_id: str
    view: str
    gap_type: str

    source_frs: list[str] = field(default_factory=list)
    supporting_excerpts: list[str] = field(default_factory=list)

    source_element: str | None = None
    target_element: str | None = None
    affected_element: str | None = None
    relation_type: str | None = None
    dependency_path: list[str] = field(default_factory=list)

    dps: float | None = None
    nomination_reason: str = ""

    evidence_decision: str | None = None
    coverage_decision: str | None = None
    boundary_decision: str | None = None

    evidence_reason: str = ""
    coverage_reason: str = ""
    boundary_reason: str = ""

    def eligible_for_generation(self) -> bool:
        return (
            self.evidence_decision == "YES"
            and self.coverage_decision == "NO"
            and self.boundary_decision == "YES"
        )


@dataclass
class GeneratedCandidate:
    candidate_id: str
    requirement: str
    diagnostic_view: str
    gap_type: str
    source_frs: list[str]
    supporting_excerpts: list[str]
    relation_type: str | None = None
    dependency_path: list[str] = field(default_factory=list)
    affected_element: str | None = None
    dps: float | None = None

    # Traceability back to the raw candidates that were merged into this one.
    # A kept group can absorb several raw candidates, so without this field
    # the refinement pass is not reversible. Stored on demoted and removed
    # records too, so every raw candidate is accounted for exactly once.
    source_candidate_ids: list[str] = field(default_factory=list)


@dataclass
class ConsolidationResult:
    """
    Outcome of the Stage-5 refinement pass.

    kept     - consolidated requirements that go into the requirements document
    demoted  - implementation-level content, retained as design constraints
    removed  - rejected content, retained only for the refinement report

    Both demoted and removed entries are plain dicts with the consolidated
    text, the contributing candidate identifiers, and the deciding reason.
    """
    kept: list[GeneratedCandidate] = field(default_factory=list)
    demoted: list[dict[str, Any]] = field(default_factory=list)
    removed: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DocumentRecord:
    system_id: str
    document_id: str
    text: str


# ============================================================
# 3. General utilities
# ============================================================

def stable_id(prefix: str, *values: Any) -> str:
    raw = "||".join(str(value) for value in values)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    # utf-8-sig transparently strips a leading byte-order mark, which
    # Windows editors (Notepad, PowerShell Set-Content, Excel exports) add
    # by default. Plain "utf-8" would raise a confusing JSONDecodeError on
    # such files.
    with Path(path).open("r", encoding="utf-8-sig") as file:
        return json.load(file)


# Requirement identifiers a source document can carry. Ordered so that the
# most specific form wins at a position: a compound id ("FR-DCI-001") must not
# be read as the shorter "F1", and "3.1.1" must not be read as "3.1".
#
# The 'FR' forms require a separator before the number ("FR-001", "FR-DCI-001",
# "FR 1"). Without it, a prose line such as "FRESH 2024 targets ..." is read as
# a requirement identifier.
FR_ID_PATTERN = r"FR(?:(?:-[A-Z]{1,6})?-\d+|\s+\d+)"

REQUIREMENT_ID_PATTERN = (
    FR_ID_PATTERN
    + r"|Function Requirement\s+\d+"
    r"|\d+\.\d+\.\d+"
    r"|\d+\.\d+"
    r"|F\d+"
)

REQUIREMENT_MARKER_PATTERN = f"({REQUIREMENT_ID_PATTERN})"

# Line-start anchors that begin a new requirement block, ordered from
# requirement-level identifiers down to section-level numbering.
#
# The ordering is not the selection rule -- the split with the MOST blocks
# wins, and this order only breaks ties. Depth is what matters: a document
# that numbers its sections "2.1" and its requirements "FR-001", or that
# nests "3.1.1" under "3.1", must be split at the requirement, and splitting
# at the section instead silently turns a whole section into one item.
#
# Each anchor therefore covers exactly one level. Merging the identifier and
# section forms into a single pattern would cut at both levels at once and
# produce more blocks than either, which is not a split any document uses.
FR_SPLIT_PATTERNS = [
    # "### FR-01 Vehicle Registration", "FR-001", "FR-DCI-001:", "FR 1: A"
    rf"(?mi)(?=^#{{0,6}}\s*(?:{FR_ID_PATTERN}))",
    # "Function Requirement 1: Register a user"
    r"(?m)(?=^Function Requirement\s+\d+)",
    # "F1 User Registration"
    r"(?m)(?=^F\d+\s)",
    # Numbered headings, deepest level first: "3.1.1", "## 2.1", "2.1"
    r"(?m)(?=^\d+\.\d+\.\d+\s)",
    r"(?m)(?=^#{1,6}\s+\d+\.\d+\s)",
    r"(?m)(?=^\d+\.\d+\s)",
]


def _requirement_splits(text: str) -> list[list[str]]:
    """Every non-degenerate way of cutting ``text`` into requirement blocks."""
    candidates = []

    for pattern in FR_SPLIT_PATTERNS:
        parts = [
            item.strip()
            for item in re.split(pattern, text.strip())
            if item.strip()
        ]

        if len(parts) >= 2:
            candidates.append(parts)

    return candidates


# A top-level section heading: "2. Functional Requirements", "3. External
# Interfaces". Deliberately excludes "2.1", which has no space after the dot.
TOP_LEVEL_SECTION_PATTERN = re.compile(
    r"(?m)^(\d+)\.\s+(?!\d)(\S.*?)\s*$"
)


def requirements_scope(text: str) -> str:
    """
    Restrict a document to its functional-requirements section, if it has one.

    Specifications commonly continue past the functional requirements into
    material that is not a requirement at all -- external interface
    definitions, data dictionaries, non-functional targets. Those sections
    often reuse the same "x.y" numbering, so segmenting the whole document
    would silently promote four interface definitions into four requirements.

    Returns the section from the first top-level heading whose title names
    requirements, up to the next top-level heading. When the document has no
    such section, or the section is empty, the text is returned unchanged.
    """
    sections = list(TOP_LEVEL_SECTION_PATTERN.finditer(text))

    for index, match in enumerate(sections):
        if not re.search(r"requirement", match.group(2), re.IGNORECASE):
            continue

        start = match.start()
        end = (
            sections[index + 1].start()
            if index + 1 < len(sections)
            else len(text)
        )
        body = text[start:end].strip()

        if body:
            return body

    return text


def split_into_frs(text: str) -> list[tuple[str, str]]:
    """
    Cut a requirements document into ``(requirement_id, block)`` pairs.

    The deepest matching numbering level wins. Taking the first matching
    dialect instead under-segments every document whose sections and
    requirements are numbered at different depths, which is the common case
    in the reference corpus: a "2.1" section holding a dozen "FR-001" items
    would arrive at element extraction as one 4,000-character requirement.
    """
    parts: list[str] = []

    scoped = requirements_scope(text)
    candidates = _requirement_splits(scoped)

    if candidates:
        # Longest split wins; ``max`` keeps the first on a tie, so the
        # pattern order above breaks equal-length ties deterministically.
        parts = max(candidates, key=len)
    else:
        parts = [
            item.strip()
            for item in re.split(r"\n{2,}", scoped.strip())
            if item.strip()
        ]

    marker = re.compile(REQUIREMENT_MARKER_PATTERN, re.IGNORECASE)

    # Most specifications open with a title or section heading before the
    # first requirement ("Functional Requirement", "3 Requirements", ...).
    # That preamble carries no requirement identifier, but naming it
    # BLOCK-1 would pay for an element-extraction call on a bare heading and
    # let heading text reach the graph as if it were a requirement. Drop it
    # only when the remaining blocks are clearly numbered, so the
    # no-dialect fallback below still returns every block.
    if (
        len(parts) >= 2
        and not marker.search(parts[0].splitlines()[0])
        and any(marker.search(part.splitlines()[0]) for part in parts[1:])
    ):
        parts = parts[1:]

    results = []

    for index, block in enumerate(parts, start=1):
        first_line = block.splitlines()[0].strip()

        match = marker.search(first_line)

        requirement_id = match.group(1) if match else f"BLOCK-{index}"
        results.append((requirement_id, block))

    return results


def flatten_unique(values: list[list[str]]) -> list[str]:
    seen = set()
    output = []

    for group in values:
        for value in group:
            value = str(value).strip()
            if value and value not in seen:
                seen.add(value)
                output.append(value)

    return output


# ------------------------------------------------------------
# 3.1 Defensive coercion of model-produced fields
# ------------------------------------------------------------
#
# Every value below comes from an LLM response, so its type is not
# guaranteed by anything except the prompt. A single JSON null where an
# object or list was requested used to raise AttributeError/TypeError and
# abort the whole run. These helpers make each field degrade to empty
# instead, so one malformed field costs at most the item that contained it.

def coerce_mapping(value: Any) -> dict:
    """Return ``value`` when it is a mapping, otherwise an empty dict."""
    return value if isinstance(value, dict) else {}


def coerce_items(value: Any) -> list[dict]:
    """Return only the mapping entries of a model-produced list field."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def coerce_float(value: Any, default: float = 0.0) -> float:
    """Return ``value`` as a float, or ``default`` if it is not numeric."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def coerce_confidence(value: Any) -> float:
    """
    Normalize a model-reported confidence to the unit interval.

    Endpoints occasionally report confidence on a 0-100 scale even though
    the prompt asks for 0-1. A bare clamp would silently turn 90 into 1.0,
    which passes every confidence threshold and corrupts the extraction
    filter, so values above 1 are read as percentages. Anything that is
    still out of range, or non-numeric, becomes 0.0 and is then rejected by
    the caller's threshold check.
    """
    score = coerce_float(value, default=-1.0)

    if score < 0.0:
        return 0.0

    if score > 1.0:
        if score <= 100.0:
            score = score / 100.0
        else:
            return 0.0

    return min(max(score, 0.0), 1.0)


def coerce_str(value: Any, default: str = "") -> str:
    """Return a stripped string, never the text ``"None"``."""
    if value is None:
        return default
    return str(value).strip()


def coerce_str_list(value: Any) -> list[str]:
    """Return a list of non-empty stripped strings; scalars wrap."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    return [
        str(item).strip()
        for item in value
        if item is not None and str(item).strip()
    ]


# ============================================================
# 4. OpenAI-compatible JSON client
# ============================================================

# Endpoints differ in how strictly they enforce the OpenAI contract for
# response_format={"type": "json_object"}: the messages must literally
# contain the word "json". The prompt constants below already say "JSON",
# but this guard keeps a future prompt edit from silently breaking every
# call in the pipeline.
JSON_GUARD = (
    "\nReturn your answer as a single JSON object and nothing else."
)

# Some reasoning models accept only their default temperature. When an
# endpoint rejects the configured value the request is retried once without
# the temperature field, so a model-level restriction never aborts a run.
TEMPERATURE_ERROR_MARKERS = (
    "temperature",
    "unsupported_value",
    "unsupported value",
)


class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL") or None,
        )
        self.usage = Usage()

    @staticmethod
    def _is_temperature_error(error: Exception) -> bool:
        text = str(error).lower()
        return (
            "temperature" in text
            and any(
                marker in text
                for marker in TEMPERATURE_ERROR_MARKERS
            )
        )

    def _create(self, model: str, messages: list[dict], temperature):
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }

        if temperature is not None:
            kwargs["temperature"] = temperature

        return self.client.chat.completions.create(**kwargs)

    def json_call(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        retries: int = 3,
    ) -> dict:
        """
        Call the model and return a parsed JSON object.

        Retries cover three distinct failure modes:
        - transport and HTTP errors (with exponential backoff),
        - a model that refuses the requested temperature (retried without it),
        - a response that is not a JSON object (retried as a format error).
        """
        if not re.search(
            r"json",
            f"{system_prompt}\n{user_prompt}",
            re.IGNORECASE,
        ):
            user_prompt = user_prompt + JSON_GUARD

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error: Exception | None = None
        send_temperature = temperature

        for attempt in range(retries):
            try:
                response = self._create(model, messages, send_temperature)

                usage = getattr(response, "usage", None)
                self.usage.api_calls += 1

                if usage:
                    self.usage.prompt_tokens += int(
                        getattr(usage, "prompt_tokens", 0) or 0
                    )
                    self.usage.completion_tokens += int(
                        getattr(usage, "completion_tokens", 0) or 0
                    )

                content = response.choices[0].message.content or "{}"
                parsed = json.loads(content)

                if not isinstance(parsed, dict):
                    raise ValueError(
                        "The model returned a JSON "
                        f"{type(parsed).__name__} where a JSON object "
                        "was required"
                    )

                return parsed

            except Exception as error:
                last_error = error

                if (
                    send_temperature is not None
                    and self._is_temperature_error(error)
                ):
                    # The endpoint refuses this temperature; drop it and
                    # retry immediately with the model default.
                    send_temperature = None
                    continue

                time.sleep(2 ** attempt)

        raise RuntimeError(f"LLM JSON call failed: {last_error}")


# ============================================================
# 5. Typed element extraction
# ============================================================

ELEMENT_SYSTEM_PROMPT = """
You are a requirements-engineering information extractor.

Extract only document-supported, dependency-relevant requirement elements.

Allowed types:
FUNCTION, DATA, ACTOR, STATE, EVENT, CONSTRAINT.

Definitions:
- FUNCTION: an independent business capability or system operation.
- DATA: a persistent business object, record, document, or artifact.
- ACTOR: a human role or external system.
- STATE: a persistent or decision-relevant process condition.
- EVENT: an occurrence that triggers behavior or a transition.
- CONSTRAINT: a policy, authorization condition, validation rule, or
  behavioral restriction.

Use titles, inputs, outputs, and descriptions.

Type-specific rules:
- DATA must have an independent business lifecycle.
- ACTOR does not need a data lifecycle.
- EVENT may be transient when it explicitly triggers behavior.
- STATE must affect a decision, transition, or downstream operation.
- CONSTRAINT must govern, authorize, validate, or restrict behavior.
- FUNCTION must be a complete business capability.

Exclude UI labels, implementation technologies, atomic field values,
unsupported concepts, and generic verb fragments.

Confidence represents documentary extraction support, not importance.
Every element must contain an exact supporting excerpt.
"""

ELEMENT_USER_TEMPLATE = """
Requirement identifier:
{requirement_id}

Requirement:
{requirement_text}

Return a single JSON object:
{{
  "elements": [
    {{
      "name": "canonical name",
      "type": "FUNCTION|DATA|ACTOR|STATE|EVENT|CONSTRAINT",
      "aliases": [],
      "confidence": 0.90,
      "source_fr": "{requirement_id}",
      "supporting_excerpt": "exact excerpt",
      "description": "brief documentary meaning"
    }}
  ]
}}

Use confidence >= 0.75. Return at most eight elements.
Do not invent missing concepts.
"""


def extract_elements(
    llm: LLMClient,
    document_text: str,
    model_config: ModelConfig,
    progress: Any = None,
) -> list[RequirementElement]:
    extracted: list[RequirementElement] = []
    blocks = split_into_frs(document_text)

    for index, (requirement_id, requirement_text) in enumerate(blocks, 1):
        # One call per requirement block, so a 90-block document is minutes of
        # silence without this. Reported before the call, not after, so the
        # message names what is currently in flight.
        if progress is not None:
            progress(
                f"  elements {index}/{len(blocks)}: {requirement_id}"
            )

        result = llm.json_call(
            model=model_config.extraction_model,
            system_prompt=ELEMENT_SYSTEM_PROMPT,
            user_prompt=ELEMENT_USER_TEMPLATE.format(
                requirement_id=requirement_id,
                requirement_text=requirement_text,
            ),
            temperature=model_config.extraction_temperature,
        )

        for item in coerce_items(result.get("elements")):
            element_type = coerce_str(item.get("type")).upper()
            confidence = coerce_confidence(item.get("confidence"))

            if element_type not in VALID_ELEMENT_TYPES:
                continue
            if confidence < 0.75:
                continue

            name = coerce_str(item.get("name"))
            excerpt = coerce_str(item.get("supporting_excerpt"))

            if not name or not excerpt:
                continue

            extracted.append(RequirementElement(
                name=name,
                element_type=element_type,
                confidence=confidence,
                source_frs=[requirement_id],
                supporting_excerpts=[excerpt],
                aliases=coerce_str_list(item.get("aliases")),
                description=coerce_str(item.get("description")),
            ))

    return merge_elements(extracted)


def merge_elements(
    elements: list[RequirementElement],
) -> list[RequirementElement]:
    grouped: dict[tuple[str, str], list[RequirementElement]] = {}

    for element in elements:
        key = (
            re.sub(r"\s+", " ", element.name.lower().strip()),
            element.element_type,
        )
        grouped.setdefault(key, []).append(element)

    merged = []

    for group in grouped.values():
        representative = max(group, key=lambda item: item.confidence)

        merged.append(RequirementElement(
            name=representative.name,
            element_type=representative.element_type,
            confidence=max(item.confidence for item in group),
            source_frs=flatten_unique(
                [item.source_frs for item in group]
            ),
            supporting_excerpts=flatten_unique(
                [item.supporting_excerpts for item in group]
            ),
            aliases=flatten_unique(
                [item.aliases for item in group]
            ),
            description=representative.description,
        ))

    return sorted(merged, key=lambda item: item.name.lower())


# ============================================================
# 6. Directional dependency extraction
# ============================================================

DEPENDENCY_SYSTEM_PROMPT = """
You extract document-supported directional dependencies from requirements.

Allowed relation types:
- ENABLES: source makes target possible.
- TRIGGERS: source initiates target.
- PREVENTS: source blocks target.
- MODIFIES: source changes target state or content.
- CONTROLS: source governs permission, validity, or execution of target.
- PRODUCES: source creates information or state consumed by target.
- DEPENDS_ON: target requires source.

Normalize every relation to source-to-target direction.

If the document says X depends on Y, encode:
Y --DEPENDS_ON--> X.

Confidence represents documentary extraction support. It is not causal
strength, probability, severity, or business importance.

Do not create relations merely to connect disconnected elements.
Do not introduce endpoints outside the supplied element list.
Every relation must contain exact supporting evidence.
"""

INTRA_RELATION_TEMPLATE = """
Fixed canonical elements:
{elements}

Requirement identifier:
{requirement_id}

Requirement:
{requirement_text}

Return a single JSON object:
{{
  "relations": [
    {{
      "source": "canonical element",
      "target": "canonical element",
      "relation_type": "ENABLES|TRIGGERS|PREVENTS|MODIFIES|CONTROLS|PRODUCES|DEPENDS_ON",
      "confidence": 0.90,
      "source_frs": ["{requirement_id}"],
      "supporting_excerpts": ["exact excerpt"],
      "justification": "brief documentary explanation",
      "cross_requirement": false
    }}
  ]
}}

Use confidence >= 0.75. Reject self-loops and unsupported relations.
"""

CROSS_RELATION_TEMPLATE = """
Fixed canonical elements:
{elements}

Existing relations:
{existing_relations}

Currently disconnected elements:
{disconnected_elements}

Complete requirements document:
{document_text}

Find only cross-requirement directional dependencies jointly supported by
different requirements.

Inspect producer-consumer relations, shared authorization constraints,
distributed prerequisites, events and downstream responses, state changes,
and documented cross-module interactions.

Return a single JSON object:
{{
  "relations": [
    {{
      "source": "canonical element",
      "target": "canonical element",
      "relation_type": "ENABLES|TRIGGERS|PREVENTS|MODIFIES|CONTROLS|PRODUCES|DEPENDS_ON",
      "confidence": 0.80,
      "source_frs": ["FR-A", "FR-B"],
      "supporting_excerpts": ["exact excerpt A", "exact excerpt B"],
      "justification": "joint documentary support",
      "cross_requirement": true
    }}
  ]
}}

Disconnected elements are review priorities only. Do not connect an element
unless the document supports the relation. Use confidence >= 0.60.
"""


def build_alias_map(
    elements: list[RequirementElement],
) -> dict[str, str]:
    aliases = {}

    for element in elements:
        aliases[element.name.lower()] = element.name

        for alias in element.aliases:
            aliases[alias.lower()] = element.name

    return aliases


def validate_relation_items(
    items: list[dict],
    elements: list[RequirementElement],
    min_confidence: float,
    cross_requirement: bool,
) -> list[DependencyRelation]:
    aliases = build_alias_map(elements)
    output = []

    for item in coerce_items(items):
        raw_source = coerce_str(item.get("source"))
        raw_target = coerce_str(item.get("target"))

        source = aliases.get(raw_source.lower())
        target = aliases.get(raw_target.lower())

        relation_type = coerce_str(
            item.get("relation_type")
        ).upper()

        confidence = coerce_confidence(item.get("confidence"))
        excerpts = coerce_str_list(item.get("supporting_excerpts"))

        if not source or not target or source == target:
            continue
        if relation_type not in VALID_RELATION_TYPES:
            continue
        if confidence < min_confidence:
            continue
        if not excerpts:
            continue

        output.append(DependencyRelation(
            source=source,
            target=target,
            relation_type=relation_type,
            confidence=confidence,
            source_frs=coerce_str_list(item.get("source_frs")),
            supporting_excerpts=excerpts,
            justification=coerce_str(item.get("justification")),
            cross_requirement=cross_requirement,
        ))

    return output


def extract_dependency_relations(
    llm: LLMClient,
    document_text: str,
    elements: list[RequirementElement],
    model_config: ModelConfig,
    use_cross_requirement_pass: bool = True,
    progress: Any = None,
) -> list[DependencyRelation]:
    relations: list[DependencyRelation] = []
    element_names = [element.name for element in elements]
    blocks = split_into_frs(document_text)
    considered = 0

    for requirement_id, requirement_text in blocks:
        relevant_elements = [
            element.name
            for element in elements
            if requirement_id in element.source_frs
        ]

        if len(relevant_elements) < 2:
            continue

        considered += 1

        if progress is not None:
            progress(
                f"  relations {considered}: {requirement_id} "
                f"({len(relevant_elements)} element(s))"
            )

        result = llm.json_call(
            model=model_config.extraction_model,
            system_prompt=DEPENDENCY_SYSTEM_PROMPT,
            user_prompt=INTRA_RELATION_TEMPLATE.format(
                elements=json.dumps(
                    relevant_elements,
                    ensure_ascii=False,
                ),
                requirement_id=requirement_id,
                requirement_text=requirement_text,
            ),
            temperature=model_config.extraction_temperature,
        )

        relations.extend(validate_relation_items(
            result.get("relations"),
            elements,
            min_confidence=0.75,
            cross_requirement=False,
        ))

    relations = deduplicate_relations(relations)

    if use_cross_requirement_pass:
        graph = nx.Graph()
        graph.add_nodes_from(element_names)

        for relation in relations:
            graph.add_edge(relation.source, relation.target)

        disconnected = sorted(
            node for node in graph.nodes if graph.degree(node) == 0
        )

        result = llm.json_call(
            model=model_config.extraction_model,
            system_prompt=DEPENDENCY_SYSTEM_PROMPT,
            user_prompt=CROSS_RELATION_TEMPLATE.format(
                elements=json.dumps(element_names, ensure_ascii=False),
                existing_relations=json.dumps(
                    [asdict(item) for item in relations],
                    ensure_ascii=False,
                ),
                disconnected_elements=json.dumps(
                    disconnected,
                    ensure_ascii=False,
                ),
                document_text=document_text,
            ),
            temperature=model_config.extraction_temperature,
        )

        relations.extend(validate_relation_items(
            result.get("relations"),
            elements,
            min_confidence=0.60,
            cross_requirement=True,
        ))

    return deduplicate_relations(relations)


def deduplicate_relations(
    relations: list[DependencyRelation],
) -> list[DependencyRelation]:
    grouped: dict[
        tuple[str, str, str],
        list[DependencyRelation],
    ] = {}

    for relation in relations:
        key = (
            relation.source,
            relation.target,
            relation.relation_type,
        )
        grouped.setdefault(key, []).append(relation)

    output = []

    for group in grouped.values():
        representative = max(group, key=lambda item: item.confidence)

        output.append(DependencyRelation(
            source=representative.source,
            target=representative.target,
            relation_type=representative.relation_type,
            confidence=max(item.confidence for item in group),
            source_frs=flatten_unique(
                [item.source_frs for item in group]
            ),
            supporting_excerpts=flatten_unique(
                [item.supporting_excerpts for item in group]
            ),
            justification=representative.justification,
            cross_requirement=any(
                item.cross_requirement for item in group
            ),
        ))

    return output


# ============================================================
# 7. Semantic graph, computational graph, and DPS
# ============================================================

def deterministic_shortest_path(
    graph: nx.DiGraph,
    source: str,
    target: str,
) -> list[str]:
    """
    Return a deterministic shortest directed path.

    NetworkX may return any one of several equally short paths depending on
    graph insertion order. For reproducibility, this function selects the
    lexicographically smallest path among all shortest paths.
    """
    if source not in graph or target not in graph:
        return []

    if source == target:
        return [source]

    if not nx.has_path(graph, source, target):
        return []

    shortest_paths = nx.all_shortest_paths(
        graph,
        source=source,
        target=target,
    )

    return list(min(
        (tuple(path) for path in shortest_paths),
        key=lambda path: path,
    ))


def _all_path_support_states(
    graph: nx.DiGraph,
    source: str,
) -> dict[str, dict[int, tuple[float, list[str]]]]:
    """
    Single-source dynamic program shared by every target reachable from
    `source`.

    Returns dp[node][edge_count] = (maximum log-support sum, path).

    Because the propagation is additive in log space, the best state for a
    given (node, edge_count) is optimal for every continuation, so inferior
    states can be discarded. Running this once per source covers all of its
    descendants, which is what makes per-target recomputation unnecessary.

    The graph must be a DAG; callers are responsible for that check.
    """
    relevant_nodes = nx.descendants(graph, source) | {source}

    topological_order = [
        node
        for node in nx.topological_sort(graph)
        if node in relevant_nodes
    ]

    dp: dict[str, dict[int, tuple[float, list[str]]]] = {
        node: {} for node in relevant_nodes
    }
    dp[source][0] = (0.0, [source])

    for node in topological_order:
        node_states = list(dp.get(node, {}).items())

        if not node_states:
            continue

        for successor in sorted(
            graph.successors(node),
            key=str,
        ):
            if successor not in relevant_nodes:
                continue

            edge_data = graph[node][successor]
            support = float(
                edge_data.get(
                    "extraction_support",
                    edge_data.get("weight", 0.0),
                )
            )

            # Extraction-support values must lie in (0, 1].
            # Zero-support edges cannot form a positive-support path.
            if support <= 0.0:
                continue

            support = min(support, 1.0)
            log_support = math.log(support)

            for edge_count, (log_sum, path) in node_states:
                new_edge_count = edge_count + 1
                new_log_sum = log_sum + log_support
                new_path = path + [successor]

                existing = dp[successor].get(new_edge_count)

                if existing is None:
                    dp[successor][new_edge_count] = (
                        new_log_sum,
                        new_path,
                    )
                    continue

                existing_log_sum, existing_path = existing

                if new_log_sum > existing_log_sum:
                    dp[successor][new_edge_count] = (
                        new_log_sum,
                        new_path,
                    )
                elif math.isclose(
                    new_log_sum,
                    existing_log_sum,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    # Deterministic tie-breaking.
                    if tuple(new_path) < tuple(existing_path):
                        dp[successor][new_edge_count] = (
                            new_log_sum,
                            new_path,
                        )

    return dp


def _best_path_support_from_states(
    states: dict[int, tuple[float, list[str]]],
) -> tuple[float, list[str]]:
    """Pick the maximum geometric-mean support from one node's DP states."""
    best_score = 0.0
    best_path: list[str] = []

    for edge_count, (log_sum, path) in states.items():
        if edge_count <= 0:
            continue

        score = math.exp(log_sum / edge_count)

        if score > best_score:
            best_score = score
            best_path = path
        elif math.isclose(
            score,
            best_score,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            # Use a stable path when several paths have the same score.
            if not best_path or tuple(path) < tuple(best_path):
                best_path = path

    return min(max(best_score, 0.0), 1.0), best_path


class AllPathSupportIndex:
    """
    Memoized access to Extraction Support over one computational graph.

    The single-source DP already covers every descendant of the source, so
    computing it once per source and reusing it for all targets removes a
    redundant factor of O(V) from the record-pool construction. Results are
    numerically identical to calling maximum_geometric_path_support per pair.
    """

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError(
                "Extraction Support requires an acyclic computational graph"
            )

        self._states: dict[
            str, dict[str, dict[int, tuple[float, list[str]]]]
        ] = {}

    def score(
        self,
        source: str,
        target: str,
    ) -> tuple[float, list[str]]:
        if source not in self.graph or target not in self.graph:
            return 0.0, []
        if source == target:
            return 0.0, []
        if not nx.has_path(self.graph, source, target):
            return 0.0, []

        if source not in self._states:
            self._states[source] = _all_path_support_states(
                self.graph,
                source,
            )

        return _best_path_support_from_states(
            self._states[source].get(target, {})
        )

    def release(self, source: str) -> None:
        """Drop cached states once a source's descendants are fully consumed."""
        self._states.pop(source, None)


def maximum_geometric_path_support(
    graph: nx.DiGraph,
    source: str,
    target: str,
) -> tuple[float, list[str]]:
    """
    Compute the Extraction Support baseline defined in the paper:

        ES(X, Y) =
            max over all directed paths p from X to Y of
            (product of c_e for e in p) ** (1 / |p|)

    Here, |p| is the number of edges on path p.

    The computational graph must be a DAG. The implementation uses dynamic
    programming in log space rather than explicitly enumerating all paths.

    For every node and feasible path length, it retains the path with the
    largest sum of log edge supports. Because all future extensions add the
    same suffix contribution to states of the same length and endpoint,
    inferior states can be safely discarded.

    Returns:
        score:
            Maximum geometric-mean extraction support.

        best_path:
            A deterministic path attaining the maximum score. This path is
            stored only as audit metadata and is not used as the nomination
            context in RQ3.
    """
    if source not in graph or target not in graph:
        return 0.0, []

    if source == target:
        return 0.0, []

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError(
            "Extraction Support requires an acyclic computational graph"
        )

    if not nx.has_path(graph, source, target):
        return 0.0, []

    return _best_path_support_from_states(
        _all_path_support_states(graph, source).get(target, {})
    )


class DependencyGraphs:
    def __init__(self, max_parents: int = 12):
        self.max_parents = max_parents
        self.semantic_graph = nx.MultiDiGraph()
        self.computational_graph = nx.DiGraph()
        self.exclusions: list[dict] = []

    def build(
        self,
        elements: list[RequirementElement],
        relations: list[DependencyRelation],
    ) -> None:
        self.semantic_graph.clear()
        self.computational_graph.clear()
        self.exclusions.clear()

        for element in elements:
            self.semantic_graph.add_node(
                element.name,
                element_type=element.element_type,
                confidence=element.confidence,
                source_frs=element.source_frs,
                supporting_excerpts=element.supporting_excerpts,
                description=element.description,
            )

        for index, relation in enumerate(relations):
            if relation.source not in self.semantic_graph:
                continue
            if relation.target not in self.semantic_graph:
                continue

            self.semantic_graph.add_edge(
                relation.source,
                relation.target,
                key=f"relation-{index}",
                relation_type=relation.relation_type,
                extraction_support=relation.confidence,
                source_frs=relation.source_frs,
                supporting_excerpts=relation.supporting_excerpts,
                justification=relation.justification,
                cross_requirement=relation.cross_requirement,
            )

        for node, data in self.semantic_graph.nodes(data=True):
            self.computational_graph.add_node(node, **data)

        pair_relations: dict[tuple[str, str], list[dict]] = {}

        for source, target, key, data in self.semantic_graph.edges(
            keys=True,
            data=True,
        ):
            pair_relations.setdefault((source, target), []).append({
                "key": key,
                **data,
            })

        for (source, target), candidates in pair_relations.items():
            selected = sorted(
                candidates,
                key=lambda item: (
                    -float(item["extraction_support"]),
                    str(item["relation_type"]),
                    str(item["key"]),
                ),
            )[0]

            self.computational_graph.add_edge(
                source,
                target,
                weight=float(selected["extraction_support"]),
                extraction_support=float(
                    selected["extraction_support"]
                ),
                relation_type=selected["relation_type"],
                semantic_edge_key=selected["key"],
            )

        self._remove_cycles()
        self._limit_parents()

        if not nx.is_directed_acyclic_graph(
            self.computational_graph
        ):
            raise RuntimeError("Computational graph is not acyclic")

    def _remove_cycles(self) -> None:
        while True:
            try:
                cycle = nx.find_cycle(
                    self.computational_graph,
                    orientation="original",
                )
            except nx.NetworkXNoCycle:
                break

            candidates = []

            for source, target, _ in cycle:
                data = self.computational_graph[source][target]
                candidates.append((
                    float(data.get("weight", 0.0)),
                    source,
                    target,
                ))

            weight, source, target = sorted(
                candidates,
                key=lambda item: (
                    item[0],
                    item[1],
                    item[2],
                ),
            )[0]

            edge_data = dict(
                self.computational_graph[source][target]
            )
            self.computational_graph.remove_edge(source, target)

            self.exclusions.append({
                "source": source,
                "target": target,
                "weight": weight,
                "reason": "cycle_removal",
                "edge_data": edge_data,
            })

    def _limit_parents(self) -> None:
        for node in list(self.computational_graph.nodes):
            incoming = list(
                self.computational_graph.in_edges(node, data=True)
            )

            if len(incoming) <= self.max_parents:
                continue

            retained = sorted(
                incoming,
                key=lambda item: (
                    -float(item[2].get("weight", 0.0)),
                    item[0],
                ),
            )[:self.max_parents]

            retained_pairs = {
                (source, target)
                for source, target, _ in retained
            }

            for source, target, data in incoming:
                if (source, target) in retained_pairs:
                    continue

                self.computational_graph.remove_edge(source, target)

                self.exclusions.append({
                    "source": source,
                    "target": target,
                    "weight": float(data.get("weight", 0.0)),
                    "reason": "parent_limit",
                    "edge_data": dict(data),
                })

    def isolated_semantic_elements(self) -> list[str]:
        return sorted(
            node
            for node in self.semantic_graph.nodes
            if self.semantic_graph.degree(node) == 0
        )

    def propagate(
        self,
        clamp_node: str | None = None,
        clamp_value: float | None = None,
    ) -> dict[str, float]:
        graph = self.computational_graph
        activation: dict[str, float] = {}

        for node in nx.topological_sort(graph):
            if node == clamp_node:
                activation[node] = float(clamp_value)
                continue

            parents = list(graph.predecessors(node))

            if not parents:
                activation[node] = 1.0
                continue

            value = sum(
                float(graph[parent][node]["weight"])
                * activation[parent]
                for parent in parents
            ) / len(parents)

            activation[node] = min(max(value, 0.0), 1.0)

        return activation

    def dps(self, source: str, target: str) -> float:
        graph = self.computational_graph

        if source == target:
            return 0.0
        if source not in graph or target not in graph:
            return 0.0
        if not nx.has_path(graph, source, target):
            return 0.0

        enabled = self.propagate(source, 1.0)
        disabled = self.propagate(source, 0.0)

        return min(
            max(enabled[target] - disabled[target], 0.0),
            1.0,
        )

    def ranking_records(self) -> list[RankingRecord]:
        """
        Build the common source-target record pool used by all RQ3 ranking
        strategies.

        Each record contains three independently calculated ranking signals:

        1. DPS:
           Normalized documentary-support contribution aggregated across paths.

        2. Extraction Support:
           Maximum geometric-mean edge support over all directed paths.

        3. Topology:
           Shortest directed-path length.

        All strategies share the same deterministic shortest path and relation
        labels as nomination context. Therefore, changing the ranking strategy
        changes only record ordering, not the information passed to nomination.
        """
        graph = self.computational_graph
        records: list[RankingRecord] = []

        # One single-source DP per source; reused across all of its targets.
        # States are released as soon as a source is finished so peak memory
        # stays bounded by a single source's reachable set.
        support_index = AllPathSupportIndex(graph)

        for source in sorted(graph.nodes, key=str):
            descendants = sorted(
                nx.descendants(graph, source),
                key=str,
            )

            for target in descendants:
                shortest_path = deterministic_shortest_path(
                    graph,
                    source,
                    target,
                )

                if len(shortest_path) < 2:
                    continue

                extraction_support, strongest_support_path = (
                    support_index.score(
                        source,
                        target,
                    )
                )

                # The diagnostic context is fixed across all ranking methods.
                # Do not replace this with strongest_support_path when the
                # Extraction Support strategy is selected.
                relation_types = [
                    str(
                        graph[parent][child].get(
                            "relation_type",
                            "",
                        )
                    )
                    for parent, child in zip(
                        shortest_path[:-1],
                        shortest_path[1:],
                    )
                ]

                records.append(RankingRecord(
                    record_id=stable_id(
                        "record",
                        source,
                        target,
                    ),
                    source=source,
                    target=target,
                    dps=self.dps(source, target),
                    extraction_support=extraction_support,
                    shortest_path_length=(
                        len(shortest_path) - 1
                    ),
                    path=shortest_path,
                    relation_types=sorted(set(relation_types)),
                    strongest_support_path=strongest_support_path,
                ))

            support_index.release(source)

        return records


# ============================================================
# 8. Ranking strategies
# ============================================================

def rank_records(
    records: list[RankingRecord],
    strategy: str,
    seed: int = 0,
) -> list[RankingRecord]:
    """
    Rank a fixed source-target record pool.

    For non-random strategies, score ties are resolved only by canonical
    source and target names. This avoids mixing one ranking signal into
    another through secondary score-based tie-breakers.
    """
    records = list(records)

    if strategy == "dps":
        return sorted(
            records,
            key=lambda item: (
                -item.dps,
                item.source,
                item.target,
            ),
        )

    if strategy == "extraction_support":
        return sorted(
            records,
            key=lambda item: (
                -item.extraction_support,
                item.source,
                item.target,
            ),
        )

    if strategy == "topology":
        return sorted(
            records,
            key=lambda item: (
                item.shortest_path_length,
                item.source,
                item.target,
            ),
        )

    if strategy == "unranked":
        return sorted(
            records,
            key=lambda item: (
                item.source,
                item.target,
            ),
        )

    if strategy == "random":
        # Sort before shuffling so that a fixed seed gives the same result
        # regardless of graph insertion order.
        records = sorted(
            records,
            key=lambda item: (
                item.source,
                item.target,
            ),
        )

        rng = random.Random(seed)
        rng.shuffle(records)
        return records

    raise ValueError(
        f"Unknown ranking strategy: {strategy}"
    )


# ============================================================
# 9. Three-view nomination
# ============================================================

DEPENDENCY_NOMINATION_PROMPT = """
You are reviewing a document-supported dependency region for potentially
missing business obligations.

Source element: {source}
Target element: {target}
Path: {path}
Relation types: {relation_types}
Applicable review patterns: {patterns}

Source requirements:
{source_requirements}

Return a single JSON object:
{{
  "findings": [
    {{
      "gap_type": "one applicable review pattern",
      "affected_element": "affected element",
      "source_frs": ["source requirement identifier"],
      "supporting_excerpts": ["exact excerpt"],
      "reason": "why this obligation should be checked"
    }}
  ]
}}

Nominate a review finding only when the supplied document supports inspecting
the obligation. Do not write the final supplementary requirement. Do not infer
new actors, entities, policies, or system responsibilities.
"""

ISOLATION_PROMPT = """
Classify a disconnected requirement element.

Element:
{element}

Type:
{element_type}

Source requirements:
{source_requirements}

Supporting excerpts:
{supporting_excerpts}

Complete document:
{document_text}

Return a single JSON object:
{{
  "label": "GENUINE_GAP|IMPLICIT_DEPENDENCY|OUT_OF_SCOPE|NOISE",
  "gap_type": "MISSING_INTEGRATION or another concise type",
  "reason": "classification reason",
  "source_frs": [],
  "supporting_excerpts": []
}}

GENUINE_GAP means an in-scope element lacks a documented integration.
IMPLICIT_DEPENDENCY means the element is used but its connection is
under-specified. OUT_OF_SCOPE and NOISE must not enter generation.
"""

OPERATION_PROMPT = """
Inspect persistent, in-scope business entities for applicable lifecycle and
management operations.

Elements:
{elements}

Complete requirements:
{document_text}

Check only operations applicable to each entity:
CREATE, READ, UPDATE, DELETE, AUTHORIZATION, AUDIT, VALIDATION,
EXCEPTION_HANDLING.

Return a single JSON object:
{{
  "findings": [
    {{
      "affected_element": "canonical entity",
      "gap_type": "MISSING_CREATE|MISSING_READ|MISSING_UPDATE|MISSING_DELETE|MISSING_AUTHORIZATION|MISSING_AUDIT|MISSING_VALIDATION|MISSING_EXCEPTION_HANDLING",
      "source_frs": [],
      "supporting_excerpts": [],
      "reason": "why the operation is applicable but under-specified"
    }}
  ]
}}

Do not assume that every entity requires every operation. Do not nominate
operations outside the documented system boundary.
"""


def requirement_context(
    document_text: str,
    requirement_ids: list[str],
) -> str:
    blocks = split_into_frs(document_text)
    selected = []

    for requirement_id, block in blocks:
        if not requirement_ids or requirement_id in requirement_ids:
            selected.append(block)

    return "\n\n".join(selected) if selected else document_text


def nominate_dependency_findings(
    llm: LLMClient,
    document_text: str,
    graphs: DependencyGraphs,
    records: list[RankingRecord],
    model_config: ModelConfig,
) -> list[DiagnosticFinding]:
    findings = []

    for record in records:
        source_frs = flatten_unique([
            list(graphs.semantic_graph.nodes[
                record.source
            ].get("source_frs", [])),
            list(graphs.semantic_graph.nodes[
                record.target
            ].get("source_frs", [])),
        ])

        patterns = sorted({
            pattern
            for relation_type in record.relation_types
            for pattern in RELATION_REVIEW_PATTERNS.get(
                relation_type,
                [],
            )
        })

        if not patterns:
            continue

        result = llm.json_call(
            model=model_config.validation_model,
            system_prompt=(
                "Nominate diagnostic findings without generating final "
                "requirements."
            ),
            user_prompt=DEPENDENCY_NOMINATION_PROMPT.format(
                source=record.source,
                target=record.target,
                path=json.dumps(record.path, ensure_ascii=False),
                relation_types=json.dumps(
                    record.relation_types,
                    ensure_ascii=False,
                ),
                patterns=json.dumps(patterns, ensure_ascii=False),
                source_requirements=requirement_context(
                    document_text,
                    source_frs,
                ),
            ),
            temperature=model_config.validation_temperature,
        )

        for item in coerce_items(result.get("findings")):
            gap_type = coerce_str(item.get("gap_type")).upper()

            if gap_type not in patterns:
                continue

            finding_id = stable_id(
                "finding",
                record.record_id,
                gap_type,
                item.get("affected_element", ""),
            )

            findings.append(DiagnosticFinding(
                finding_id=finding_id,
                view="dependency",
                gap_type=gap_type,
                source_frs=(
                    coerce_str_list(item.get("source_frs"))
                    or list(source_frs)
                ),
                supporting_excerpts=coerce_str_list(
                    item.get("supporting_excerpts")
                ),
                source_element=record.source,
                target_element=record.target,
                affected_element=(
                    coerce_str(item.get("affected_element"))
                    or record.target
                ),
                relation_type="|".join(record.relation_types),
                dependency_path=record.path,
                dps=record.dps,
                nomination_reason=coerce_str(item.get("reason")),
            ))

    return findings


def nominate_isolation_findings(
    llm: LLMClient,
    document_text: str,
    graphs: DependencyGraphs,
    model_config: ModelConfig,
) -> list[DiagnosticFinding]:
    findings = []

    for element in graphs.isolated_semantic_elements():
        data = graphs.semantic_graph.nodes[element]

        result = llm.json_call(
            model=model_config.validation_model,
            system_prompt=(
                "Classify disconnected requirement elements conservatively."
            ),
            user_prompt=ISOLATION_PROMPT.format(
                element=element,
                element_type=data.get("element_type", ""),
                source_requirements=requirement_context(
                    document_text,
                    list(data.get("source_frs", [])),
                ),
                supporting_excerpts=json.dumps(
                    data.get("supporting_excerpts", []),
                    ensure_ascii=False,
                ),
                document_text=document_text,
            ),
            temperature=model_config.validation_temperature,
        )

        label = coerce_str(result.get("label")).upper()

        if label not in {"GENUINE_GAP", "IMPLICIT_DEPENDENCY"}:
            continue

        gap_type = (
            coerce_str(result.get("gap_type")).upper()
            or "MISSING_INTEGRATION"
        )

        findings.append(DiagnosticFinding(
            finding_id=stable_id(
                "finding",
                "isolation",
                element,
                gap_type,
            ),
            view="isolation",
            gap_type=gap_type,
            source_frs=(
                coerce_str_list(result.get("source_frs"))
                or coerce_str_list(data.get("source_frs"))
            ),
            supporting_excerpts=(
                coerce_str_list(result.get("supporting_excerpts"))
                or coerce_str_list(data.get("supporting_excerpts"))
            ),
            affected_element=element,
            nomination_reason=coerce_str(result.get("reason")),
        ))

    return findings


def nominate_operation_findings(
    llm: LLMClient,
    document_text: str,
    elements: list[RequirementElement],
    model_config: ModelConfig,
) -> list[DiagnosticFinding]:
    eligible = [
        element
        for element in elements
        if element.element_type in {"DATA", "FUNCTION"}
    ]

    if not eligible:
        return []

    result = llm.json_call(
        model=model_config.validation_model,
        system_prompt=(
            "Perform conservative operation-coverage diagnosis."
        ),
        user_prompt=OPERATION_PROMPT.format(
            elements=json.dumps(
                [asdict(item) for item in eligible],
                ensure_ascii=False,
            ),
            document_text=document_text,
        ),
        temperature=model_config.validation_temperature,
    )

    findings = []

    for item in coerce_items(result.get("findings")):
        affected = coerce_str(item.get("affected_element"))
        gap_type = coerce_str(item.get("gap_type")).upper()

        if not affected or not gap_type:
            continue

        findings.append(DiagnosticFinding(
            finding_id=stable_id(
                "finding",
                "operation",
                affected,
                gap_type,
            ),
            view="operation",
            gap_type=gap_type,
            source_frs=coerce_str_list(item.get("source_frs")),
            supporting_excerpts=coerce_str_list(
                item.get("supporting_excerpts")
            ),
            affected_element=affected,
            nomination_reason=coerce_str(item.get("reason")),
        ))

    return findings


# ============================================================
# 10. Evidence, coverage, and boundary validation
# ============================================================

VALIDATION_PROMPT = """
Evaluate one nominated requirement-gap finding against the source document.

Finding:
{finding}

Source document:
{document_text}

Return a single JSON object:
{{
  "evidence": {{
    "decision": "YES|NO|UNCERTAIN",
    "reason": "Does the finding cite and follow from documentary elements or relations?"
  }},
  "coverage": {{
    "decision": "YES|NO|UNCERTAIN",
    "reason": "Does an existing requirement already specify an equivalent response?"
  }},
  "boundary": {{
    "decision": "YES|NO|UNCERTAIN",
    "reason": "Does the response belong to the target system or a documented external interaction?"
  }}
}}

Decision semantics:
- evidence YES means documentarily anchored.
- coverage YES means already covered and must not be generated.
- coverage NO means not already covered.
- boundary YES means in scope.
- UNCERTAIN must not be automatically forwarded to generation.

Judge business-semantic equivalence rather than textual similarity.
Do not treat plausibility as evidence.
"""


def normalize_decision(value: Any) -> str:
    value = str(value).upper().strip()
    return value if value in {"YES", "NO", "UNCERTAIN"} else "UNCERTAIN"


def validate_findings(
    llm: LLMClient,
    document_text: str,
    findings: list[DiagnosticFinding],
    model_config: ModelConfig,
    config: PipelineConfig,
) -> list[DiagnosticFinding]:
    validated = []

    for finding in findings:
        result = llm.json_call(
            model=model_config.validation_model,
            system_prompt=(
                "You are a conservative requirements-review validator."
            ),
            user_prompt=VALIDATION_PROMPT.format(
                finding=json.dumps(
                    asdict(finding),
                    ensure_ascii=False,
                ),
                document_text=document_text,
            ),
            temperature=model_config.validation_temperature,
        )

        evidence = coerce_mapping(result.get("evidence"))
        coverage = coerce_mapping(result.get("coverage"))
        boundary = coerce_mapping(result.get("boundary"))

        finding.evidence_decision = (
            normalize_decision(evidence.get("decision"))
            if config.evidence_validation
            else "YES"
        )
        finding.coverage_decision = (
            normalize_decision(coverage.get("decision"))
            if config.coverage_validation
            else "NO"
        )
        finding.boundary_decision = (
            normalize_decision(boundary.get("decision"))
            if config.boundary_validation
            else "YES"
        )

        finding.evidence_reason = coerce_str(evidence.get("reason"))
        finding.coverage_reason = coerce_str(coverage.get("reason"))
        finding.boundary_reason = coerce_str(boundary.get("reason"))

        validated.append(finding)

    return validated


# ============================================================
# 11. Requirement generation and consolidation
# ============================================================

GENERATION_PROMPT = """
Generate one reviewable functional requirement from a validated diagnostic
finding.

Validated finding:
{finding}

Relevant source requirements:
{source_requirements}

Validated element names:
{element_names}

Return a single JSON object:
{{
  "requirement": "The system shall ..."
}}

Rules:
- Express required behavior, not implementation.
- Use only supplied actors, entities, functions, states, and constraints.
- Preserve the documented system boundary.
- Make the requirement testable and concise.
- Do not invent policies, thresholds, responsibilities, or external systems.
- Do not mention DPS, scores, prompts, graphs, or diagnostic machinery.
"""

CONSOLIDATION_PROMPT = """
Consolidate generated requirement candidates.

Candidates:
{candidates}

Source document:
{document_text}

Configuration:
- merge semantic duplicates: {merge}
- refine wording: {refine}
- remove duplicates: {deduplicate}
- filter unsupported or out-of-scope candidates: {filtering}

Return a single JSON object:
{{
  "candidates": [
    {{
      "source_candidate_ids": [],
      "requirement": "The system shall ...",
      "decision": "KEEP|DEMOTE|REMOVE",
      "reason": "brief reason"
    }}
  ]
}}

Group candidates by affected source requirement, principal entity, gap type,
and business-semantic similarity. Several candidates may be merged into one
consolidated requirement; list every contributing identifier.

Decide one of three outcomes per group:

KEEP
  A reviewable functional requirement inside the documented system boundary.
  Merge complementary candidates and refine wording so the result is a
  complete, testable requirement.

DEMOTE
  The content is valid engineering guidance but describes an implementation
  detail, architecture choice, technology selection, or non-functional
  constraint rather than required behavior. Demoted entries are retained
  separately as design constraints and are excluded from the requirements
  document.

REMOVE
  Discard the candidate because it duplicates an existing or previously
  generated requirement; introduces an unsupported principal entity;
  exceeds the documented system boundary; or lacks traceable source
  evidence.

Every input candidate identifier must appear in exactly one returned group,
so that nothing is silently dropped.
"""


def generate_candidates(
    llm: LLMClient,
    document_text: str,
    elements: list[RequirementElement],
    findings: list[DiagnosticFinding],
    model_config: ModelConfig,
) -> list[GeneratedCandidate]:
    candidates = []
    element_names = [element.name for element in elements]

    for finding in findings:
        if not finding.eligible_for_generation():
            continue

        # DPS is deliberately omitted from the generation payload.
        generation_finding = {
            "finding_id": finding.finding_id,
            "view": finding.view,
            "gap_type": finding.gap_type,
            "source_frs": finding.source_frs,
            "supporting_excerpts": finding.supporting_excerpts,
            "source_element": finding.source_element,
            "target_element": finding.target_element,
            "affected_element": finding.affected_element,
            "relation_type": finding.relation_type,
            "dependency_path": finding.dependency_path,
            "nomination_reason": finding.nomination_reason,
            "boundary_decision": finding.boundary_decision,
        }

        result = llm.json_call(
            model=model_config.generation_model,
            system_prompt=(
                "Generate evidence-grounded, reviewable functional "
                "requirements."
            ),
            user_prompt=GENERATION_PROMPT.format(
                finding=json.dumps(
                    generation_finding,
                    ensure_ascii=False,
                ),
                source_requirements=requirement_context(
                    document_text,
                    finding.source_frs,
                ),
                element_names=json.dumps(
                    element_names,
                    ensure_ascii=False,
                ),
            ),
            temperature=model_config.generation_temperature,
        )

        requirement = coerce_str(result.get("requirement"))

        if not requirement:
            continue

        candidates.append(GeneratedCandidate(
            candidate_id=stable_id(
                "candidate",
                finding.finding_id,
                requirement,
            ),
            requirement=requirement,
            diagnostic_view=finding.view,
            gap_type=finding.gap_type,
            source_frs=finding.source_frs,
            supporting_excerpts=finding.supporting_excerpts,
            relation_type=finding.relation_type,
            dependency_path=finding.dependency_path,
            affected_element=finding.affected_element,
            # Added only as audit metadata after generation.
            dps=(
                finding.dps
                if finding.view == "dependency"
                else None
            ),
        ))

    return candidates


def consolidate_candidates(
    llm: LLMClient,
    document_text: str,
    candidates: list[GeneratedCandidate],
    model_config: ModelConfig,
    config: PipelineConfig,
) -> ConsolidationResult:
    """
    Stage-5 refinement: merge, refine, deduplicate and filter candidates.

    Returns a ConsolidationResult rather than a bare list, because the
    refinement pass has three outcomes (KEEP / DEMOTE / REMOVE) and the
    demoted and removed groups are reported downstream. See Section 3.5;
    'describes implementation rather than required behavior' is the DEMOTE
    case and produces design constraints rather than discarding the content.
    """
    if not candidates:
        return ConsolidationResult()

    if not any([
        config.consolidation,
        config.refinement,
        config.deduplication,
        config.final_filtering,
    ]):
        # Ablation: pass every candidate through untouched.
        return ConsolidationResult(kept=list(candidates))

    result = llm.json_call(
        model=model_config.validation_model,
        system_prompt=(
            "Consolidate generated requirements while preserving "
            "traceability."
        ),
        user_prompt=CONSOLIDATION_PROMPT.format(
            candidates=json.dumps(
                [asdict(item) for item in candidates],
                ensure_ascii=False,
            ),
            document_text=document_text,
            merge=str(config.consolidation),
            refine=str(config.refinement),
            deduplicate=str(config.deduplication),
            filtering=str(config.final_filtering),
        ),
        temperature=model_config.validation_temperature,
    )

    by_id = {
        candidate.candidate_id: candidate
        for candidate in candidates
    }

    outcome = ConsolidationResult()
    claimed: set[str] = set()

    for item in coerce_items(result.get("candidates")):
        decision = coerce_str(item.get("decision")).upper()

        source_ids = [
            value
            for value in coerce_str_list(item.get("source_candidate_ids"))
            if value in by_id
        ]

        if not source_ids:
            continue

        source_candidates = [by_id[value] for value in source_ids]
        requirement = coerce_str(item.get("requirement"))
        reason = coerce_str(item.get("reason"))

        if decision == "DEMOTE":
            outcome.demoted.append({
                "requirement": requirement,
                "reason": reason,
                "source_candidate_ids": source_ids,
                "diagnostic_view": "|".join(sorted({
                    candidate.diagnostic_view
                    for candidate in source_candidates
                    if candidate.diagnostic_view
                })),
                "affected_element": "|".join(sorted({
                    candidate.affected_element
                    for candidate in source_candidates
                    if candidate.affected_element
                })),
                "source_frs": flatten_unique([
                    candidate.source_frs for candidate in source_candidates
                ]),
            })
            claimed.update(source_ids)
            continue

        if decision != "KEEP" or not requirement:
            outcome.removed.append({
                "requirement": requirement or "|".join(
                    candidate.requirement for candidate in source_candidates
                ),
                "reason": reason or "rejected by consolidation",
                "source_candidate_ids": source_ids,
            })
            claimed.update(source_ids)
            continue

        outcome.kept.append(GeneratedCandidate(
            candidate_id=stable_id(
                "consolidated",
                *source_ids,
                requirement,
            ),
            requirement=requirement,
            diagnostic_view="|".join(sorted({
                item.diagnostic_view
                for item in source_candidates
            })),
            gap_type="|".join(sorted({
                item.gap_type
                for item in source_candidates
            })),
            source_frs=flatten_unique([
                item.source_frs
                for item in source_candidates
            ]),
            supporting_excerpts=flatten_unique([
                item.supporting_excerpts
                for item in source_candidates
            ]),
            relation_type="|".join(sorted({
                item.relation_type
                for item in source_candidates
                if item.relation_type
            })) or None,
            dependency_path=flatten_unique([
                item.dependency_path
                for item in source_candidates
            ]),
            affected_element="|".join(sorted({
                item.affected_element
                for item in source_candidates
                if item.affected_element
            })) or None,
            dps=max(
                (
                    item.dps
                    for item in source_candidates
                    if item.dps is not None
                ),
                default=None,
            ),
            source_candidate_ids=list(source_ids),
        ))
        claimed.update(source_ids)

    # The prompt asks for full coverage, but never trust the model to be
    # exhaustive: anything it failed to mention is kept rather than silently
    # dropped.
    for candidate in candidates:
        if candidate.candidate_id in claimed:
            continue
        outcome.kept.append(GeneratedCandidate(
            candidate_id=stable_id(
                "consolidated",
                candidate.candidate_id,
                candidate.requirement,
            ),
            requirement=candidate.requirement,
            diagnostic_view=candidate.diagnostic_view,
            gap_type=candidate.gap_type,
            source_frs=list(candidate.source_frs),
            supporting_excerpts=list(candidate.supporting_excerpts),
            relation_type=candidate.relation_type,
            dependency_path=list(candidate.dependency_path),
            affected_element=candidate.affected_element,
            dps=candidate.dps,
            source_candidate_ids=[candidate.candidate_id],
        ))

    return outcome


# ============================================================
# 11.1 Refined requirements document (terminal Stage 5 output)
# ============================================================
#
# Section 3.5 and Figure 1 both terminate the pipeline at a *refined
# requirements document*: the original specification with the accepted
# supplementary requirements appended under numbers that continue the
# document's own numbering. The functions below produce that artifact.
#
# Numbering is deterministic. The paper's Section 4.1.2 lists a temperature
# for "renumbering", which would only apply if numbering were a model call;
# continuing an existing integer sequence needs no model and is reproducible,
# so no LLM call is issued here. If a model-assigned numbering scheme is
# preferred, only assign_requirement_numbers() needs to change.

FR_FORMAT_PATTERNS = {
    "word": re.compile(r"(?m)^Function Requirement \d+"),
    "mddot": re.compile(r"(?m)^#{1,3}\s+\d+\.\d+\s"),
    "dot": re.compile(r"(?m)^\d+\.\d+\s"),
    "dash": re.compile(r"(?m)^FR-\d+", re.IGNORECASE),
    "letter": re.compile(r"(?m)^F\d+"),
    "space": re.compile(r"(?m)^FR\s+\d+\s*:"),
}


def classify_fr_id(requirement_id: str) -> str | None:
    """Name the numbering dialect a single requirement identifier uses."""
    if re.fullmatch(r"\d+\.\d+\.\d+", requirement_id):
        return "dot3"
    if re.fullmatch(r"\d+\.\d+", requirement_id):
        return "dot"
    if re.fullmatch(r"Function Requirement \d+", requirement_id):
        return "word"
    if re.fullmatch(r"FR\s+\d+", requirement_id):
        return "space"
    if re.fullmatch(r"FR-[A-Z]{1,6}-\d+", requirement_id, re.IGNORECASE):
        return "compound"
    if re.fullmatch(r"FR-\d+", requirement_id, re.IGNORECASE):
        return "dash"
    if re.fullmatch(r"F\d+", requirement_id):
        return "letter"
    return None


def fr_heading_marker(document_text: str) -> str:
    """
    Leading markdown hashes used by the requirement headings, else ''.

    A document whose requirements are written '## 1.1 Login' must get its
    supplementary requirements written the same way, or the appended block
    stops matching the document it was appended to.
    """
    for _, block in split_into_frs(document_text):
        match = re.match(r"^(#{1,6})\s*\S", block)
        if match:
            return match.group(1)
    return ""


def detect_fr_format(document_text: str) -> str:
    """
    Detect the requirement-numbering dialect of a document.

    The dialect is read from the identifiers that split_into_frs actually
    produced, not from a scan of the raw text. That keeps segmentation and
    numbering in agreement: a document whose sections are numbered "3.1" but
    whose requirements are numbered "FR-001" is segmented at FR-001, so its
    supplementary requirements must continue as FR-015 rather than as 3.7.

    The regex scan below remains as the fallback for a document that carries
    no recognisable identifier at all.
    """
    votes: dict[str, int] = {}

    for requirement_id, _ in split_into_frs(document_text):
        if requirement_id.startswith("BLOCK-"):
            continue
        name = classify_fr_id(requirement_id)
        if name:
            votes[name] = votes.get(name, 0) + 1

    if votes:
        winner = max(votes.items(), key=lambda pair: pair[1])[0]
        # '1.1' and '## 1.1' produce the same identifier; only the heading
        # marker distinguishes them, and it must survive into the output.
        if winner == "dot" and fr_heading_marker(document_text):
            return "mddot"
        return winner

    for name in ("word", "mddot", "dot", "dash", "letter"):
        if FR_FORMAT_PATTERNS[name].search(document_text):
            return name
    return "space"


def original_fr_major(document_text: str) -> str:
    """
    Major section number of the first requirement, e.g. '1' in '1.1'.

    For three-level numbering this is the full "3.1" prefix, so that
    continuing the sequence stays inside the deepest section rather than
    restarting at the top level.
    """
    for requirement_id, _ in split_into_frs(document_text):
        match = re.fullmatch(r"(\d+\.\d+)\.\d+", requirement_id)
        if match:
            return match.group(1)
        match = re.fullmatch(r"(\d+)\.\d+", requirement_id)
        if match:
            return match.group(1)
    return ""


def last_fr_number(document_text: str, major: str = "") -> int:
    """
    Highest existing requirement number in the document's own dialect.

    Only the trailing number is returned, so callers can continue the
    sequence within the same section. Pass ``major`` to count within that
    section: without it, a document numbered 2.1-2.8 followed by a
    non-requirement section numbered 3.1-3.4 would report 4, and the next
    generated requirement would be numbered 2.5 -- colliding with an
    existing 2.5.
    """
    numbers = []

    for requirement_id, _ in split_into_frs(document_text):
        match = re.fullmatch(r"(\d+\.\d+)\.(\d+)", requirement_id)
        if match:
            if not major or match.group(1) == major:
                numbers.append(int(match.group(2)))
            continue

        match = re.fullmatch(r"(\d+)\.(\d+)", requirement_id)
        if match:
            if not major or match.group(1) == major:
                numbers.append(int(match.group(2)))
            continue

        match = re.fullmatch(
            r"(?:Function Requirement\s+|FR-[A-Z]{1,6}-|FR-0*|FR\s+0*|F0*)"
            r"(\d+)",
            requirement_id,
            re.IGNORECASE,
        )
        if match:
            numbers.append(int(match.group(1)))

    return max(numbers, default=0)


def _format_number(
    fmt: str,
    major: str,
    minor: int,
    prefix: str = "",
) -> str:
    """Render one requirement number in the document's own dialect."""
    if fmt == "dot3":
        return f"{major}.{minor}" if major else str(minor)
    if fmt in ("dot", "mddot"):
        return f"{major}.{minor}" if major else str(minor)
    if fmt == "word":
        return f"Function Requirement {minor}"
    if fmt == "dash":
        width = max(3, len(str(minor)))
        return f"FR-{minor:0{width}d}"
    if fmt == "compound":
        stem = prefix or "FR-NEW"
        return f"{stem}-{minor:03d}"
    if fmt == "letter":
        return f"F{minor}"
    return f"FR {minor}:"


def numbering_context(document_text: str) -> dict[str, str]:
    """
    Everything needed to continue a document's numbering.

    Segmentation, the major section, the last used number and the compound
    identifier prefix are all derived from the same split, so they cannot
    disagree with each other.
    """
    fmt = detect_fr_format(document_text)
    major = original_fr_major(document_text)
    last = last_fr_number(document_text, major)

    prefix = ""
    if fmt == "compound":
        for requirement_id, _ in split_into_frs(document_text):
            match = re.fullmatch(
                r"(FR-[A-Z]{1,6})-\d+", requirement_id, re.IGNORECASE
            )
            if match:
                prefix = match.group(1).upper()

    return {
        "format": fmt,
        "major": major,
        "last": str(last),
        "prefix": prefix,
    }


def assign_requirement_numbers(
    original_text: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Continue the document's own numbering for the accepted candidates.

    Takes candidate dicts (as produced by consolidate_candidates and
    serialized by DIOReqPipeline.run) and returns copies with an added
    'fr_number' field, preserving input order.
    """
    context = numbering_context(original_text)
    fmt = context["format"]
    major = context["major"]
    prefix = context["prefix"]
    next_minor = int(context["last"]) + 1

    numbered: list[dict[str, Any]] = []

    for candidate in candidates:
        record = dict(candidate)
        record["fr_number"] = _format_number(fmt, major, next_minor, prefix)
        next_minor += 1
        numbered.append(record)

    return numbered


def build_refined_document(
    original_text: str,
    candidates: list[dict[str, Any]],
) -> str:
    """
    Original specification followed by the numbered supplementary
    requirements, i.e. the 'refined requirements document' of Figure 1.
    """
    numbered = assign_requirement_numbers(original_text, candidates)
    marker = fr_heading_marker(original_text)

    blocks = [original_text.strip(), ""]

    for record in numbered:
        heading = record["fr_number"]
        if marker:
            heading = f"{marker} {heading}"
        title = str(record.get("affected_element") or "").strip()
        blocks.append(f"{heading} {title}".rstrip())
        blocks.append(
            f"Requirement: {str(record.get('requirement', '')).strip()}"
        )

        if record.get("source_frs"):
            blocks.append(
                "Source requirements: "
                + ", ".join(str(x) for x in record["source_frs"])
            )
        if record.get("supporting_excerpts"):
            blocks.append(
                "Evidence: "
                + " | ".join(
                    str(x).strip()
                    for x in record["supporting_excerpts"]
                    if str(x).strip()
                )
            )
        if record.get("diagnostic_view"):
            blocks.append(f"Diagnostic view: {record['diagnostic_view']}")
        if record.get("gap_type"):
            blocks.append(f"Gap type: {record['gap_type']}")
        if record.get("dps") is not None:
            blocks.append(f"DPS: {record['dps']:.4f}")

        blocks.append("")

    return "\n".join(blocks).strip() + "\n"


def save_text(path: str | Path, text: str) -> None:
    """Write UTF-8 text, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_design_constraints_document(
    demoted: list[dict[str, Any]],
    removed: list[dict[str, Any]],
) -> str:
    """
    Design-constraint companion to the refined requirements document.

    Holds the candidates that the refinement pass judged to be valid
    engineering guidance but not required behavior, plus the rejected
    entries with their reasons. Neither group belongs in the requirements
    document, but discarding them entirely would lose reviewable content.
    """
    blocks = [
        "=" * 72,
        "DESIGN CONSTRAINTS AND REJECTED CANDIDATES",
        "=" * 72,
        "",
        "Produced by the Stage-5 refinement pass (Section 3.5).",
        "These entries are NOT part of the refined requirements document.",
        "",
        f"Demoted to design constraints : {len(demoted)}",
        f"Rejected during refinement    : {len(removed)}",
        "",
    ]

    blocks += ["-" * 72, f"1. DESIGN CONSTRAINTS ({len(demoted)})", "-" * 72, ""]
    if not demoted:
        blocks += ["(none)", ""]
    for index, item in enumerate(demoted, start=1):
        blocks.append(f"DC-{index:03d}")
        blocks.append(f"Statement: {item.get('requirement', '').strip()}")
        if item.get("reason"):
            blocks.append(f"Demoted because: {item['reason']}")
        if item.get("diagnostic_view"):
            blocks.append(f"Diagnostic view: {item['diagnostic_view']}")
        if item.get("affected_element"):
            blocks.append(f"Affected element: {item['affected_element']}")
        if item.get("source_frs"):
            blocks.append(
                "Source requirements: "
                + ", ".join(str(x) for x in item["source_frs"])
            )
        blocks += ["", "-" * 60, ""]

    blocks += ["-" * 72, f"2. REJECTED CANDIDATES ({len(removed)})", "-" * 72, ""]
    if not removed:
        blocks += ["(none)", ""]
    for index, item in enumerate(removed, start=1):
        blocks.append(f"REMOVED-{index:03d}")
        blocks.append(f"Statement: {item.get('requirement', '').strip()}")
        if item.get("reason"):
            blocks.append(f"Rejected because: {item['reason']}")
        blocks += ["", "-" * 60, ""]

    return "\n".join(blocks).rstrip() + "\n"


def build_refinement_report(
    original_text: str,
    kept: list[dict[str, Any]],
    demoted: list[dict[str, Any]],
    removed: list[dict[str, Any]],
) -> str:
    """Markdown report summarising what the refinement pass did and why."""
    context = numbering_context(original_text)
    fmt = context["format"]
    major = context["major"]
    original_count = int(context["last"])

    lines = [
        "# Stage 5 - Requirement Refinement Report",
        "",
        "## Summary",
        "",
        "| Outcome | Count |",
        "|---|---:|",
        f"| Original requirements in the source document | {original_count} |",
        f"| Refined (kept) | {len(kept)} |",
        f"| Demoted to design constraints | {len(demoted)} |",
        f"| Rejected | {len(removed)} |",
        f"| **Total requirements in the refined document** | "
        f"**{original_count + len(kept)}** |",
        "",
        f"Numbering dialect: `{fmt}`"
        + (f", major section `{major}`" if major else ""),
        "",
        "## Kept requirements",
        "",
    ]

    if not kept:
        lines += ["(none)", ""]
    for record in kept:
        lines.append(
            f"- `{record.get('fr_number', '?')}` "
            f"{record.get('requirement', '').strip()}"
        )
        detail = []
        if record.get("diagnostic_view"):
            detail.append(f"view `{record['diagnostic_view']}`")
        if record.get("gap_type"):
            detail.append(f"gap `{record['gap_type']}`")
        if record.get("source_frs"):
            detail.append(
                "sources " + ", ".join(str(x) for x in record["source_frs"])
            )
        if record.get("source_candidate_ids"):
            detail.append(
                f"{len(record['source_candidate_ids'])} merged candidate(s)"
            )
        if detail:
            lines.append(f"  - {', '.join(detail)}")
    lines.append("")

    lines += ["## Demoted to design constraints", ""]
    if not demoted:
        lines += ["(none)", ""]
    for item in demoted:
        lines.append(f"- {item.get('requirement', '').strip()}")
        if item.get("reason"):
            lines.append(f"  - reason: {item['reason']}")
        if item.get("source_candidate_ids"):
            lines.append(
                "  - merged candidates: "
                + ", ".join(str(x) for x in item["source_candidate_ids"])
            )
    lines.append("")

    lines += ["## Rejected", ""]
    if not removed:
        lines += ["(none)", ""]
    for item in removed:
        lines.append(f"- {item.get('requirement', '').strip()}")
        if item.get("reason"):
            lines.append(f"  - reason: {item['reason']}")
        if item.get("source_candidate_ids"):
            lines.append(
                "  - merged candidates: "
                + ", ".join(str(x) for x in item["source_candidate_ids"])
            )
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ============================================================
# 12. Complete DIOReq pipeline
# ============================================================

class DIOReqPipeline:
    def __init__(
        self,
        llm: LLMClient,
        model_config: ModelConfig | None = None,
        pipeline_config: PipelineConfig | None = None,
    ):
        self.llm = llm
        # Resolved on construction so late environment changes are honoured.
        self.model_config = (
            model_config
            if model_config is not None
            else default_model_config()
        )
        self.config = (
            pipeline_config
            if pipeline_config is not None
            else PipelineConfig()
        )

    def prepare_document(
        self,
        document_text: str,
        progress: Any = None,
    ) -> tuple[
        list[RequirementElement],
        list[DependencyRelation],
        DependencyGraphs,
        list[RankingRecord],
    ]:
        elements = extract_elements(
            self.llm,
            document_text,
            self.model_config,
            progress=progress,
        )

        relations = extract_dependency_relations(
            self.llm,
            document_text,
            elements,
            self.model_config,
            use_cross_requirement_pass=(
                self.config.cross_requirement_extraction
            ),
            progress=progress,
        )

        graphs = DependencyGraphs(
            max_parents=self.config.max_parents
        )
        graphs.build(elements, relations)

        records = graphs.ranking_records()

        return elements, relations, graphs, records

    def run(
        self,
        document_text: str,
        ranking_strategy: str = "dps",
        random_seed: int = 0,
        progress: Any = None,
        prepared: Any = None,
    ) -> dict:
        """
        Execute all five stages for one document.

        ``progress`` is an optional callable receiving a short status string.
        A full experiment is 126 document-runs of roughly a hundred model
        calls each, so a caller that prints nothing appears to hang for
        hours; the callback is how the runner reports liveness.

        ``prepared`` is an optional ``(elements, relations, graphs, records)``
        tuple from ``prepare_document``. Supplying it skips stages 1-2 and
        reuses an existing graph, which is what an ablation study needs: the
        endpoint is not deterministic even at temperature 0, so re-extracting
        the graph for every ablation variant would mix the ablation effect
        with extraction noise. Only reuse a graph when the preparation
        settings (``cross_requirement_extraction``, ``max_parents``) that
        produced it match this pipeline's configuration.
        """
        def report(message: str) -> None:
            if progress is not None:
                progress(message)

        usage_before = Usage(
            prompt_tokens=self.llm.usage.prompt_tokens,
            completion_tokens=self.llm.usage.completion_tokens,
            api_calls=self.llm.usage.api_calls,
        )

        if prepared is None:
            elements, relations, graphs, records = (
                self.prepare_document(document_text, progress=progress)
            )
            report(
                f"stage 1-2: {len(elements)} element(s), "
                f"{len(relations)} relation(s), "
                f"{len(records)} ranking record(s)"
            )
        else:
            elements, relations, graphs, records = prepared
            report(
                f"stage 1-2: reused prepared graph "
                f"({len(elements)} element(s), "
                f"{len(records)} ranking record(s))"
            )

        findings: list[DiagnosticFinding] = []

        if self.config.dependency_view:
            strategy = (
                ranking_strategy
                if self.config.use_dps_ranking
                else "unranked"
            )

            ranked = rank_records(
                records,
                strategy=strategy,
                seed=random_seed,
            )

            selected = ranked[
                :min(self.config.dependency_budget, len(ranked))
            ]

            findings.extend(nominate_dependency_findings(
                self.llm,
                document_text,
                graphs,
                selected,
                self.model_config,
            ))
            report(
                f"stage 3 dependency view: "
                f"{len(selected)} record(s) reviewed"
            )

        if self.config.isolation_view:
            findings.extend(nominate_isolation_findings(
                self.llm,
                document_text,
                graphs,
                self.model_config,
            ))

        if self.config.operation_view:
            findings.extend(nominate_operation_findings(
                self.llm,
                document_text,
                elements,
                self.model_config,
            ))

        report(f"stage 3: {len(findings)} finding(s) nominated")

        validated_findings = validate_findings(
            self.llm,
            document_text,
            findings,
            self.model_config,
            self.config,
        )

        eligible = sum(
            1 for finding in validated_findings
            if finding.eligible_for_generation()
        )
        report(
            f"stage 4: {len(validated_findings)} validated, "
            f"{eligible} eligible for generation"
        )

        raw_candidates = generate_candidates(
            self.llm,
            document_text,
            elements,
            validated_findings,
            self.model_config,
        )

        consolidation = consolidate_candidates(
            self.llm,
            document_text,
            raw_candidates,
            self.model_config,
            self.config,
        )

        report(
            f"stage 5: {len(raw_candidates)} candidate(s) -> "
            f"{len(consolidation.kept)} kept, "
            f"{len(consolidation.demoted)} demoted, "
            f"{len(consolidation.removed)} removed"
        )

        usage_after = self.llm.usage

        run_usage = Usage(
            prompt_tokens=(
                usage_after.prompt_tokens
                - usage_before.prompt_tokens
            ),
            completion_tokens=(
                usage_after.completion_tokens
                - usage_before.completion_tokens
            ),
            api_calls=(
                usage_after.api_calls
                - usage_before.api_calls
            ),
        )

        return {
            "elements": [asdict(item) for item in elements],
            "relations": [asdict(item) for item in relations],
            "computational_exclusions": graphs.exclusions,
            "ranking_records": [
                asdict(item) for item in records
            ],
            "findings": [
                asdict(item) for item in validated_findings
            ],
            "raw_candidates": [
                asdict(item) for item in raw_candidates
            ],
            "candidates": [
                asdict(item) for item in consolidation.kept
            ],
            "design_constraints": consolidation.demoted,
            "removed_candidates": consolidation.removed,
            "usage": asdict(run_usage),
        }


# ============================================================
# 13. DIOReq experiment runner
# ============================================================

def load_documents(
    dataset_path: str | Path,
) -> list[DocumentRecord]:
    """
    Load requirements documents from a JSON dataset.

    Each dataset item must contain:
    - system_id
    - document_id
    - text
    """
    raw = load_json(dataset_path)
    documents: list[DocumentRecord] = []

    if not isinstance(raw, list):
        raise ValueError(
            "The dataset must be a JSON array of document objects"
        )

    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(
                f"Dataset item {index} must be a JSON object"
            )

        missing_fields = [
            field_name
            for field_name in (
                "system_id",
                "document_id",
                "text",
            )
            if field_name not in item
        ]

        if missing_fields:
            raise ValueError(
                f"Dataset item {index} is missing required fields: "
                f"{', '.join(missing_fields)}"
            )

        documents.append(DocumentRecord(
            system_id=str(item["system_id"]),
            document_id=str(item["document_id"]),
            text=str(item["text"]),
        ))

    return documents


def run_dioreq(
    dataset_path: str,
    output_path: str,
    repeated_runs: int,
    dependency_budget: int,
    refined_dir: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    shard: tuple[int, int] | None = None,
) -> None:
    """
    Run DIOReq over every document for the requested number of
    independent runs.

    Besides the raw JSON record, each document-run can be written out as a
    refined requirements document (Section 3.5 / Figure 1). Pass
    ``refined_dir`` to choose where those files go, or ``None`` to skip them.

    ``model`` overrides all three stage models; ``base_url`` and ``api_key``
    override the OpenAI endpoint. Each falls back to the corresponding
    environment variable, so a caller can also leave them unset.

    ``shard`` restricts this process to a slice of the documents; see
    ``shard_documents``.
    """
    if repeated_runs < 1:
        raise ValueError(
            "repeated_runs must be at least 1"
        )

    if dependency_budget < 1:
        raise ValueError(
            "dependency_budget must be at least 1"
        )

    # Load the dataset before constructing the client so that a bad
    # dataset path or schema is reported before any API configuration error.
    documents = shard_documents(load_documents(dataset_path), shard)

    if not documents:
        print(
            "No documents assigned to this shard; nothing to do."
        )
        return

    llm = LLMClient(api_key=api_key, base_url=base_url)
    results: list[dict[str, Any]] = []

    model_config = default_model_config()

    if model:
        model_config = ModelConfig(
            extraction_model=model,
            validation_model=model,
            generation_model=model,
            extraction_temperature=model_config.extraction_temperature,
            validation_temperature=model_config.validation_temperature,
            generation_temperature=model_config.generation_temperature,
        )

    if refined_dir:
        Path(refined_dir).mkdir(parents=True, exist_ok=True)

    total_runs = len(documents) * repeated_runs
    completed_runs = 0
    failures = 0

    for document in documents:
        for run_id in range(1, repeated_runs + 1):
            completed_runs += 1

            print(
                f"[DIOReq] "
                f"run={completed_runs}/{total_runs}, "
                f"system={document.system_id}, "
                f"document={document.document_id}, "
                f"repetition={run_id}"
            )

            pipeline = DIOReqPipeline(
                llm=llm,
                model_config=model_config,
                pipeline_config=PipelineConfig(
                    dependency_budget=dependency_budget,
                ),
            )

            # One failing document-run must not discard the runs that
            # already succeeded: the batch is 126 runs, and a single
            # endpoint hiccup at run 120 would otherwise lose everything.
            try:
                output = pipeline.run(
                    document_text=document.text,
                    ranking_strategy="dps",
                    random_seed=run_id,
                    progress=lambda message: print(f"          {message}"),
                )
            except Exception as error:
                failures += 1
                print(
                    f"          FAILED ({type(error).__name__}): {error}"
                )
                results.append({
                    "method": "DIOReq",
                    "system_id": document.system_id,
                    "document_id": document.document_id,
                    "run_id": run_id,
                    "ranking_strategy": "dps",
                    "dependency_budget": dependency_budget,
                    "error": f"{type(error).__name__}: {error}",
                })
                save_json(output_path, results)
                continue

            results.append({
                "method": "DIOReq",
                "system_id": document.system_id,
                "document_id": document.document_id,
                "run_id": run_id,
                "ranking_strategy": "dps",
                "dependency_budget": dependency_budget,
                **output,
            })

            # Save after every run so completed results survive an
            # interruption or API failure.
            save_json(output_path, results)

            if refined_dir:
                # Deterministic numbering, so calling it here and inside
                # build_refined_document yields identical numbers.
                numbered = assign_requirement_numbers(
                    document.text, output["candidates"]
                )
                stem = (
                    f"{document.system_id}_{document.document_id}"
                    f"_run{run_id}"
                )
                directory = Path(refined_dir)

                refined_path = directory / f"{stem}_refined.txt"
                save_text(
                    refined_path,
                    build_refined_document(
                        document.text, output["candidates"]
                    ),
                )

                constraints_path = directory / f"{stem}_design_constraints.txt"
                save_text(
                    constraints_path,
                    build_design_constraints_document(
                        output.get("design_constraints", []),
                        output.get("removed_candidates", []),
                    ),
                )

                report_path = directory / f"{stem}_refinement_report.md"
                save_text(
                    report_path,
                    build_refinement_report(
                        document.text,
                        numbered,
                        output.get("design_constraints", []),
                        output.get("removed_candidates", []),
                    ),
                )

                print(
                    f"          refined {len(numbered)} requirement(s), "
                    f"demoted {len(output.get('design_constraints', []))}, "
                    f"removed {len(output.get('removed_candidates', []))} "
                    f"-> {directory}"
                )

    print(
        f"DIOReq completed: {total_runs - failures}/{total_runs} runs "
        f"succeeded; results saved to {output_path}"
    )

    if failures:
        print(
            f"{failures} run(s) failed and are recorded with an "
            f"'error' field in {output_path}"
        )


def parse_shard(value: str | None) -> tuple[int, int] | None:
    """Parse ``--shard I/N`` into ``(index, total)``; ``None`` when unset."""
    if value is None:
        return None

    match = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", str(value))

    if not match:
        raise ValueError(
            f"--shard expects I/N (e.g. 0/4), got {value!r}"
        )

    index, total = int(match.group(1)), int(match.group(2))

    if total < 1:
        raise ValueError("--shard total must be at least 1")

    if not 0 <= index < total:
        raise ValueError(
            f"--shard index must be in [0, {total - 1}], got {index}"
        )

    return index, total


def shard_documents(
    documents: list[DocumentRecord],
    shard: tuple[int, int] | None,
) -> list[DocumentRecord]:
    """
    Keep only this process's share of the documents.

    The pipeline is sequential and one document-run costs 150-200 model
    calls, so a full protocol is days of wall-clock time in a single
    process. Every runner is embarrassingly parallel across documents, and
    process-level sharding avoids putting shared mutable state (the usage
    counter, the output list) behind a lock.

    Launch N processes with distinct ``--output`` paths and ``--shard i/N``,
    then concatenate the resulting JSON arrays.
    """
    if shard is None:
        return documents

    index, total = shard

    return [
        document
        for position, document in enumerate(documents)
        if position % total == index
    ]


def add_shard_argument(parser: argparse.ArgumentParser) -> None:
    """Register the shared ``--shard I/N`` option."""
    parser.add_argument(
        "--shard",
        default=None,
        metavar="I/N",
        help=(
            "Process only the documents whose position modulo N equals I. "
            "Run N processes with distinct --output paths, then concatenate "
            "the JSON arrays. Example: --shard 0/4."
        ),
    )


def add_client_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Register the endpoint and model options shared by every runner.

    Kept in one place so dioreq.py, rq2.py and rq3.py cannot drift into
    accepting different sets of options for the same client.
    """
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model used for every stage. Overrides "
            "DIOREQ_EXTRACTION_MODEL / DIOREQ_VALIDATION_MODEL / "
            "DIOREQ_GENERATION_MODEL."
        ),
    )

    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "OpenAI-compatible endpoint, e.g. https://host/v1. Falls back "
            "to OPENAI_BASE_URL."
        ),
    )

    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "API key. Falls back to OPENAI_API_KEY; prefer the environment "
            "variable so the key does not appear in shell history."
        ),
    )

    parser.add_argument(
        "--temperature",
        default=None,
        help=(
            "Sampling temperature applied to every stage. Use 'none' for "
            "models that accept only their built-in temperature. Falls back "
            "to DIOREQ_EXTRACTION_TEMPERATURE and friends."
        ),
    )


def apply_cli_environment(args: Any) -> None:
    """
    Copy the --temperature option into the environment before the model
    configuration is resolved, since default_model_config reads it there.
    """
    temperature = getattr(args, "temperature", None)

    if temperature is None:
        return

    for stage in ("EXTRACTION", "VALIDATION", "GENERATION"):
        os.environ[f"DIOREQ_{stage}_TEMPERATURE"] = temperature


def resolve_client(
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> tuple[LLMClient, ModelConfig]:
    """Build the API client and the stage model configuration."""
    config = default_model_config()

    if model:
        config = ModelConfig(
            extraction_model=model,
            validation_model=model,
            generation_model=model,
            extraction_temperature=config.extraction_temperature,
            validation_temperature=config.validation_temperature,
            generation_temperature=config.generation_temperature,
        )

    return LLMClient(api_key=api_key, base_url=base_url), config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the DIOReq requirements-completion pipeline "
            "over a JSON dataset."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "Path to a JSON dataset containing system_id, "
            "document_id, and text."
        ),
    )

    parser.add_argument(
        "--output",
        default="results/dioreq_raw.json",
        help="Path used to save raw DIOReq results.",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of independent runs per document.",
    )

    parser.add_argument(
        "--dependency-budget",
        type=int,
        default=20,
        help=(
            "Maximum number of ranked dependency records "
            "reviewed per document."
        ),
    )

    parser.add_argument(
        "--refined-dir",
        default="results/refined",
        help=(
            "Directory for the refined requirements documents produced by "
            "Stage 5 (Section 3.5 / Figure 1). One UTF-8 text file is "
            "written per document-run."
        ),
    )

    parser.add_argument(
        "--no-refined-documents",
        action="store_true",
        help="Skip writing the Stage 5 refined requirements documents.",
    )

    add_client_arguments(parser)
    add_shard_argument(parser)

    args = parser.parse_args()

    apply_cli_environment(args)

    run_dioreq(
        dataset_path=args.dataset,
        output_path=args.output,
        repeated_runs=args.runs,
        dependency_budget=args.dependency_budget,
        refined_dir=(
            None if args.no_refined_documents else args.refined_dir
        ),
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        shard=parse_shard(args.shard),
    )


if __name__ == "__main__":
    main()
