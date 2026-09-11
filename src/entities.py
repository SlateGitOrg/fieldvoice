"""Typed entity extraction for field-maintenance transcripts.

The entities defined here are exactly the fields that get written into the
structured maintenance record: part IDs, fault codes, quantities (magnitude
AND unit), dates, and named locations.  Everything else in a transcript is
narrative.  The evaluation in ``metrics.py`` scores these and only these.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

PART_ID = "PART_ID"
FAULT_CODE = "FAULT_CODE"
QUANTITY = "QUANTITY"
DATE = "DATE"
LOCATION = "LOCATION"

ENTITY_TYPES = (PART_ID, FAULT_CODE, QUANTITY, DATE, LOCATION)

_PART_RE = re.compile(r"^[A-Z]{2}-\d{4}$")
_FAULT_RE = re.compile(r"^E-\d{3}$")
_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_NUMERIC_RE = re.compile(r"^\d+(?:\.\d+)?$")

# Spoken numbers have to normalise to magnitudes, otherwise "fifteen" and "15"
# would count as a substitution purely because of orthography while "fifteen"
# and "fifty" -- an actual dosing error -- would count the same.  Composition
# ("forty four") is deliberately NOT handled: a spelled-out identifier is an
# identifier failure, not a number, and collapsing it would hide the error.
NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100,
}

# Unit aliases fold orthography (millilitres/milliliters/ml) but NEVER fold
# across physical dimensions.  mg and ml sound alike and are different drugs'
# worth of difference; keeping them distinct is the entire point of scoring a
# quantity on magnitude and unit jointly.
UNIT_ALIASES = {
    "mg": "mg", "milligram": "mg", "milligrams": "mg",
    "ml": "ml", "millilitre": "ml", "millilitres": "ml",
    "milliliter": "ml", "milliliters": "ml",
    "l": "l", "litre": "l", "litres": "l", "liter": "l", "liters": "l",
    "nm": "nm", "newton-metres": "nm", "newton-meters": "nm",
    "mm": "mm", "millimetre": "mm", "millimetres": "mm",
    "millimeter": "mm", "millimeters": "mm",
    "cm": "cm", "centimetre": "cm", "centimetres": "cm",
    "m": "m", "metre": "m", "metres": "m", "meter": "m", "meters": "m",
    "kg": "kg", "kilogram": "kg", "kilograms": "kg",
    "bar": "bar", "volts": "v", "v": "v", "amps": "a", "a-amps": "a",
    "rpm": "rpm", "degrees": "deg", "deg": "deg",
    "hours": "h", "hrs": "h",
}

# Named locations are a closed gazetteer, the way a real deployment has a
# closed site list.  Multi-word names are matched longest-first so "north
# ridge" never degrades into the single token "north".
LOCATION_GAZETTEER = (
    "north ridge", "south ridge", "tower base", "nacelle deck",
    "hub access", "kilbraur", "ardrossan", "braes", "stronelairg",
    "west pier",
)
_MAX_LOCATION_TOKENS = max(len(name.split()) for name in LOCATION_GAZETTEER)


@dataclass(frozen=True)
class Entity:
    """One typed field, with the token index where it starts.

    ``value`` is the canonical comparison key.  For a quantity it encodes the
    magnitude and the unit together, so equality is only true when both agree.
    """

    etype: str
    value: str
    pos: int
    magnitude: Optional[float] = None
    unit: Optional[str] = None

    def matches(self, other: "Entity") -> bool:
        return self.etype == other.etype and self.value == other.value


def tokenize(text: str) -> List[str]:
    """Lowercase whitespace tokens with edge punctuation stripped.

    Hyphens and dots inside a token survive because they carry meaning in
    ``GB-4471``, ``E-212`` and ``2024-03-12``.
    """
    raw = text.split()
    out = []
    for tok in raw:
        tok = tok.strip(",.;:!?()\"'")
        if tok:
            out.append(tok.lower())
    return out


def parse_number(token: str) -> Optional[float]:
    if _NUMERIC_RE.match(token):
        return float(token)
    if token in NUMBER_WORDS:
        return float(NUMBER_WORDS[token])
    return None


def quantity_value(magnitude: float, unit: str) -> str:
    return "%g %s" % (magnitude, unit)


def extract_entities(text_or_tokens) -> List[Entity]:
    """Pull typed fields out of a transcript.

    Runs identically on a reference and on an ASR hypothesis -- that symmetry
    is what makes the entity error rate an honest comparison rather than a
    scoring of two different pipelines.
    """
    tokens = (
        list(text_or_tokens)
        if isinstance(text_or_tokens, (list, tuple))
        else tokenize(text_or_tokens)
    )
    found: List[Entity] = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        upper = tok.upper()

        span = _match_location(tokens, i)
        if span is not None:
            name, width = span
            found.append(Entity(LOCATION, name, i))
            i += width
            continue

        if _PART_RE.match(upper):
            found.append(Entity(PART_ID, upper, i))
            i += 1
            continue

        if _FAULT_RE.match(upper):
            found.append(Entity(FAULT_CODE, upper, i))
            i += 1
            continue

        date = _normalize_date(tok)
        if date is not None:
            found.append(Entity(DATE, date, i))
            i += 1
            continue

        magnitude = parse_number(tok)
        if magnitude is not None and i + 1 < n:
            unit = UNIT_ALIASES.get(tokens[i + 1])
            if unit is not None:
                found.append(
                    Entity(
                        QUANTITY,
                        quantity_value(magnitude, unit),
                        i,
                        magnitude=magnitude,
                        unit=unit,
                    )
                )
                i += 2
                continue

        i += 1
    return found


def _match_location(tokens: Sequence[str], i: int) -> Optional[Tuple[str, int]]:
    for width in range(min(_MAX_LOCATION_TOKENS, len(tokens) - i), 0, -1):
        candidate = " ".join(tokens[i:i + width])
        if candidate in LOCATION_GAZETTEER:
            return candidate, width
    return None


def _normalize_date(token: str) -> Optional[str]:
    m = _DATE_RE.match(token)
    if not m:
        return None
    year, month, day = (int(g) for g in m.groups())
    # A transcription can produce 2024-03-32; it is still a date-shaped field
    # and still wrong, so it is kept as an entity rather than dropped (dropping
    # it would turn a substitution into a deletion and flatter the system).
    return "%04d-%02d-%02d" % (year, month, day)


def entities_by_type(entities: Sequence[Entity], etype: str) -> List[Entity]:
    return [e for e in entities if e.etype == etype]
