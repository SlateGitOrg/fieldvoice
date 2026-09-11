"""Synthetic field reports with PLANTED ground-truth entities.

Each utterance is assembled from a structured record, so the gold entity list
is known by construction rather than by running the extractor on the
reference.  That distinction matters: if gold came from the extractor, a bug
in the extractor would be invisible, because it would corrupt both sides of
the comparison identically.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .entities import (
    DATE,
    FAULT_CODE,
    LOCATION,
    LOCATION_GAZETTEER,
    PART_ID,
    QUANTITY,
    Entity,
    quantity_value,
    tokenize,
)

PART_PREFIXES = ("GB", "YD", "PT", "HV", "BL")
NUMBER_SPOKEN = {
    13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen",
    17: "seventeen", 18: "eighteen", 19: "nineteen",
}

# (unit, plausible magnitude range) -- keeping magnitudes inside a physically
# sensible band stops the generator from producing quantities no technician
# would ever say, which would make the error model unrepresentative.
UNIT_BANDS = (
    ("nm", 20, 180),
    ("ml", 50, 500),
    ("mg", 5, 90),
    ("bar", 1, 12),
    ("rpm", 800, 1600),
    ("mm", 2, 40),
)

TEMPLATES = (
    "crew {crew} inspected the {component} {part} at {location} on {date} "
    "torque was {qty1} and we applied {qty2} of grease before the fault code "
    "{fault} cleared",
    "on {date} at {location} the {component} {part} was tested reading {qty1} "
    "and {qty2} then fault code {fault} was logged by crew {crew}",
    "we replaced the {component} {part} at {location} the bearing was greased "
    "with {qty2} and torque checked at {qty1} fault code {fault} on {date}",
)

COMPONENTS = ("gearbox", "pitch actuator", "yaw drive", "blade root",
              "hydraulic pump")
CREWS = ("alpha", "bravo", "charlie", "delta")


@dataclass
class Utterance:
    reference: str
    gold: List[Entity]

    def gold_of(self, etype: str) -> List[Entity]:
        return [e for e in self.gold if e.etype == etype]


def _format_quantity(rng: random.Random) -> Tuple[str, float, str]:
    unit, low, high = rng.choice(UNIT_BANDS)
    magnitude = rng.randint(low, high)
    # Some magnitudes are spoken as words.  The extractor has to normalise
    # them, otherwise "fifteen mg" and "15 mg" would look like different
    # readings and a correct transcript would score as an error.
    if magnitude in NUMBER_SPOKEN and rng.random() < 0.5:
        surface = "%s %s" % (NUMBER_SPOKEN[magnitude], unit)
    else:
        surface = "%d %s" % (magnitude, unit)
    return surface, float(magnitude), unit


def make_utterance(rng: random.Random) -> Utterance:
    part = "%s-%d" % (rng.choice(PART_PREFIXES), rng.randint(1000, 9999))
    fault = "E-%03d" % rng.randint(100, 999)
    location = rng.choice(LOCATION_GAZETTEER)
    date = "2024-%02d-%02d" % (rng.randint(1, 12), rng.randint(1, 28))
    q1, m1, u1 = _format_quantity(rng)
    q2, m2, u2 = _format_quantity(rng)
    template = rng.choice(TEMPLATES)
    text = template.format(
        crew=rng.choice(CREWS),
        component=rng.choice(COMPONENTS),
        part=part,
        location=location,
        date=date,
        qty1=q1,
        qty2=q2,
        fault=fault,
    )
    gold = _plant(text, part, fault, location, date, (q1, m1, u1), (q2, m2, u2))
    return Utterance(reference=text, gold=gold)


def _plant(text, part, fault, location, date, qa, qb) -> List[Entity]:
    """Locate each known field in the rendered text and record it as gold.

    Positions come from the surface forms the template was given, so the gold
    list is derived from the record, never from the extractor.
    """
    tokens = tokenize(text)
    gold: List[Entity] = []

    gold.append(Entity(PART_ID, part.upper(), _find(tokens, [part.lower()])))
    gold.append(Entity(FAULT_CODE, fault.upper(), _find(tokens, [fault.lower()])))
    gold.append(Entity(DATE, date, _find(tokens, [date])))
    gold.append(Entity(LOCATION, location, _find(tokens, location.split())))
    for surface, magnitude, unit in (qa, qb):
        pos = _find(tokens, surface.split(), taken=[e.pos for e in gold])
        gold.append(
            Entity(QUANTITY, quantity_value(magnitude, unit), pos,
                   magnitude=magnitude, unit=unit)
        )
    gold.sort(key=lambda e: e.pos)
    return gold


def _find(tokens: Sequence[str], needle: Sequence[str], taken=()) -> int:
    needle = [t.lower() for t in needle]
    for i in range(len(tokens) - len(needle) + 1):
        if list(tokens[i:i + len(needle)]) == needle and i not in taken:
            return i
    raise ValueError("planted surface form not found: %r" % (needle,))


def make_corpus(n: int, seed: int = 11) -> List[Utterance]:
    rng = random.Random(seed)
    return [make_utterance(rng) for _ in range(n)]


# ---------------------------------------------------------------------------
# The headline case: two hand-built hypotheses whose ranking flips.
# ---------------------------------------------------------------------------

HEADLINE_REFERENCE = (
    "technician inspected gearbox GB-4471 at north ridge on 2024-03-12 "
    "torque was 47 nm and we applied 250 ml of grease before the fault code "
    "E-212 cleared"
)

# System A damages five tokens, every one of them a field: part ID digit,
# date transposition, unit slip, unit slip, fault-code digit.
HEADLINE_HYP_A = (
    "technician inspected gearbox GB-4771 at north ridge on 2024-03-21 "
    "torque was 47 mm and we applied 250 mg of grease before the fault code "
    "E-272 cleared"
)

# System B damages six tokens, every one of them narrative: it is the worse
# transcript by WER and the only usable one by the maintenance system.
HEADLINE_HYP_B = (
    "technician expected gearbox GB-4471 at north ridge on 2024-03-12 "
    "torque is 47 nm an he applied 250 ml off grease before a fault code "
    "E-212 cleared"
)


def headline_case() -> Tuple[str, List[Entity], str, str]:
    """Reference, planted gold entities, and the two rival hypotheses."""
    gold = _plant(
        HEADLINE_REFERENCE,
        "GB-4471",
        "E-212",
        "north ridge",
        "2024-03-12",
        ("47 nm", 47.0, "nm"),
        ("250 ml", 250.0, "ml"),
    )
    return HEADLINE_REFERENCE, gold, HEADLINE_HYP_A, HEADLINE_HYP_B
