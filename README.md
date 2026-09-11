# fieldvoice

> Offline speech-to-structured field reports evaluated on entity error rate, not word error rate - because only the part numbers matter.

> ### No speech recogniser, by design of the environment
>
> This repository contains **no ASR runtime, no audio, no model weights and
> no network access**. Hypotheses come from a deterministic, seeded
> **simulated ASR backend** sitting behind the `AsrBackend` interface: it
> takes a reference transcript plus a noise / accent / domain-vocabulary
> profile and emits a realistically corrupted hypothesis using phonetic
> confusion classes, number and unit mangling, and out-of-vocabulary
> spell-out of domain identifiers.
>
> What is real, and what the test suite attacks, is the **interface and the
> evaluation** - the Levenshtein WER, the typed entity extraction, and the
> entity error rate. Dropping a real whisper.cpp binding in behind
> `AsrBackend` would change nothing else in the repo.

`FLAGSHIP` · **AI / ML Engineering** · Advanced · ~4-5 weeks · Energy - wind turbine maintenance crews

**Primary language:** Python
**Tags:** `speech`, `offline-ai`, `constrained-decoding`, `evaluation`, `edge`, `whisper`

---

## The problem

A technician at the top of a turbine cannot type a report. They speak it, and later somebody transcribes it into a form. Generic transcription gets the narrative roughly right and mangles exactly the parts that matter - part numbers, torque values, fault codes - which are the only fields the maintenance system actually consumes.

## ⭐ The differentiator

Evaluates on **entity-level error rate for the fields that matter, not global WER**, and uses **domain-constrained decoding against a parts-and-fault-code vocabulary** so `GB-4471` cannot be transcribed as 'GB forty-four seventy-one'. A model with 12% global WER but 0.8% part-number error is far more useful here than one at 6% WER that fluffs every identifier - and only the entity-level metric reveals that. Runs fully offline, because there is no connectivity at the top of a turbine.

This is the sentence to lead with when someone asks you to walk through the
project. Everything else in this repo exists to make it true and to prove it.

## Data

**Common Voice** (CC0) and LibriSpeech for base evaluation, plus a documented generator splicing domain vocabulary into utterances with **known ground-truth entities**, and a small self-recorded domain set.

> No paid API key is required to run or demo this project. Where a paid
> service would add value it is wired as an optional enhancement behind an
> interface with an offline mock as the default implementation.

## Stack

- Python
- whisper.cpp - local, offline, runs on a laptop CPU
- A local model via Ollama for the structuring step
- SQLite, Typer CLI plus a minimal PWA
- Docker, pytest

### Implementation note

The Stack list above names whisper.cpp, an Ollama-hosted local model, Docker
and pytest. None are available in this environment and none are used. The
runnable core is **Python standard library only**: `unittest` instead of
pytest, and a seeded simulated ASR backend instead of whisper.cpp plus Ollama.
The substitution does not weaken the differentiator, because the
differentiator is the *metric*, not the acoustic model - the WER
implementation, the typed entity extraction and the entity error rate are all
real and are what the suite attacks. A real recogniser would be swapped in
behind `AsrBackend` without touching `metrics.py` or `entities.py`.

## Core capabilities

- Fully offline transcription with a domain-constrained decoding vocabulary
- Entity extraction into a strict schema, validated against the parts catalogue
- Dual metric reporting: global WER *and* per-entity-class error rate
- Confidence-gated fields flagged for confirmation rather than silently guessed
- Deferred sync with conflict handling when the crew regains connectivity

## Repository layout

```
src/entities.py    typed field extraction (IDs, quantities+units, dates, places)
src/metrics.py     WER by Levenshtein DP + entity error rate, per type
src/asr.py         AsrBackend interface + deterministic simulated recogniser
src/generator.py   synthetic field reports with PLANTED ground-truth entities
src/demo.py        the 60-second artefact
tests/             62 tests
```

## Build plan

1. Build the dual-metric evaluation harness before touching the model. The metric is the project.
2. Baseline unconstrained transcription and measure entity error. It will be bad in a specific, interesting way.
3. Constrained decoding against the vocabulary, then re-measure.
4. Offline verification - run the whole thing with networking disabled - then the sync path.

## Testing strategy

Assert **entity-class error rate below stated thresholds per class**, with a regression suite that fails if a vocabulary change degrades any class. Assert the full pipeline operates **with networking disabled**, which is the deployment condition and therefore a test rather than a claim.

Tests assert **correctness**, not merely that the code runs. A green suite on
this repo is a claim about behaviour under adversarial conditions; treat any
test that would pass against a deliberately broken implementation as a bug in
the test.

## Quality & safety layer

The system never emits an unvalidated part number. Codes absent from the catalogue route to a confirmation prompt instead of being guessed into a maintenance record.

## Measurable outcome

> Part-number error rate of 4.0% against 56.0% for the unconstrained system,
> at a *worse* global WER (21.3% against 20.1%) - the field that the
> maintenance system consumes is 14x better on the transcript WER would have
> rejected.

(The original spec target was 0.8% against 11%. The numbers above are what
this repo actually measures against its simulated backend; the shape of the
result - a large entity-level gap hidden by a flat or adverse WER - is the
claim, not the specific rates.)

State it in these terms — business units, not technical ones — in your CV
bullet and in the first thirty seconds of describing the project.

## Measured results

All numbers below are printed by `python -m src.demo` on this machine.

### The headline: the ranking flips

One utterance, two hypotheses. System A gets **fewer words wrong**; every one
of them is a field. System B gets **more words wrong**; not one of them is a
field.

```
REFERENCE  technician inspected gearbox GB-4471 at north ridge on 2024-03-12
           torque was 47 nm and we applied 250 ml of grease before the fault
           code E-212 cleared
SYSTEM A   ... GB-4771 ... 2024-03-21 ... 47 mm ... 250 mg ... E-272 ...
SYSTEM B   technician expected ... torque is 47 nm an he applied 250 ml off
           grease before a fault code E-212 cleared
```

| metric | system A | system B | prefers |
|---|---|---|---|
| word error rate | 0.192 | 0.231 | **A** |
| entity error rate | 0.833 | 0.000 | **B** |

A: 5 word errors, 5 of them on scored fields, 0 on narrative.
B: 6 word errors, 0 of them on scored fields, 6 on narrative.

A loses the part number, the date, both quantities and the fault code. Two of
those five are **unit-only** slips (`nm`->`mm`, `ml`->`mg`): the magnitude is
right, so a number-only check calls them correct. B is word-for-word worse and
files a perfect maintenance record.

### Corpus: 200 generated field reports, two ASR profiles

| metric | A-general | B-domain |
|---|---|---|
| word error rate | 0.201 | 0.213 |
| entity error rate | 0.504 | 0.042 |

Per-entity-type error rate, which the average above hides:

| entity type | A-general | B-domain |
|---|---|---|
| PART_ID | 0.560 | 0.040 |
| FAULT_CODE | 0.560 | 0.030 |
| QUANTITY | 0.585 | 0.048 |
| DATE | 0.440 | 0.045 |
| LOCATION | 0.295 | 0.040 |

Ranked by WER: `A-general < B-domain`. Ranked by entity error rate:
`B-domain < A-general`. The flip survives at corpus scale, not only in the
hand-built sentence.

Suite: **62 tests**, `python -m unittest discover -s tests`, under 1 second.

### Limits / caveats

- **The recogniser is simulated.** Absolute error rates here are properties of
  the corruption model, not of any real acoustic model. Only the *relationship*
  between the two metrics is a claim about the world, and it follows from the
  arithmetic of the metrics rather than from the simulator.
- **The corpus-level flip is conditional and it is tested as such.** It needs
  the domain-constrained system to be noisy enough on narrative; the measured
  crossover is around a 0.28 narrative error rate. Below it, `B-domain` wins on
  *both* metrics - operationally better, rhetorically weaker.
  `test_flip_needs_narrative_noise` asserts exactly that, so the claim cannot
  quietly become a universal one. The headline single-utterance flip is
  unconditional, being hand-constructed.
- **The entity error rate is invariant to narrative noise, not to everything.**
  Measured spread across narrative rates 0.05 to 0.45 is under 0.02.
- **Entity extraction is rule-based** over a closed gazetteer and fixed ID
  shapes. A real deployment needs the parts catalogue; an ID absent from the
  catalogue is out of scope here, and confidence gating is not implemented.
- **Same-type entities are aligned by order.** Two quantities spoken in the
  wrong order score as two substitutions. That is the right call for a report
  where field order carries meaning, and the wrong one for free-form dictation.
- **Not implemented from the spec:** constrained decoding itself (there is no
  decoder), the PWA, deferred sync, and SQLite persistence. The domain-
  constrained system is represented as a noise profile, not as a beam search.

## Interview questions this project answers

- **Why is WER the wrong metric here?**
- **How does constrained decoding work?**
- **What does your system do with a code it does not recognise?**

## What this deliberately is *not*

- Not a general transcription service - the constraint is the point.
- Not cloud-dependent; the offline requirement drives every design choice.


## Run it now

```bash
python -m unittest discover -s tests -v   # the suite
python -m src.demo                        # the 60-second artefact
```

Requires Python 3.11+. The runnable core uses **only the standard
library** (including `sqlite3`), so there is nothing to install.

## Getting started

```bash
git clone <your-fork-url> fieldvoice
cd fieldvoice
python -m unittest discover -s tests   # 62 tests, no install, under a second
python -m src.demo                     # the flip, then the corpus tables
```

There is nothing to install and nothing to download: no `pip install`, no
model weights, no Docker, no network.

Every path above works on a plain Windows/macOS/Linux laptop with a
standard Python 3.11+ interpreter and no cloud account.

## Definition of done

- [ ] The differentiator above is implemented, and a test proves it
- [ ] The measurable outcome is produced by a command anyone can run
- [ ] `README` explains the one decision a generic version gets wrong
- [ ] CI runs the full suite on every push and is green on `main`
- [ ] A recruiter can see the headline artefact in under 60 seconds

## Licence

MIT — see [LICENSE](LICENSE).
