"""PEAR's robot-use protocol (paper Appendix H), reusable by any environment.

The robot is a set of named *parts*, each with named *quantities*. Every turn the model sees the
current value of every quantity as JSON, then the camera images. Its action is a JSON object of the
same structure giving the *change* of every quantity; it ends its reply with one fenced json block.
Only the ENVIRONMENT section and the TASK instruction are task-specific.

``Schema`` = ordered {part: [quantity, ...]}; it maps between that JSON and a flat action vector.
"""
from __future__ import annotations

import json
import re
from typing import Sequence

import numpy as np

Schema = dict[str, list[str]]

PROTOCOL = """You are controlling a robot. Each turn you see the robot's current state and its camera images, and you respond with the next action.

## OBSERVATION

The robot is made of named parts, and each part has named quantities you can set. Every turn begins with the current value of every such quantity, grouped by part, followed by the camera images:

{{"<part>": {{"<quantity>": <value>, ...}}, ...}}

Lengths are in {length_unit} and angles in radians; any other quantity is a plain number. Nothing beyond these names is given about what each quantity corresponds to.

## ACTION

The action changes the listed quantities by these amounts:

{{"<part>": {{"<quantity>": <change>, ...}}, ...}}

Write the change of every quantity; 0 leaves a quantity as it is.

## EXECUTION

The action ends once every target is reached and the robot is at rest. If something keeps a target from being reached, the robot stops there and the action ends. Either way, you then receive the next observation.

## RESPONSE FORMAT

End your response with exactly one fenced json block containing the action, and nothing after it:

```json
<action>
```

## ENVIRONMENT

{environment}

## TASK

{instruction}"""


def system_prompt(environment: str, instruction: str, length_unit: str = "centimeters") -> str:
    return PROTOCOL.format(environment=environment.strip(), instruction=instruction.strip(), length_unit=length_unit)


def format_state(schema: Schema, values: Sequence[float], ndigits: int = 2) -> str:
    """Flat values (schema order) -> the per-turn state JSON line."""
    it = iter(values)
    return json.dumps({p: {q: round(float(next(it)), ndigits) for q in qs} for p, qs in schema.items()})


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TAG = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)


def extract_action_json(text: str):
    """Last fenced json block (PEAR) or, failing that, last <action>...</action>; parsed JSON or None."""
    for rx in (_FENCE, _TAG):
        blocks = rx.findall(text)
        if blocks:
            try:
                return json.loads(blocks[-1].strip())
            except Exception:
                return None
    return None


def parse_action(text: str, schema: Schema) -> np.ndarray | None:
    """Parse a PEAR-style action into a flat vector (schema order). Every part and quantity must be
    present ("Write the change of every quantity"); anything missing or non-finite -> None."""
    obj = extract_action_json(text)
    if not isinstance(obj, dict):
        return None
    out = []
    try:
        for p, qs in schema.items():
            for q in qs:
                out.append(float(obj[p][q]))
    except (KeyError, TypeError, ValueError):
        return None
    a = np.array(out, dtype=np.float64)
    return a if np.all(np.isfinite(a)) else None


def example_action(schema: Schema, values: Sequence[float] | None = None) -> str:
    vals = list(values) if values is not None else [0.0] * sum(len(q) for q in schema.values())
    return "```json\n" + format_state(schema, vals) + "\n```"
