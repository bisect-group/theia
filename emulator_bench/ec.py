import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Set, Tuple


LEVEL_DEPTH = {"ec1": 1, "ec12": 2, "ec123": 3, "ec1234": 4}
EC_RE = re.compile(r"^(?P<a>[1-7])(?:\.(?P<b>\d+|-|x))?(?:\.(?P<c>\d+|-|x))?(?:\.(?P<d>\d+|n\d*|-|x))?$", re.I)


@dataclass(frozen=True)
class ParsedEC:
    parts: Tuple[str, str, str, str]
    provisional: bool = False
    partial: bool = False


def normalize_ec(value: str) -> str:
    text = str(value).strip()
    if text.lower().startswith("ec:"):
        text = text.split(":", 1)[1].strip()
    return text.replace(" ", "")


def is_unlabeled_ec(value: str) -> bool:
    return normalize_ec(value).lower() in {"", "nan", "none", "null", "-.-.-.-", "x.x.x.x"}


def parse_ec(value: str, allow_provisional: bool = False) -> ParsedEC:
    text = normalize_ec(value)
    match = EC_RE.match(text)
    if not match:
        raise ValueError(f"Invalid EC number syntax: {value}")
    raw = [match.group(name) for name in ("a", "b", "c", "d")]
    parts = [raw[0]]
    for index in range(1, 4):
        parts.append("-" if raw[index] is None else raw[index].lower())

    seen_unspecified = False
    for index, part in enumerate(parts):
        unspecified = part in {"-", "x"}
        if seen_unspecified and not unspecified:
            raise ValueError(f"Invalid EC number with specified child after wildcard: {value}")
        seen_unspecified = seen_unspecified or unspecified
        if part.startswith("n") and index != 3:
            raise ValueError(f"Provisional EC marker is only valid in L4: {value}")
        if not unspecified and not part.startswith("n"):
            number = int(part)
            if index in {1, 2} and not 1 <= number <= 99:
                raise ValueError(f"EC L{index + 1} component out of accepted range 1-99: {value}")
            if index == 3 and not 1 <= number <= 999:
                raise ValueError(f"EC L4 component out of accepted range 1-999: {value}")
    if parts[3].startswith("n") and not allow_provisional:
        raise ValueError(f"Provisional EC labels are disabled by default: {value}")
    return ParsedEC(tuple(parts), provisional=parts[3].startswith("n"), partial=any(part in {"-", "x"} for part in parts))


def ec_label(value: str, level: str, allow_provisional: bool = False) -> str:
    if level not in LEVEL_DEPTH:
        raise ValueError(f"Unsupported EC level: {level}")
    parsed = parse_ec(value, allow_provisional=allow_provisional)
    parts = parsed.parts[: LEVEL_DEPTH[level]]
    if any(part in {"-", "x"} for part in parts):
        raise ValueError(f"Cannot build {level} label from partial EC {value}")
    return ".".join(parts)


def build_level_vocab(values: Iterable[str], level: str, allow_provisional: bool = False) -> Set[str]:
    vocab: Set[str] = set()
    for value in values:
        if is_unlabeled_ec(value):
            continue
        try:
            vocab.add(ec_label(value, level, allow_provisional=allow_provisional))
        except ValueError:
            continue
    return vocab


def partial_match(true_ec: str, pred_label: str, level: str, vocab: Optional[Set[str]] = None, allow_provisional: bool = False) -> bool:
    try:
        pred_parts = parse_ec(pred_label, allow_provisional=allow_provisional).parts[: LEVEL_DEPTH[level]]
        true_parts = parse_ec(true_ec, allow_provisional=allow_provisional).parts[: LEVEL_DEPTH[level]]
    except ValueError:
        return False
    if vocab is not None and ".".join(pred_parts) not in vocab:
        return False
    for true_part, pred_part in zip(true_parts, pred_parts):
        if true_part in {"-", "x"}:
            return True
        if true_part != pred_part:
            return False
    return True


def hierarchy_accuracies(true_values: Sequence[str], pred_values: Sequence[str], vocab_by_level: Optional[dict] = None) -> dict:
    result = {}
    for level in ("ec1", "ec12", "ec123", "ec1234"):
        total = 0
        correct = 0
        for true, pred in zip(true_values, pred_values):
            if is_unlabeled_ec(true):
                continue
            total += 1
            correct += int(partial_match(true, pred, level, None if vocab_by_level is None else vocab_by_level.get(level)))
        result[f"{level}_accuracy"] = correct / total if total else 0.0
    return result
