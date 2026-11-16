"""Conservative fixes to legacy answer extraction/equivalence.

Adds balanced fbox/boxed extraction, membership wrappers, choice wrappers and
exact rational matrix cells. This is not a general symbolic algebra grader.
"""
import re
from fractions import Fraction

from math_normalize import is_equiv, strip_string


def extract_answer(text):
    boxes = list(re.finditer(r'\\(?:boxed|fbox)\b\s*', text))
    if not boxes:
        return None
    pos = boxes[-1].end()
    if pos >= len(text):
        return None
    if text[pos] != '{':
        return re.split(r'\$|\n', text[pos:], maxsplit=1)[0].strip() or None
    depth = 0
    for i in range(pos, len(text)):
        if text[i] == '{': depth += 1
        elif text[i] == '}': depth -= 1
        if depth == 0:
            return text[pos + 1:i]
    return None  # Do not replace an unfinished last answer with an earlier one.


def normalize_wrapper(s):
    s = s.strip().replace('\\left', '').replace('\\right', '')
    s = re.sub(r'^\s*[a-zA-Z]\s*\\in\s*', '', s)
    choice = re.fullmatch(r'(?:\\(?:text|mathrm)\{)?\s*\(?([A-Z])\)?\s*\}?', s)
    return choice.group(1) if choice else s


def rational(s):
    s = re.sub(r'\s+', '', s).replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    m = re.fullmatch(r'([+-]?)\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}', s)
    try:
        if m: return (-1 if m[1] == '-' else 1) * Fraction(int(m[2]), int(m[3]))
        if re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?\d+)?', s):
            return Fraction(s)
    except (ValueError, ZeroDivisionError):
        pass
    return None


def matrix(s):
    m = re.fullmatch(r'\s*\\begin\{([pbvBV]?matrix)\}(.*?)\\end\{\1\}\s*', s, re.S)
    if not m:
        return None
    # Preserve row/column shape before legacy normalization collapses slashes.
    rows = [tuple(cell.strip() for cell in row.split('&')) for row in re.split(r'\\\\', m[2])]
    return tuple(rows)


def equivalent(pred, gold):
    if pred is None or gold is None:
        return False
    pred, gold = normalize_wrapper(pred), normalize_wrapper(gold)
    pm, gm = matrix(pred), matrix(gold)
    if pm is not None or gm is not None:
        if pm is None or gm is None or len(pm) != len(gm): return False
        return all(len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b)) for a, b in zip(pm, gm))
    pr, gr = rational(pred), rational(gold)
    if pr is not None and gr is not None:
        return pr == gr
    return is_equiv(pred, gold)
