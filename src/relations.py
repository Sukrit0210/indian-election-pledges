"""A conservative rule baseline and verbatim evidence candidates for ordered earlier/later pairs."""

import re
from difflib import SequenceMatcher

NUMBER = r"(?:\d[\d,]*(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|fifty|hundred)"
MEASURE = re.compile(
    r"(?P<money>(?:₹|rs\.?|rupees)\s*"
    + NUMBER
    + r"(?:\s*(?:lakh|crore|million|billion))?)|(?P<percent>"
    + NUMBER
    + r"\s*(?:%|per\s*cent\b|percent\b))|(?P<age>"
    + NUMBER
    + r"\s*years?\s*(?:of age|and above|or above|and over|old))|(?P<years>"
    + NUMBER
    + r"\s*years?\b)|(?P<date>\b(?:19|20)\d{2}\b)",
    re.I,
)
WORD = re.compile(r"\w+|₹|%", re.UNICODE)
STOP = set(
    "the of a an for to and will shall we in on by it be is ensure also that all as this with party government".split()
)
DIGITS = dict(
    zip("one two three four five six seven eight nine ten".split(), range(1, 11))
)


def words(t):
    return [m.group().casefold() for m in WORD.finditer(t)]


def norm(t):
    t = t.casefold().replace(",", "")
    for w, v in DIGITS.items():
        t = re.sub(r"\b" + w + r"\b", str(v), t)
    return re.sub(
        r"\s+",
        "",
        re.sub(r"rs\.?|rupees|₹", "rs", re.sub(r"per\s*cent|percent", "%", t)),
    )


def stats(earlier, later):
    a = set(words(earlier)) - STOP
    b = set(words(later)) - STOP
    return len(a & b) / max(1, len(a | b))


def delta(earlier, later):
    lhs = list(MEASURE.finditer(earlier))
    rhs = list(MEASURE.finditer(later))
    if (
        len(lhs) == len(rhs) == 1
        and lhs[0].lastgroup == rhs[0].lastgroup
        and norm(lhs[0].group()) != norm(rhs[0].group())
    ):
        return lhs[0], rhs[0]


def rule(a, b, cos):
    j = stats(a, b)
    if j >= 0.52 and cos >= 0.77 and delta(a, b):
        return "revised"
    if j >= 0.74 and cos >= 0.85 and set(words(a)) == set(words(b)):
        return "same"
    if (
        j >= 0.78
        and cos >= 0.88
        and not delta(a, b)
        and set(re.findall(r"\d+", a)) == set(re.findall(r"\d+", b))
        and not re.search(r"\b(only|except|excluding|not|instead)\b", a + " " + b, re.I)
    ):
        return "same"
    if j < 0.11 and cos < 0.51:
        return "different"
    return "unclear"


def evidence(a, b, label):
    if label != "revised":
        return None
    m = delta(a, b)
    if m:
        return (
            m[0].start(),
            m[0].end(),
            m[1].start(),
            m[1].end(),
            "possible_" + m[0].lastgroup + "_change",
        )
    x = list(WORD.finditer(a))
    y = list(WORD.finditer(b))
    ops = SequenceMatcher(
        None,
        [m.group().lower() for m in x],
        [m.group().lower() for m in y],
        autojunk=False,
    ).get_opcodes()
    changes = [
        (i, j, k, l)
        for tag, i, j, k, l in ops
        if tag == "replace"
        and j - i <= 12
        and l - k <= 12
        and (set(words(a[x[i].start() : x[j - 1].end()])) - STOP)
        and (set(words(b[y[k].start() : y[l - 1].end()])) - STOP)
    ]
    if not changes:
        return None
    i, j, k, l = max(changes, key=lambda z: min(z[1] - z[0], z[3] - z[2]))
    return (
        x[i].start(),
        x[j - 1].end(),
        y[k].start(),
        y[l - 1].end(),
        "lexical_edit_candidate",
    )
