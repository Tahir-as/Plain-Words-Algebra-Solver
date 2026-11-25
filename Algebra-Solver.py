#!/usr/bin/env python3
"""
 algebra solver (standard library only).

Usage:
    python algebra_solver.py "Twice a number plus 5 equals 21"
    python algebra_solver.py "3(x - 4) = 2x + 1"
    python algebra_solver.py            # interactive prompt
    python algebra_solver.py --test     # run built-in self-tests

Pipeline: normalise text -> translate words to an equation string ->
parse both sides into exact polynomials -> solve -> verify.
Handles one unknown. Linear and quadratic solved exactly; degree 3+ numerically.
"""
import re
import sys
import cmath
import math
from fractions import Fraction as F

# ---------------------------------------------------------------- number words
UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
TENS = {w: 10 * (i + 2) for i, w in enumerate(
    "twenty thirty forty fifty sixty seventy eighty ninety".split())}


def words_to_numbers(s):
    s = re.sub(r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)-"
               r"(one|two|three|four|five|six|seven|eight|nine)\b", r"\1 \2", s)
    out, run = [], None

    def flush():
        nonlocal run
        if run:
            out.append(str(run["tot"] + run["cur"]) + run["suf"])
            run = None

    for tok in s.split():
        m = re.match(r"^([a-z]+)([^a-z]*)$", tok)
        w, suf = (m.group(1), m.group(2)) if m else ("", "")
        if not (w in UNITS or w in TENS or w in ("hundred", "thousand")):
            flush()
            out.append(tok)
            continue
        if run is None:
            run = {"cur": 0, "tot": 0, "suf": ""}
        if w in UNITS:
            run["cur"] += UNITS[w]
        elif w in TENS:
            run["cur"] += TENS[w]
        elif w == "hundred":
            run["cur"] = (run["cur"] or 1) * 100
        else:
            run["tot"] += (run["cur"] or 1) * 1000
            run["cur"] = 0
        run["suf"] = suf
    flush()
    return " ".join(out)


# ---------------------------------------------------------------- translation
ATOM = r"(@\d+|\d+(?:\.\d+)?|\b[a-z]\b)"
NUM = r"(-?\d+(?:\.\d+)?)"


def _rule(pattern, fn):
    return re.compile(pattern.replace("{A}", ATOM)), fn


def build_rules(g):
    """Each rule rewrites a phrase into a bracketed placeholder token."""
    return [
        _rule(r"\b(twice|double|triple|thrice)\s+{A}",
              lambda m: g(f"({2 if m[1] in ('twice', 'double') else 3}*{m[2]})")),
        _rule(r"\b(square|cube)\s+of\s+{A}",
              lambda m: g(f"({m[2]}^{2 if m[1] == 'square' else 3})")),
        _rule(r"{A}\s+(doubled|tripled|halved|squared|cubed)\b",
              lambda m: g({"doubled": f"({m[1]}*2)", "tripled": f"({m[1]}*3)",
                           "halved": f"({m[1]}/2)", "squared": f"({m[1]}^2)",
                           "cubed": f"({m[1]}^3)"}[m[2]])),
        _rule(r"\bhalf\s+(?:of\s+)?{A}", lambda m: g(f"({m[1]}/2)")),
        _rule(r"\b(third|quarter|fifth|tenth)\s+of\s+{A}",
              lambda m: g(f"({m[2]}/{ {'third': 3, 'quarter': 4, 'fifth': 5, 'tenth': 10}[m[1]] })")),
        _rule(r"{A}\s*(?:%|percent)\s+of\s+{A}", lambda m: g(f"({m[1]}/100*{m[2]})")),
        _rule(r"{A}\s*(?:%|percent)", lambda m: g(f"({m[1]}/100)")),
        _rule(r"{A}\s+(?:to\s+the\s+power\s+of|raised\s+to(?:\s+the\s+power\s+of)?)\s+{A}",
              lambda m: g(f"({m[1]}^{m[2]})")),
        _rule(r"{A}\s+(?:times|multiplied\s+by)\s+{A}", lambda m: g(f"({m[1]}*{m[2]})")),
        _rule(r"{A}\s+(?:divided\s+by|over)\s+{A}", lambda m: g(f"({m[1]}/{m[2]})")),
        _rule(r"{A}\s+(?:less\s+than|fewer\s+than|subtracted\s+from|taken\s+from)\s+{A}",
              lambda m: g(f"({m[2]}-{m[1]})")),
        _rule(r"{A}\s+(?:more\s+than|greater\s+than|added\s+to)\s+{A}",
              lambda m: g(f"({m[2]}+{m[1]})")),
        _rule(r"{A}\s+(?:minus|decreased\s+by|reduced\s+by)\s+{A}",
              lambda m: g(f"({m[1]}-{m[2]})")),
        _rule(r"{A}\s+(?:plus|increased\s+by)\s+{A}", lambda m: g(f"({m[1]}+{m[2]})")),
        _rule(r"\b(?:sum|total)\s+of\s+{A}\s+and\s+{A}", lambda m: g(f"({m[1]}+{m[2]})")),
        _rule(r"\bdifference\s+(?:of|between)\s+{A}\s+and\s+{A}",
              lambda m: g(f"({m[1]}-{m[2]})")),
        _rule(r"\bproduct\s+of\s+{A}\s+and\s+{A}", lambda m: g(f"({m[1]}*{m[2]})")),
        _rule(r"\bquotient\s+of\s+{A}\s+and\s+{A}", lambda m: g(f"({m[1]}/{m[2]})")),
    ]


class SolveError(Exception):
    pass


def translate(raw):
    """Return (equation_string, consecutive_info_or_None)."""
    s = (raw.lower().replace("−", "-").replace("–", "-").replace("×", "*")
         .replace("÷", "/").replace("²", "^2").replace("³", "^3").strip())
    s = words_to_numbers(s)
    if not s:
        raise SolveError("Type a problem first.")

    if re.search(r"[a-z]{2,}", s):  # natural language
        nums = re.findall(r"-?\d+(?:\.\d+)?", s)

        # consecutive integers
        m = re.search(r"\b([2-6])\s+consecutive\s+(even\s+|odd\s+)?(?:integers|numbers)", s)
        if m and re.search(r"\b(sum|product)\b", s) and len(nums) >= 2:
            k, step = int(m[1]), 2 if m[2] else 1
            op = "*" if re.search(r"\bproduct\b", s) else "+"
            parts = ["x"] + [f"(x+{j * step})" for j in range(1, k)]
            return op.join(parts) + "=" + nums[-1], (k, step)

        # "I think of a number, double it, add 5, and get 17"
        if re.search(r"\b(think|thinking|pick|choose|start|take|have)\b[^.]*\bnumber\b", s) and nums:
            e, ops = "x", 0
            for clause in re.split(r",|\.|;|\bthen\b|\band\b", s):
                if re.search(r"\bdouble", clause):
                    e, ops = f"({e})*2", ops + 1
                elif re.search(r"\btriple", clause):
                    e, ops = f"({e})*3", ops + 1
                elif re.search(r"\bhalve", clause):
                    e, ops = f"({e})/2", ops + 1
                elif re.search(r"\bsquare", clause):
                    e, ops = f"({e})^2", ops + 1
                else:
                    for pat, sym in ((r"(?:multiply|multiplied|times)", "*"),
                                     (r"divide", "/"),
                                     (r"(?:add|plus|increase)", "+"),
                                     (r"(?:subtract|minus|take away|decrease|reduce)", "-")):
                        mm = re.search(pat + r"\D*?" + NUM, clause)
                        if mm:
                            e, ops = f"({e}){sym}{mm[1]}", ops + 1
                            break
            if ops:
                return e + "=" + nums[-1], None

        s = re.sub(r"[?!,;:]", " ", s)
        s = re.sub(r"\.(?!\d)", " ", s)
        s = re.sub(r"\bwhat(?:['’]s)?(?:\s+(?:is|are))?\b", " ", s)
        s = re.sub(r"\b(?:(?:a|an|the|some|this|that)\s+)?"
                   r"(?:(?:certain|unknown|mystery|particular|given)\s+)?"
                   r"(?:number|unknown|integer|value)\b", " x ", s)
        s = re.sub(r"\bx\s+x\b", "x", s)
        s = re.sub(r"\b(solve|find|calculate|compute|determine|evaluate|simplify|if|then|"
                   r"please|for|me|us|a|an|the|of the)\b", " ", s)

        groups = []

        def g(text):
            groups.append(text)
            return f" @{len(groups) - 1} "

        rules = build_rules(g)
        for _ in range(200):
            for pat, fn in rules:
                m = pat.search(s)
                if m:
                    s = s[:m.start()] + fn(m) + s[m.end():]
                    break
            else:
                break

        s = re.sub(r"\b(?:is equal to|is the same as|equal to|equals|equal|results? in|will be|"
                   r"gives|give|makes|yields|(?:you |we |i )?(?:get|gets|got|obtain)|are|was|is)\b",
                   " = ", s)
        for _ in range(50):
            if "@" not in s:
                break
            s = re.sub(r"@(\d+)", lambda m: groups[int(m[1])], s)
        words = re.findall(r"[a-z]{2,}", s)
        if words:
            raise SolveError("I didn't recognise: " + ", ".join(sorted(set(words))) +
                             ". Try rephrasing, or write the equation with symbols.")

    s = re.sub(r"\s+", "", s)
    bad = re.search(r"[^0-9a-z.+\-*/^()=]", s)
    if bad:
        raise SolveError(f"I can't read the character '{bad[0]}'.")
    return s, None


# ---------------------------------------------------------------- polynomials
# A polynomial is a list of Fractions, index = degree.
def trim(p):
    p = list(p)
    while len(p) > 1 and p[-1] == 0:
        p.pop()
    return p


def padd(a, b, sign=1):
    n = max(len(a), len(b))
    return trim([(a[i] if i < len(a) else 0) + sign * (b[i] if i < len(b) else 0)
                 for i in range(n)])


def pmul(a, b):
    r = [F(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            r[i + j] += x * y
    return trim(r)


def pval(p, x):
    acc = 0
    for c in reversed(p):
        acc = acc * x + c
    return acc


def tokenize(s):
    toks = re.findall(r"\d+\.?\d*|\.\d+|[a-z]|[-+*/^()]", s)
    return toks


def parse_poly(tokens):
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def starts_atom(t):
        return t is not None and (t[0].isdigit() or t[0] == "." or t.isalpha() or t == "(")

    def take():
        nonlocal pos
        t = peek()
        pos += 1
        return t

    def expr():
        left = term()
        while peek() in ("+", "-"):
            op = take()
            left = padd(left, term(), 1 if op == "+" else -1)
        return left

    def term():
        left = unary()
        while True:
            t = peek()
            if t == "*":
                take()
                left = pmul(left, unary())
            elif t == "/":
                take()
                d = unary()
                if len(d) > 1:
                    raise SolveError("I can't divide by something containing the unknown.")
                if d[0] == 0:
                    raise SolveError("That divides by zero.")
                left = [c / d[0] for c in left]
            elif starts_atom(t):  # implicit multiplication: 3x, 2(x+1)
                left = pmul(left, unary())
            else:
                return left

    def unary():
        if peek() == "-":
            take()
            return [-c for c in unary()]
        if peek() == "+":
            take()
            return unary()
        return power()

    def power():
        base = atom()
        if peek() == "^":
            take()
            e = unary()
            if len(e) > 1 or e[0].denominator != 1 or not (0 <= e[0] <= 8):
                raise SolveError("Exponents must be whole numbers from 0 to 8.")
            r = [F(1)]
            for _ in range(int(e[0])):
                r = pmul(r, base)
            return r
        return base

    def atom():
        t = take()
        if t is None:
            raise SolveError("The expression ends too early.")
        if t == "(":
            r = expr()
            if take() != ")":
                raise SolveError("A bracket is never closed.")
            return r
        if t[0].isdigit() or t[0] == ".":
            try:
                return [F(t)]
            except ValueError:
                raise SolveError(f"'{t}' is not a number.")
        if t.isalpha():
            return [F(0), F(1)]
        raise SolveError(f"Unexpected '{t}'.")

    result = expr()
    if pos < len(tokens):
        raise SolveError(f"Unexpected '{tokens[pos]}'.")
    return result


# ---------------------------------------------------------------- formatting
SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def sup(s):
    return re.sub(r"\^(\d+)", lambda m: m[1].translate(SUP), s)


def num(x):
    """Exact Fraction -> '3/4'; float -> trimmed decimal."""
    if isinstance(x, F):
        return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.6g}"


def num_full(x):
    t = num(x)
    return f"{t} (≈ {float(x):.4g})" if "/" in t else t


def poly_str(p, v="x"):
    out = ""
    for k in range(len(p) - 1, -1, -1):
        c = p[k]
        if c == 0 and len(p) > 1:
            continue
        a = abs(c)
        cs = num(a)
        cf = f"({cs})" if "/" in cs else cs
        if k == 0:
            body = cs
        else:
            var = v if k == 1 else f"{v}^{k}"
            body = var if a == 1 else cf + var
        if out == "":
            out = ("-" if c < 0 else "") + body
        else:
            out += (" - " if c < 0 else " + ") + body
    return sup(out or "0")


def strip_outer(s):
    while s.startswith("(") and s.endswith(")"):
        depth = 0
        for i, ch in enumerate(s):
            depth += (ch == "(") - (ch == ")")
            if depth == 0 and i < len(s) - 1:
                return s
        s = s[1:-1]
    return s


def pretty(eq):
    eq = "=".join(strip_outer(side) for side in eq.split("="))
    eq = re.sub(r"\((\d+(?:\.\d+)?|[a-z])\)", r"\1", eq).replace("*", "·")
    eq = re.sub(r"([\w)])([+\-=])", r"\1 \2 ", eq)
    return sup(re.sub(r"\s+", " ", eq).strip())


# ---------------------------------------------------------------- root finding
def exact_sqrt(fr):
    """sqrt of a non-negative Fraction: exact Fraction if perfect square, else float."""
    n, d = fr.numerator, fr.denominator
    rn, rd = math.isqrt(n), math.isqrt(d)
    return F(rn, rd) if rn * rn == n and rd * rd == d else math.sqrt(fr)


def durand_kerner(p):
    """Real roots of a polynomial of degree >= 3 (floating point)."""
    c = [float(x) for x in p]
    n = len(c) - 1
    c = [x / c[n] for x in c]
    radius = 1 + max(abs(x) for x in c[:n])
    z = [radius * cmath.exp(1j * (2 * math.pi * k / n + 0.4)) for k in range(n)]

    def ev(x):
        r = 1
        for k in range(n - 1, -1, -1):
            r = r * x + c[k]
        return r

    for _ in range(2000):
        for i in range(n):
            d = 1
            for j in range(n):
                if j != i:
                    d *= z[i] - z[j]
            z[i] -= ev(z[i]) / d
    real = sorted(r.real for r in z if abs(r.imag) < 1e-6)
    out = []
    for r in real:
        if not out or abs(r - out[-1]) > 1e-5:
            out.append(r)
    return out


# ---------------------------------------------------------------- solver
def solve(raw):
    eq, seq = translate(raw)
    parts = eq.split("=")
    if len(parts) > 2:
        raise SolveError('I found more than one "equals". Try simpler sentences.')
    letters = sorted(set(re.findall(r"[a-z]", eq)))
    if len(letters) > 1:
        raise SolveError("I can only solve for one unknown, but I found: " + ", ".join(letters))
    v = letters[0] if letters else "x"

    res = {"interp": pretty(eq), "steps": [], "answers": [], "notes": []}

    if len(parts) == 1:
        p = parse_poly(tokenize(parts[0]))
        if len(p) == 1:
            res["answers"].append("= " + num_full(p[0]))
        else:
            res["steps"].append("No equals sign, so I simplified the expression.")
            res["answers"].append(poly_str(p, v))
        return res

    if not parts[0] or not parts[1]:
        raise SolveError("One side of the equation is empty.")
    L, R = parse_poly(tokenize(parts[0])), parse_poly(tokenize(parts[1]))
    res["steps"].append(f"Expand and tidy each side: {poly_str(L, v)} = {poly_str(R, v)}")
    D = padd(L, R, -1)
    deg = len(D) - 1
    roots = []

    if deg == 0:
        always = D[0] == 0
        res["steps"].append("Both sides are always equal." if always else
                            "Both sides reduce to different numbers, so they can never match.")
        res["answers"].append("every real number works" if always else "no solution")
        return res

    if deg == 1:
        b, a = D
        res["steps"].append(f"Move everything to one side: {poly_str(D, v)} = 0")
        if b != 0:
            res["steps"].append(f"Move the constant across: {poly_str([F(0), a], v)} = {num(-b)}")
        res["steps"].append(f"Divide both sides by {num(a)}: {v} = {num(-b / a)}")
        roots = [-b / a]
    elif deg == 2:
        c, b, a = D
        disc = b * b - 4 * a * c
        res["steps"].append(f"Move everything to one side: {poly_str(D, v)} = 0")
        res["steps"].append(f"Quadratic formula with a = {num(a)}, b = {num(b)}, c = {num(c)}.")
        res["steps"].append(f"Discriminant: b² − 4ac = {num(disc)}")
        if disc == 0:
            roots = [-b / (2 * a)]
            res["steps"].append("The discriminant is zero: one repeated solution.")
        elif disc > 0:
            s = exact_sqrt(disc)
            roots = sorted([(-b - s) / (2 * a), (-b + s) / (2 * a)], key=float)
            res["steps"].append("The discriminant is positive: two real solutions.")
        else:
            re_part = -b / (2 * a)
            im_part = math.sqrt(-disc) / float(2 * abs(a))
            res["steps"].append("The discriminant is negative: no real solutions, only complex ones.")
            res["answers"].append(f"{v} = {num(re_part)} ± {im_part:.4g}i")
            return res
    else:
        res["steps"].append(f"Move everything to one side: {poly_str(D, v)} = 0")
        roots = durand_kerner(D)
        res["steps"].append("Degree 3 and up has no short formula, so I found roots numerically.")
        if not roots:
            res["answers"].append("no real solutions")
            return res
        res["notes"].append("Values are approximate.")

    for r in roots:
        if not isinstance(r, F) and abs(r) < 1e-10:
            r = 0.0
        res["answers"].append(f"{v} = {num_full(r)}")
        lv, rv = pval(L, r), pval(R, r)
        ok = abs(float(lv) - float(rv)) < 1e-6 * max(1.0, abs(float(lv)))
        res["steps"].append(f"Check {v} = {num(r)}: left side {num(lv)}, right side {num(rv)}"
                            + (" ✓" if ok else " (approximately)"))

    if seq:
        k, step = seq
        whole = [r for r in roots if abs(float(r) - round(float(r))) < 1e-9]
        if not whole:
            res["notes"].append("None of these are whole numbers, so no set of consecutive integers fits.")
        for r in whole:
            start = round(float(r))
            res["notes"].append("The numbers: " + ", ".join(str(start + j * step) for j in range(k)))
    return res


def show(raw):
    try:
        r = solve(raw)
    except SolveError as e:
        print(f"\n  Could not solve: {e}\n")
        return
    print(f"\n  I read it as:  {r['interp']}")
    if r["steps"]:
        print("\n  Steps")
        for i, s in enumerate(r["steps"], 1):
            print(f"   {i}. {s}")
    print("\n  Answer:  " + "   or   ".join(r["answers"]).replace("= -", "= −"))
    for n in r["notes"]:
        print(f"  {n}")
    print()


# ---------------------------------------------------------------- self-test
TESTS = [
    ("Twice a number plus 5 equals 21", ["x = 8"]),
    ("Seven less than three times a number is twenty", ["x = 9"]),
    ("Half of a number is 14", ["x = 28"]),
    ("20% of a number is 30", ["x = 150"]),
    ("The sum of three consecutive integers is 72", ["x = 23"]),
    ("I think of a number, double it, add 5, and get 17", ["x = 6"]),
    ("3(x - 4) = 2x + 1", ["x = 13"]),
    ("x^2 - 5x + 6 = 0", ["x = 2", "x = 3"]),
    ("2x + 1 = 2x + 3", ["no solution"]),
    ("A number divided by 4 is 6", ["x = 24"]),
    ("5 + 3 * 2", ["= 11"]),
    ("x^3 - 6x^2 + 11x - 6 = 0", ["x = 1", "x = 2", "x = 3"]),
]


def self_test():
    bad = 0
    for text, want in TESTS:
        try:
            got = solve(text)["answers"]
        except SolveError as e:
            got = [f"ERROR {e}"]
        ok = got == want
        bad += not ok
        print(("PASS " if ok else "FAIL ") + text + ("" if ok else f"   got {got}, want {want}"))
    print(f"\n{len(TESTS) - bad}/{len(TESTS)} passed")
    return bad


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--test"]:
        sys.exit(1 if self_test() else 0)
    if args:
        show(" ".join(args))
    else:
        print("Plain-words algebra solver. Type a problem, or 'q' to quit.")
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if line.lower() in ("q", "quit", "exit"):
                break
            if line:
                show(line)