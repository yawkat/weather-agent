"""Queries for the server log, without where they are about.

`Redacted(text)` keeps a query's shape (methods, variables, reductions, thresholds, times, models) and hides what
locates it: coordinates, place names, polylines, and names the client chose (bindings, dict keys, places(Name=…)).
Strings are kept only if they are known vocabulary or ISO times; numbers are hidden where they may be coordinates.
Messages (errors, warnings) may quote the query's locations, so the log never shows them: `at(span)` gives the
redacted node an ExprError points to instead.
"""

import ast
import re
from datetime import datetime

from ..variables import CATALOG
from .axes import DIMS
from .language import COORDINATES, MAX_CHARS, NUMPY, REDUCTIONS, RESAMPLE, SPECIAL, _check_size

VOCABULARY = {*CATALOG, *DIMS, *RESAMPLE, *COORDINATES, "time.hour", "distance_km"}
LOCATING = ("places", "distance_from")  # route(…) keeps speed and start; its polyline is an unknown string
COORDINATE_NAMES = ("lat", "lon")
# Names that aren't the client's: built-ins the language knows, or rejects with a hint.
# Attributes and keyword arguments the language knows; others may be the client's words (`fc.koeln`).
METHODS = {"interp", "sel", "isel", "where", "clip", "round", "rolling", "resample", "groupby", "groupby_bins",
           "idxmax", "idxmin", "sortby", "quantile", "dt", "hour", "dayofweek"}
KEYWORDS = {"ascending", "bins", "by", "cond", "decimals", "dim", "gpx", "max", "min", "n", "other", "polyline", "q",
            "speed_kmh", "start", "use_gpx_times"}
BUILTINS = {*SPECIAL, "slice", "range", "max", "min", "sum", "len", "round", "any", "all", "True", "False", "None"}
HIDDEN = ...  # numbers are shown as `...`


def _is_time(text: str) -> bool:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}[0-9T:+\-. Z]*", text):
        return False
    try:
        datetime.fromisoformat(text)
    except ValueError:
        return False
    return True


def _combines(node: ast.AST) -> bool:
    """Conditions joined with &, |, ^: each part decides on its own whether it compares coordinates."""
    return isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr, ast.BitXor))


def _mentions_coordinates(node: ast.AST) -> bool:
    if _combines(node):
        return False
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, ast.Name) and n.id in COORDINATE_NAMES or \
                isinstance(n, ast.Attribute) and n.attr in COORDINATE_NAMES or \
                isinstance(n, ast.Constant) and n.value in COORDINATE_NAMES:
            return True
        stack.extend(c for c in ast.iter_child_nodes(n) if not _combines(c) and not isinstance(c, ast.Compare))
    return False


def _is_literal(node: ast.AST) -> bool:
    return not any(isinstance(n, (ast.Call, ast.Name, ast.Attribute)) for n in ast.walk(node))


def _is_pair(node: ast.AST) -> bool:
    def number(n):
        if isinstance(n, ast.UnaryOp):
            n = n.operand
        return isinstance(n, ast.Constant) and type(n.value) in (int, float)
    return isinstance(node, (ast.Tuple, ast.List)) and len(node.elts) == 2 and all(map(number, node.elts))


class _Redactor(ast.NodeTransformer):
    def __init__(self, keep: set[str]):
        self.keep = keep
        self.aliases: dict[tuple[str, str], str] = {}  # (kind, original) → placeholder
        self.locating = 0  # depth of subtrees whose numbers are hidden
        self.labelling = 0  # inside places(…)/distance_from(…): every string and keyword is the client's

    def alias(self, kind: str, original: str) -> str:
        key = (kind, original)
        if key not in self.aliases:
            self.aliases[key] = f"{kind}{sum(k == kind for k, _ in self.aliases) + 1}"
        return self.aliases[key]

    def hiding(self, node: ast.AST, hide: bool) -> ast.AST:
        self.locating += hide
        try:
            return self.generic_visit(node)
        finally:
            self.locating -= hide

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        # `home = (50.9, 6.9)` or `la = 50.9` may be a location in disguise.
        return self.hiding(node, _is_literal(node.value))

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id not in BUILTINS and node.id not in self.keep:  # `precip = fc.precip` locates nothing
            node.id = self.alias("v", node.id)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        func = node.func
        locating = isinstance(func, ast.Name) and func.id in LOCATING
        if locating:
            for k in node.keywords:  # places(Köln=(50.9, 6.9))
                if k.arg is not None:
                    k.arg = self.alias("p", k.arg)
        self.labelling += locating
        try:
            return self.hiding(node, locating)
        finally:
            self.labelling -= locating

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        if node.attr not in self.keep and node.attr not in NUMPY and node.attr not in REDUCTIONS \
                and node.attr not in METHODS:
            node.attr = self.alias("a", node.attr)
        return self.generic_visit(node)

    def visit_keyword(self, node: ast.keyword) -> ast.AST:
        if node.arg is not None and not self.labelling and node.arg not in self.keep and node.arg not in KEYWORDS:
            node.arg = self.alias("k", node.arg)
        return self.hiding(node, node.arg in COORDINATE_NAMES)

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        return self.hiding(node, _mentions_coordinates(node))

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        return self.hiding(node, _mentions_coordinates(node))

    def visit_Tuple(self, node: ast.Tuple) -> ast.AST:
        return self.hiding(node, _is_pair(node))

    def visit_List(self, node: ast.List) -> ast.AST:
        return self.hiding(node, _is_pair(node))

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        value = node.value
        if isinstance(value, str):
            if not self.labelling and (value in self.keep or _is_time(value)):
                return node
            return ast.copy_location(ast.Constant(self.alias("s", value)), node)
        if type(value) in (int, float, complex) and self.locating:
            return ast.copy_location(ast.Constant(HIDDEN), node)
        return node


class Redacted:
    def __init__(self, text: str, keep=()):
        """`keep`: further strings that locate nothing, e.g. model names."""
        self.keep = VOCABULARY | set(keep)
        self.tree: ast.Module | None = None
        try:
            if len(text) > MAX_CHARS:
                raise ValueError
            tree = ast.parse(text.strip(), mode="exec")  # as parse() does, so ExprError spans match
            _check_size(tree)  # bounds the walks below, as for the query itself
            self.tree = _Redactor(self.keep).visit(tree)  # replaced nodes keep their spans
            self.text = ast.unparse(self.tree).replace("\n", "; ")
        except BaseException as e:  # SyntaxError, ExprError (too large), or anything unexpected: log no query
            if isinstance(e, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                raise
            self.tree = None
            self.text = f"<unparseable or too large, {len(text)} characters>"

    def __str__(self) -> str:
        return self.text

    def at(self, span: tuple[int, int, int, int] | None) -> str | None:
        """The redacted node at a span (lineno, col_offset, end_lineno, end_col_offset) of the query, if any."""
        if self.tree is None or span is None:
            return None
        for node in ast.walk(self.tree):
            if span == tuple(getattr(node, a, None) for a in ("lineno", "col_offset", "end_lineno", "end_col_offset")):
                return ast.unparse(node).replace("\n", "; ")
        return None
