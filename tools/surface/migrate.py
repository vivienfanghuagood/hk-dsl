"""Rewrite call sites from the DSL spellings the surface migration deprecates to the new ones
(docs/design/dsl-surface.md §5; docs/migrating.md), in place:

    python3 tools/surface/migrate.py FILE.py...          # rewrite, print a line per file
    python3 tools/surface/migrate.py --check FILE.py...  # only report; exit 1 if anything would change

S2: hk.mma_ABt_mx(a, b, acc, sa, sb, a_fmt=, b_fmt=, sa_at=, sb_at=, into=) ->
    hk.mma("mma_ABt", a.scaled(sa, fmt=, at=), b.scaled(sb, fmt=, at=), acc, into=);
    hk.copy_mx_scales(dst, src) -> src.copy(into=dst);
    hk.load_fp4_frags_lds(...) -> hk.load_fp4_frags(...).

**The order trap** (docs/design/dsl-surface.md §5.1, docs/migrating.md). A kernel is its trace,
and the trace is the order ops are emitted in. The old call evaluates its arguments a, b, acc,
sa, sb; the new one a, sa, b, sb, acc. When arguments that emit ops -- an inline load
(`w.at[...].load(T)`, `hk.load_mx_scales(...)`), `hk.zeros(...)`, any other DSL call -- would be
evaluated in another relative order, the new spelling would be another kernel (on gfx950 the load
issue order is performance). So those arguments are bound to locals first, in the old order, and
the new spelling takes the locals: the generated C++ stays byte-identical. Taken not to emit: a
name, a constant, an attribute of one, and a subscript of a name by names, constants and their
arithmetic (read as indexing a Python list; `--strict` treats it as emitting). A call nested in a larger
expression cannot be reordered without moving what is evaluated before it; it is left as it is
and reported.
"""

from __future__ import annotations

import argparse
import ast
import sys

_STMT = (ast.Expr, ast.Assign, ast.AnnAssign, ast.Return)
_INDEX = (ast.Constant, ast.Name, ast.BinOp, ast.UnaryOp, ast.Tuple, ast.Slice, ast.operator,
          ast.unaryop, ast.Load)


def emits(node: ast.AST, strict: bool = False) -> bool:
    """Whether evaluating `node` may emit ops. A name, a constant or an attribute of one does not;
    nor -- unless `strict` -- a subscript of a name by names, constants and their arithmetic,
    which is taken to index a Python list (`S[u % 2]`, `ws[k][0]`). Anything else (a call, DSL
    arithmetic, `w.at[...]`) may. An index into a DSL value does emit (an LDS stack `S[i]`); the
    tool cannot see types, so a file that passes such an argument is migrated with `--strict`."""
    if isinstance(node, (ast.Name, ast.Constant)):
        return False
    if isinstance(node, ast.Attribute):
        return emits(node.value, strict)
    if isinstance(node, (ast.Tuple, ast.List)):
        return any(emits(e, strict) for e in node.elts)
    if isinstance(node, ast.Subscript) and not strict:
        base = node.value
        while isinstance(base, ast.Subscript):
            if not all(isinstance(n, _INDEX) for n in ast.walk(base.slice)):
                return True
            base = base.value
        return not (isinstance(base, ast.Name)
                    and all(isinstance(n, _INDEX) for n in ast.walk(node.slice)))
    return True


class _Rewriter:
    def __init__(self, src: str, strict: bool = False):
        self.src = src
        self.strict = strict
        self.edits = []          # (start offset, end offset, text)
        self.manual = []         # (line, why)
        self.n = 0
        self.names = {n.id for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Name)}
        self.lines = src.splitlines(keepends=True)
        self.offs = [0]
        for line in self.lines:
            self.offs.append(self.offs[-1] + len(line))

    def seg(self, node):
        return ast.get_source_segment(self.src, node)

    def wrap(self, node, text=None):
        t = text if text is not None else self.seg(node)
        ok = isinstance(node, (ast.Name, ast.Attribute, ast.Subscript, ast.Call))
        return t if ok or text is not None else f"({t})"

    def at(self, line, col):
        return self.offs[line - 1] + col

    def fresh(self, role):
        while True:
            self.n += 1
            name = f"_{role}{self.n}"
            if name not in self.names:
                self.names.add(name)
                return name

    def hoist(self, call, stmt, args, new_order):
        """args: [(role, node)] in the old call's evaluation order; new_order: the roles in the new
        spelling's. Returns {role: text} for the new spelling. When the arguments that may emit
        ops would be evaluated in another relative order, they are bound to locals before the
        statement, in the old order, and the new spelling takes the locals. None when that cannot
        be done (the call is nested in a larger expression)."""
        texts = {role: self.seg(node) for role, node in args}
        emitting = [role for role, node in args if emits(node, self.strict)]
        if [r for r in new_order if r in emitting] == emitting:
            return texts
        # The call must be the whole statement (or the whole right-hand side of an assignment,
        # which Python evaluates before its targets), on a line of its own: then nothing else in
        # the statement is evaluated before the arguments, and binding them first moves nothing.
        if stmt is None or stmt.value is not call:
            return None
        indent = self.lines[stmt.lineno - 1][:stmt.col_offset]
        if indent.strip():
            return None
        nodes = dict(args)
        pre = f"{indent}# (bound in the old call's argument order, which is the trace's)\n"
        for role in emitting:
            name = self.fresh(role)
            pre += f"{indent}{name} = {self.seg(nodes[role])}\n"
            texts[role] = name
        self.edits.append((self.at(stmt.lineno, 0), self.at(stmt.lineno, 0), pre))
        return texts

    def operand(self, t, args, role):
        """An argument as the object of a method call: its local, or its source (parenthesized
        unless it is a primary)."""
        node = dict(args)[role]
        return t[role] if t[role] != self.seg(node) else self.wrap(node)

    def scaled(self, t, args, o, s, fmt, at):
        extra = [f"fmt={t[fmt]}"] if fmt in t else []
        extra += [f"at={t[at]}"] if at in t else []
        return f"{self.operand(t, args, o)}.scaled({', '.join([t[s], *extra])})"

    def visit(self, tree):
        parents = {}
        for p in ast.walk(tree):
            for c in ast.iter_child_nodes(p):
                parents[c] = p
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call):
                continue
            f = call.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            prefix = self.seg(f.value) + "." if isinstance(f, ast.Attribute) else ""
            stmt = parents.get(call)
            stmt = stmt if isinstance(stmt, _STMT) else None
            kw = {k.arg: k.value for k in call.keywords}
            if name == "mma_ABt_mx":
                if len(call.args) != 5 or not set(kw) <= {"a_fmt", "b_fmt", "sa_at", "sb_at", "into"}:
                    self.manual.append((call.lineno, "mma_ABt_mx with other arguments"))
                    continue
                args = list(zip(("a", "b", "acc", "sa", "sb"), call.args)) + [
                    (k.arg, k.value) for k in call.keywords]
                new_order = ["a", "sa", "a_fmt", "sa_at", "b", "sb", "b_fmt", "sb_at", "acc", "into"]
                t = self.hoist(call, stmt, args, new_order)
                if t is None:
                    self.manual.append((call.lineno, ("inline loads in a nested call: bind them "
                                                      "to locals in the old order by hand")))
                    continue

                parts = ['"mma_ABt"', self.scaled(t, args, "a", "sa", "a_fmt", "sa_at"),
                         self.scaled(t, args, "b", "sb", "b_fmt", "sb_at"), t["acc"]]
                parts += [f"into={t['into']}"] if "into" in t else []
                new = f"{prefix}mma({', '.join(parts)})"
            elif name == "copy_mx_scales" and len(call.args) == 2 and not kw:
                args = [("dst", call.args[0]), ("src", call.args[1])]
                t = self.hoist(call, stmt, args, ["src", "dst"])
                if t is None:
                    self.manual.append((call.lineno, ("copy_mx_scales of two op-emitting "
                                                      "arguments in a nested call")))
                    continue
                new = f"{self.operand(t, args, 'src')}.copy(into={t['dst']})"
            elif name == "load_fp4_frags_lds":
                parts = [self.seg(a) for a in call.args] + [
                    f"{k.arg}={self.seg(k.value)}" for k in call.keywords]
                new = f"{prefix}load_fp4_frags({', '.join(parts)})"
            else:
                continue
            self.edits.append((self.at(call.lineno, call.col_offset),
                               self.at(call.end_lineno, call.end_col_offset), new))

    def result(self):
        out = self.src
        for a, b, text in sorted(self.edits, key=lambda e: (e[0], e[1]), reverse=True):
            out = out[:a] + text + out[b:]
        return out


def rewrite_source(src: str, strict: bool = False):
    """(new source, [(line, why)] left for a person)."""
    r = _Rewriter(src, strict)
    r.visit(ast.parse(src))
    return r.result(), r.manual


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--check", action="store_true")
    p.add_argument("--strict", action="store_true",
                   help="treat every argument but a name or a constant as emitting ops")
    p.add_argument("files", nargs="+")
    a = p.parse_args(argv)
    changed = 0
    for path in a.files:
        with open(path) as f:
            src = f.read()
        new, manual = rewrite_source(src, a.strict)
        for line, why in manual:
            print(f"{path}:{line}: left as it is: {why}")
        if new != src:
            changed += 1
            if not a.check:
                with open(path, "w") as f:
                    f.write(new)
            print(f"{path}: {'would be ' if a.check else ''}rewritten")
    return 1 if a.check and changed else 0


if __name__ == "__main__":
    sys.exit(main())
