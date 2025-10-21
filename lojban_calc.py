
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Union, Optional, Dict, Iterable
import re
import sys
import textwrap

# ----------------------------
# Tokenization
# ----------------------------

ALLOWED_CHARS_RE = re.compile(r'^[A-Za-z0-9\s\.]*\Z')
NUM_RE = re.compile(r'^(0|[1-9][0-9]*)\Z')
NAME_RE = re.compile(r'^\.[A-Za-z]+\.\Z')

def lower_ascii(s: str) -> str:
    return s.lower()

def is_gismu(word: str) -> bool:
    if len(word) != 5:
        return False
    consonants = set("bcdfghjklmnpqrstvxzwy")
    vowels = set("aeiou")
    w = word
    def C(ch): return ch in consonants
    def V(ch): return ch in vowels
    return (C(w[0]) and V(w[1]) and C(w[2]) and C(w[3]) and V(w[4])) or \
           (C(w[0]) and C(w[1]) and V(w[2]) and C(w[3]) and V(w[4]))

@dataclass
class Token:
    typ: str           # 'WORD', 'NUMBER', 'NAME', 'I'
    text: str

def tokenize(src: str) -> List[Token]:
    if not ALLOWED_CHARS_RE.match(src):
        for ch in src:
            if not re.match(r'[A-Za-z0-9\s\.]', ch):
                raise ValueError(f"Illegal character in input: {repr(ch)}")
    raw = [t for t in re.split(r'(\s+)', src) if t and not t.isspace()]
    toks: List[Token] = []
    for t in raw:
        lt = lower_ascii(t)
        if lt == 'i':
            toks.append(Token('I', 'i'))
        elif NUM_RE.match(lt):
            toks.append(Token('NUMBER', lt))
        elif NAME_RE.match(t):
            toks.append(Token('NAME', t))
        else:
            if not re.match(r'^[A-Za-z]+\Z', t):
                raise ValueError(f"Invalid token: {t}")
            toks.append(Token('WORD', lower_ascii(t)))
    return toks

# ----------------------------
# Values & Calls
# ----------------------------

@dataclass(frozen=True)
class Atom:
    name: str
    def __repr__(self) -> str: return self.name

@dataclass(frozen=True)
class Var:
    name: str
    def __repr__(self) -> str: return self.name

Value = Union[int, 'Atom', 'ListVal', 'PredCall']

@dataclass(frozen=True)
class ListVal:
    items: Tuple[Value, ...]
    def __repr__(self) -> str:
        inner = " ".join(map(repr, self.items))
        return f"({inner})"

@dataclass
class PredCall:
    name: str
    args: List[Value]

@dataclass
class Rule:
    head: PredCall
    body: List[PredCall]

# ----------------------------
# Parser
# ----------------------------

class Parser:
    def __init__(self, tokens: List[Token]):
        self.toks = tokens
        self.i = 0
    def peek(self) -> Optional[Token]:
        return self.toks[self.i] if self.i < len(self.toks) else None
    def take(self) -> Token:
        if self.i >= len(self.toks):
            raise ValueError("Unexpected end of input")
        t = self.toks[self.i]; self.i += 1; return t
    def at_end(self) -> bool:
        return self.i >= len(self.toks)

BUILTIN_PREDICATES = {"fatci", "sumji", "vujni", "dunli", "steni", "steko", "cmavo"}

def parse_name_as_value(tok: Token) -> Value:
    nm = tok.text
    inner = nm.strip('.')
    if inner.isupper():
        return Var(nm)
    else:
        return Atom(nm)

def parse_number(tok: Token) -> int:
    return int(tok.text)

def parse_argument(p: Parser, user_pred_names: set) -> Value:
    t = p.peek()
    if t is None: raise ValueError("Unexpected end while parsing argument")
    if t.typ == 'NUMBER':
        p.take(); return parse_number(t)
    if t.typ == 'WORD' and t.text == 'lo':
        p.take()
        nxt = p.take()
        if nxt.typ == 'WORD' and nxt.text == 'steni':
            return ListVal(())
        elif nxt.typ == 'WORD' and nxt.text == 'steko':
            head = parse_list_element(p, user_pred_names)
            tail = parse_argument(p, user_pred_names)
            if not isinstance(tail, ListVal):
                raise ValueError("steko must be followed by list (lo steni/lo steko ...)")
            return ListVal((head,) + tail.items)
        elif nxt.typ == 'NAME':
            return parse_name_as_value(nxt)
        elif nxt.typ == 'WORD':
            return Atom(f".{nxt.text}.")
        else:
            raise ValueError(f"Invalid token after 'lo': {nxt.text}")
    if t.typ == 'NAME':
        p.take(); return parse_name_as_value(t)
    raise ValueError(f"Invalid argument start: {t.text}")

def parse_list_element(p: Parser, user_pred_names: set) -> Value:
    save = p.i
    try:
        a1 = parse_argument(p, user_pred_names)
        t = p.take()
        if t.typ != 'WORD':
            raise ValueError("Expected predicate word in list element")
        pred = t.text
        if not (is_gismu(pred) or pred in user_pred_names or pred in BUILTIN_PREDICATES):
            raise ValueError("Not a predicate")
        args = [a1]
        for _ in range(1, 5):
            nxt = p.peek()
            if nxt is None: break
            if nxt.typ == 'I': break
            try:
                args.append(parse_argument(p, user_pred_names))
            except Exception:
                break
        return PredCall(pred, args)
    except Exception:
        p.i = save
        return parse_argument(p, user_pred_names)

@dataclass
class Statement:
    call: PredCall

class Program:
    def __init__(self):
        self.rules: Dict[str, List[Rule]] = {}
        self.store: Dict[str, Value] = {}

    def parse(self, src: str) -> List[Statement]:
        tokens = tokenize(src)
        p = Parser(tokens)
        stmts: List[Statement] = []
        while not p.at_end():
            t = p.take()
            if t.typ != 'I':
                raise ValueError(f"Every statement must start with 'i', got '{t.text}'")
            stmts.append(self._parse_statement_body(p))
        return stmts

    def _parse_statement_body(self, p: Parser) -> Statement:
        seg: List[Token] = []
        while not p.at_end() and p.peek().typ != 'I':
            seg.append(p.take())
        sp = Parser(seg)
        # Optional 'se' before first arg
        pre_se = False
        if sp.peek() and sp.peek().typ == 'WORD' and sp.peek().text == 'se':
            sp.take(); pre_se = True
        arg1 = parse_argument(sp, set(self.rules.keys()))
        tw = sp.take()
        if tw.typ != 'WORD': raise ValueError("Expected a predicate word")
        pred = tw.text
        if not (is_gismu(pred) or pred in self.rules or pred in BUILTIN_PREDICATES):
            raise ValueError(f"Unknown or invalid predicate word: {pred}")
        # Optional 'se' before second arg
        post_se = False
        if sp.peek() and sp.peek().typ == 'WORD' and sp.peek().text == 'se':
            sp.take(); post_se = True
        args: List[Value] = [arg1]
        if pred == 'cmavo':
            # Accept 2 or 3 args (body defaults to empty list)
            arglist = parse_argument(sp, set(self.rules.keys()))
            bodylist: Value
            if not sp.at_end():
                try:
                    bodylist = parse_argument(sp, set(self.rules.keys()))
                except Exception:
                    bodylist = ListVal(())
            else:
                bodylist = ListVal(())
            args.extend([arglist, bodylist])
        else:
            while not sp.at_end():
                args.append(parse_argument(sp, set(self.rules.keys())))
        if pre_se or post_se:
            if len(args) >= 2:
                args[0], args[1] = args[1], args[0]
        return Statement(PredCall(pred, args))

    def eval(self, stmts: List[Statement]) -> List[str]:
        for st in stmts[:-1]:
            self._eval_statement(st, as_query=False)
        result_bindings = self._eval_statement(stmts[-1], as_query=True)
        final_vars: List[Var] = []
        def collect_vars(v: Value):
            if isinstance(v, Var): final_vars.append(v)
            elif isinstance(v, ListVal):
                for it in v.items: collect_vars(it)
        for a in stmts[-1].call.args: collect_vars(a)
        seen = set(); outputs: List[str] = []
        for v in final_vars:
            if v.name in seen: continue
            seen.add(v.name)
            bound = result_bindings.get(v.name)
            outputs.append(render_value(bound) if bound is not None else f"{v.name}=unbound")
        return outputs

    def _eval_statement(self, st: Statement, as_query: bool) -> Dict[str, Value]:
        call = st.call
        if call.name in BUILTIN_PREDICATES:
            return self._eval_builtin(call, as_query)
        rules = self.rules.get(call.name, [])
        for sol in solve(call, rules, self):
            return sol
        return {}

    def _eval_builtin(self, call: PredCall, as_query: bool) -> Dict[str, Value]:
        name = call.name
        if name == 'fatci':
            if len(call.args) != 1: raise ValueError("fatci takes exactly 1 argument")
            if isinstance(call.args[0], Atom):
                self.store[call.args[0].name] = call.args[0]
            return {}
        if name == 'sumji':
            if len(call.args) != 3: raise ValueError("sumji takes exactly 3 arguments")
            a2 = evaluate_value(call.args[1], self.store)
            a3 = evaluate_value(call.args[2], self.store)
            if not (isinstance(a2, int) and isinstance(a3, int)):
                raise ValueError("sumji: arguments 2 and 3 must be numbers")
            res = a2 + a3
            return self._assign_or_check(call.args[0], res)
        if name == 'vujni':
            if len(call.args) != 3: raise ValueError("vujni takes exactly 3 arguments")
            a2 = evaluate_value(call.args[1], self.store)
            a3 = evaluate_value(call.args[2], self.store)
            if not (isinstance(a2, int) and isinstance(a3, int)):
                raise ValueError("vujni: arguments 2 and 3 must be numbers")
            res = a2 - a3
            return self._assign_or_check(call.args[0], res)
        if name == 'dunli':
            if len(call.args) != 2: raise ValueError("dunli takes exactly 2 arguments")
            a1 = evaluate_value(call.args[0], self.store)
            a2 = evaluate_value(call.args[1], self.store)
            if not value_equal(a1, a2):
                raise ValueError("dunli: arguments are not equal")
            return {}
        if name == 'steni':
            if len(call.args) != 1: raise ValueError("steni takes exactly 1 argument")
            return self._assign_or_check(call.args[0], ListVal(()))
        if name == 'steko':
            if len(call.args) not in (2, 3): raise ValueError("steko takes 2 or 3 arguments")
            head = evaluate_value(call.args[1], self.store)
            tail = ListVal(())
            if len(call.args) == 3:
                tail = evaluate_value(call.args[2], self.store)
                if not isinstance(tail, ListVal): raise ValueError("steko: third argument must be a list")
            lst = ListVal((head,) + tail.items)
            return self._assign_or_check(call.args[0], lst)
        if name == 'cmavo':
            if len(call.args) != 3:
                raise ValueError("cmavo takes 3 args internally (parser supplies empty body if omitted)")
            pred_name_val = call.args[0]
            if isinstance(pred_name_val, Atom):
                pred_name = pred_name_val.name.strip('.')
            elif isinstance(pred_name_val, Var):
                pred_name = pred_name_val.name.strip('.')
            else:
                raise ValueError("cmavo: predicate name must be a name or word")
            arglist_val = evaluate_value(call.args[1], self.store)
            if not isinstance(arglist_val, ListVal):
                raise ValueError("cmavo: second argument must be a list")
            arg_vars: List[Value] = list(arglist_val.items)
            body_val = evaluate_value(call.args[2], self.store)
            body_calls: List[PredCall] = []
            if isinstance(body_val, ListVal):
                for elem in body_val.items:
                    if isinstance(elem, PredCall): body_calls.append(elem)
            rule = Rule(PredCall(pred_name, arg_vars), body_calls)
            self.rules.setdefault(pred_name, []).append(rule)
            return {}
        raise ValueError(f"Unknown builtin: {name}")

    def _assign_or_check(self, target: Value, value: Value) -> Dict[str, Value]:
        if isinstance(target, Atom):
            key = target.name
            if key in self.store:
                if not value_equal(self.store[key], value):
                    raise ValueError(f"Inconsistent assignment for {key}")
            else:
                self.store[key] = value
            return {}
        if isinstance(target, Var):
            return {target.name: value}
        if not value_equal(evaluate_value(target, self.store), value):
            raise ValueError("Assignment target does not match value")
        return {}

# ----------------------------
# Unification / solving
# ----------------------------

Subst = Dict[str, Value]

def evaluate_value(v: Value, store: Dict[str, Value]) -> Value:
    if isinstance(v, Atom): return store.get(v.name, v)
    if isinstance(v, ListVal): return ListVal(tuple(evaluate_value(it, store) for it in v.items))
    return v

def value_equal(a: Value, b: Value) -> bool:
    if isinstance(a, Atom) and isinstance(b, Atom): return a.name == b.name
    if isinstance(a, int) and isinstance(b, int): return a == b
    if isinstance(a, ListVal) and isinstance(b, ListVal):
        return len(a.items) == len(b.items) and all(value_equal(x, y) for x, y in zip(a.items, b.items))
    return False

def unify(a: Value, b: Value, subst: Subst, store: Dict[str, Value]) -> Optional[Subst]:
    def deref(v: Value) -> Value:
        if isinstance(v, Atom): return store.get(v.name, v)
        return v
    a = deref(a); b = deref(b)
    if isinstance(a, Var): return unify_var(a, b, subst, store)
    if isinstance(b, Var): return unify_var(b, a, subst, store)
    if isinstance(a, int) and isinstance(b, int): return subst if a == b else None
    if isinstance(a, Atom) and isinstance(b, Atom): return subst if a.name == b.name else None
    if isinstance(a, ListVal) and isinstance(b, ListVal):
        if len(a.items) != len(b.items): return None
        for x, y in zip(a.items, b.items):
            subst = unify(x, y, subst, store)
            if subst is None: return None
        return subst
    return None

def unify_var(v: Var, x: Value, subst: Subst, store: Dict[str, Value]) -> Optional[Subst]:
    if v.name in subst: return unify(subst[v.name], x, subst, store)
    if occurs_in(v, x, subst, store): return None
    subst[v.name] = x; return subst

def occurs_in(v: Var, x: Value, subst: Subst, store: Dict[str, Value]) -> bool:
    if isinstance(x, Var):
        if x.name == v.name: return True
        if x.name in subst: return occurs_in(v, subst[x.name], subst, store)
        return False
    if isinstance(x, ListVal):
        return any(occurs_in(v, it, subst, store) for it in x.items)
    return False

def solve(goal: PredCall, rules: List[Rule], program: 'Program') -> Iterable[Subst]:
    for rule in rules:
        if rule.head.name != goal.name: continue
        if len(rule.head.args) != len(goal.args): continue
        subst: Subst = {}
        ok = True
        for a, b in zip(rule.head.args, goal.args):
            sub2 = unify(a, b, subst, program.store)
            if sub2 is None: ok = False; break
            subst = sub2
        if not ok: continue
        if not rule.body:
            yield subst; continue
        def solve_body(idx: int, subst: Subst) -> Iterable[Subst]:
            if idx == len(rule.body): 
                yield subst; return
            body_call = rule.body[idx]
            applied_args = [apply_subst(arg, subst, program.store) for arg in body_call.args]
            applied = PredCall(body_call.name, applied_args)
            if applied.name in BUILTIN_PREDICATES:
                try:
                    result = program._eval_builtin(applied, as_query=True)
                except Exception:
                    return
                new_subst = subst.copy()
                new_subst.update(result)
                yield from solve_body(idx + 1, new_subst)
            else:
                for deeper in solve(applied, program.rules.get(applied.name, []), program):
                    merged = subst.copy(); merged.update(deeper)
                    yield from solve_body(idx + 1, merged)
        yield from solve_body(0, subst)

def apply_subst(v: Value, subst: Subst, store: Dict[str, Value]) -> Value:
    if isinstance(v, Var): return subst.get(v.name, v)
    if isinstance(v, ListVal): return ListVal(tuple(apply_subst(it, subst, store) for it in v.items))
    if isinstance(v, Atom): return store.get(v.name, v)
    return v

def render_value(v: Value) -> str:
    if isinstance(v, int): return str(v)
    if isinstance(v, Atom): return v.name
    if isinstance(v, ListVal):
        inner = " ".join(render_value(it) for it in v.items); return f"({inner})"
    if isinstance(v, Var): return v.name
    return str(v)

# ----------------------------
# Public API
# ----------------------------

def run_program(src: str) -> List[str]:
    prog = Program()
    stmts = prog.parse(src)
    return prog.eval(stmts)

# ----------------------------
# CLI
# ----------------------------

def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            src = f.read()
    else:
        src = sys.stdin.read()
    outs = run_program(src)
    print("\n".join(outs))

if __name__ == "__main__":
    main()
