from .core import get_name_by_index
import lark
import operator as op
from typing import FrozenSet,Dict, List, Collection, NoReturn, Tuple
import functools
import textwrap
import itertools

@functools.lru_cache(None)
def get_expr_parser():
    return lark.Lark(textwrap.dedent("""
    ?start: expr
    
    expr: NAME
        | setop
        | negate 
        | "(" expr ")"
    
    negate: "~" expr
    
    setop: expr BINOP expr
    
    BINOP: "&"
        | "^"
        | "-"
        | "|"
    
    NAME: ("a".."z" | "A".."Z" )+
    
    %import common.WS_INLINE
    %ignore WS_INLINE
    """))

OP_MAP = {
    '|': op.or_,
    '^': op.xor,
    '&': op.and_,
    '-': op.sub
}

def _evaluate_ast(n, named_datasets : Dict[str, FrozenSet[int]]):
    if isinstance(n, lark.Token):
        try:
            return named_datasets[n.value]
        except KeyError:
            raise SyntaxError(f"`{n.value}` is not defined (must be one of: {','.join(named_datasets.keys())})")

    if n.data == 'expr':
        return _evaluate_ast(n.children[0], named_datasets)
    elif n.data == 'setop':
        a = _evaluate_ast(n.children[0], named_datasets)
        b = _evaluate_ast(n.children[2], named_datasets)
        c = n.children[1].value
        return OP_MAP[c](a, b)
    elif n.data == 'negate':
        return named_datasets['all'] - _evaluate_ast(n.children[0])
    else:
        raise RuntimeError("something with wrong during expression evaluation (bug)")


def _group_consecutive(arr: List[int]) -> List[List[int]]:
    grouped = []
    g = [arr[0]]
    for i, j in zip(arr, arr[1:]):
        if j - i == 1:
            g.append(j)
        else:
            grouped.append(g)
            g = [j]

    grouped.append(g)
    return grouped


def format_dataset_index_range(idxs: Collection[int]) -> str:
    result = []
    for g in _group_consecutive(sorted(idxs)):
        if len(g) <= 2:
            result.extend(map(str, g))
        else:
            result.append(f'{g[0]:d}-{g[-1]:d}')
    return ','.join(result)


def evaluate_expression(e : str, named_datasets : Dict[str, FrozenSet[int]]):
    try:
        ast = get_expr_parser().parse(e)
    except lark.LarkError as e:
        raise SyntaxError("bad expression:\n" + str(e))

    return _evaluate_ast(ast, named_datasets)

def set_from_ranges_inc(l: List[Tuple[int, int]]) -> FrozenSet[int]:
    return frozenset(itertools.chain(*(range(s, e + 1) for s, e in l)))


def main_func(dataset : str, named_datasets : Dict[str, FrozenSet[int]]) -> NoReturn:
    import argparse
    import sys
    p = argparse.ArgumentParser()
    p.add_argument('expr', type=str,
                   help=f"Dataset selection.  Defined datasets: {','.join(named_datasets.keys())}. "
                        f"Arbitrary Python set expressions are supported."
                   )
    g = p.add_mutually_exclusive_group()
    g.add_argument('-c', '--concise', action='store_true', help='concise single-line format compatible with Slurm array specs')
    g.add_argument('-n', '--name', action='store_true', help='Print names instead of indices.')
    g.add_argument('-m', '--map', action='store_true', help="Print map from index to name")
    g.add_argument('-r', '--rev-map', action='store_true', help="Like --map but reversed")
    args = p.parse_args()

    try:
        selection = evaluate_expression(args.expr, named_datasets)
    except Exception as e:

        print(str(e), file=sys.stderr)
        sys.exit(1)

    if args.concise:
        print(format_dataset_index_range(selection))
    elif args.name:
        for i in sorted(selection):
            print(get_name_by_index(dataset, i))
    elif args.map:
        for i in sorted(selection):
            print(f"{i} {get_name_by_index(dataset, i)}")
    elif args.rev_map:
        for i in sorted(selection):
            print(f"{get_name_by_index(dataset, i)} {i}")
    else:
        for i in sorted(selection):
            print(i)

    sys.exit(0)