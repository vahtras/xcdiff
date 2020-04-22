import itertools
import collections
import sys

n = int(sys.argv[1])

variables = ['ra', 'rb', 'ga', 'gb', 'gab']

for group in itertools.combinations_with_replacement(variables, n):

    counts = collections.Counter(group)
    lhs = f"ds->df{counts['ra']}{counts['rb']}{counts['ga']}{counts['gb']}"
    if counts['gab'] > 0:
        lhs += f"{counts['gab']}"

    rhs = '({ccode(self.F.diff('
    rhs += ', '.join(f'self.{v}' for v in variables for _ in range(counts[v]))
    rhs += '))})*factor;'

    print(14*' ' + lhs, '+=', rhs)
