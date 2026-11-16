"""Exact-arithmetic checks of the manuscript identities; no model calls.

For N=8, test all coefficients of the homogeneous quadratic partition
identities by evaluating e_i and e_i+e_j, across all 105 unlabelled pairings.
These executable checks complement the general proofs in the manuscript.
"""

import json
from fractions import Fraction as F
from itertools import combinations, product
from pathlib import Path


def mean(values):
    return sum(values, F(0)) / len(values)


def pairings(items):
    if not items:
        yield ()
        return
    first, *rest = items
    for j, second in enumerate(rest):
        for tail in pairings(rest[:j] + rest[j + 1 :]):
            yield ((first, second), *tail)


def check_math():
    partitions = list(pairings(list(range(8))))
    assert len(partitions) == len(set(partitions)) == 105
    supports = [(i,) for i in range(8)] + list(combinations(range(8), 2))
    for support in supports:
        x = [F(i in support) for i in range(8)]
        xb = mean(x)
        diversity = mean([(v - xb) ** 2 for v in x])
        within, between = [], []
        for groups in partitions:
            means = [mean([x[i] for i in group]) for group in groups]
            dw = sum((x[i] - m) ** 2 for g, m in zip(groups, means, strict=True) for i in g) / 8
            db = mean([(m - xb) ** 2 for m in means])
            assert dw + db == diversity
            for target in (F(0), F(3, 7)):
                individual_error = mean([(v - target) ** 2 for v in x])
                assert (xb - target) ** 2 == individual_error - dw - db
            within.append(dw)
            between.append(db)
        assert mean(within) == F(4, 7) * diversity
        assert mean(between) == F(3, 7) * diversity
        assert mean(between) - F(3, 4) * mean(within) == 0

    update_cases = 0
    for alpha, x, a, target in product(
        (F(0), F(2, 5), F(1)), (F(-3), F(0), F(2)), (F(-1), F(0), F(4)), (F(0), F(2))
    ):
        after = (1 - alpha) * x + alpha * a
        terms = (
            (1 - alpha) ** 2 * (x - target) ** 2
            + alpha**2 * (a - target) ** 2
            + 2 * alpha * (1 - alpha) * (x - target) * (a - target)
        )
        assert (after - target) ** 2 == terms
        update_cases += 1

    jensen_cases = 0
    for biases in product((F(-2), F(0), F(3)), repeat=4):
        for weights in ((F(1, 4),) * 4, (F(1, 10), F(2, 10), F(3, 10), F(4, 10))):
            b = sum(w * v for w, v in zip(weights, biases, strict=True))
            gap = sum(w * v**2 for w, v in zip(weights, biases, strict=True)) - b**2
            assert gap == sum(w * (v - b) ** 2 for w, v in zip(weights, biases, strict=True))
            assert (gap == 0) == (len(set(biases)) == 1)
            jensen_cases += 1

    # EVRR = 1 does not imply identical answers: centered a = -3x at alpha=.4.
    x = [F(-1), F(1)]
    a = [-3 * v for v in x]
    after = [F(3, 5) * v + F(2, 5) * z for v, z in zip(x, a, strict=True)]
    assert mean([v**2 for v in after]) / (F(9, 25) * mean([v**2 for v in x])) == 1
    assert a[0] != a[1]
    # Perfect individual answers can still leave an under/overestimated crowd.
    # The reporting transform is reciprocal-symmetric, ordinary % error is not.
    assert F(1, 2) ** -1 - 1 == 2 - 1
    assert abs(F(1, 2) - 1) != abs(2 - 1)
    return {
        "arithmetic": "exact rational; no floating-point tolerance",
        "pairings": len(partitions),
        "quadratic_coefficient_vectors": len(supports),
        "partition_checks": len(partitions) * len(supports),
        "update_checks": update_cases,
        "weighted_jensen_checks": jensen_cases,
        "evrr_nonconvergence_counterexample": "passed",
        "status": "passed",
    }


if __name__ == "__main__":
    result = check_math()
    Path(__file__).with_name("math_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
