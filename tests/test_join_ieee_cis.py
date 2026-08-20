import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from generate.join_ieee_cis import _sample_legitimate_transactions, _relabel_benign_subtlety

# Tiny in-memory fixture, not the real 650MB IEEE-CIS file — per plan.md
# §5, tests run against small fixture data to keep CI fast.
_FIXTURE = pd.DataFrame({
    "TransactionID": range(6),
    "isFraud": [0, 1, 0, 1, 0, 0],
    "TransactionAmt": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
})


def test_sample_only_draws_from_legitimate_rows():
    sampled = _sample_legitimate_transactions(_FIXTURE, n=3)
    assert len(sampled) == 3
    assert (sampled["isFraud"] == 0).all()


def test_sample_is_reproducible_with_fixed_seed():
    a = _sample_legitimate_transactions(_FIXTURE, n=2)
    b = _sample_legitimate_transactions(_FIXTURE, n=2)
    pd.testing.assert_frame_equal(a, b)


def test_relabel_benign_subtlety_avoids_pandas_na_string():
    # Regression test: pandas' default read_csv NA-value list includes
    # "n/a" — writing that literal into a CSV silently becomes NaN on a
    # normal load. "benign" doesn't collide with any pandas NA sentinel.
    result = _relabel_benign_subtlety(["n/a", "obvious", "n/a", "subtle"])
    assert result == ["benign", "obvious", "benign", "subtle"]

    roundtrip = pd.read_csv(pd.io.common.StringIO("subtlety\n" + "\n".join(result)))
    assert roundtrip["subtlety"].isnull().sum() == 0
