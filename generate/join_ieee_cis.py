"""
Joins the 210 generated AgentSessions onto real IEEE-CIS transaction rows.

IMPORTANT — this is a FABRICATED pairing, not real paired data. No dataset
exists where a real IEEE-CIS transaction is actually linked to an AI
shopping agent's session (agent commerce didn't exist when IEEE-CIS was
collected in 2019). Each session is joined to a randomly sampled real
transaction purely to give the eventual LightGBM model (Task 5) realistic
transaction-level features (amount, card/device/timing patterns) sitting
alongside our synthetic session-level features. Never state or imply this
pairing is real correspondence — see TEAM_BRIEF.md's fidelity-validation
section for the same caveat applied to the broader synthetic pipeline.

Sampling is restricted to IEEE-CIS rows where isFraud == 0. This isn't
arbitrary: the project's whole thesis (TEAM_BRIEF.md Sec 1.2) is that a
hijacked agent payment is cryptographically valid and looks like an
ordinary transaction on the card rail — classic transaction-level fraud
signals don't catch it. Pairing a hijacked session with a transaction
IEEE-CIS already flagged fraudulent (for unrelated reasons — stolen card,
classic carding, etc.) would muddy that thesis and hand the eventual
LightGBM model a spurious shortcut feature.
"""

import os
import random

import pandas as pd

from defend.constraint_drift import compute_constraint_drift, compute_ingestion_source_trust_score
from defend.divergence import score_sessions
from generate.generated_sessions import load_cached_dataset

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TRANSACTION_PATH = os.path.join(_DATA_DIR, "ieee-fraud-detection", "train_transaction.csv")
IDENTITY_PATH = os.path.join(_DATA_DIR, "ieee-fraud-detection", "train_identity.csv")
OUTPUT_PATH = os.path.join(_DATA_DIR, "joined_sessions.csv")

_JOIN_SEED = 42  # fixed for reproducibility — a clean clone gets the same join


def _load_ieee_cis() -> pd.DataFrame:
    transactions = pd.read_csv(TRANSACTION_PATH)
    identity = pd.read_csv(IDENTITY_PATH)
    # Left join: only ~24% of IEEE-CIS transactions have identity rows —
    # that's the real dataset's shape, not a bug. NaNs for the rest are
    # expected; LightGBM (Task 5) handles them natively.
    return transactions.merge(identity, on="TransactionID", how="left")


def _sample_legitimate_transactions(transactions: pd.DataFrame, n: int) -> pd.DataFrame:
    legitimate = transactions[transactions["isFraud"] == 0]
    rng = random.Random(_JOIN_SEED)
    sample_idx = rng.sample(list(legitimate.index), n)
    return legitimate.loc[sample_idx].reset_index(drop=True)


def _relabel_benign_subtlety(subtlety: list[str]) -> list[str]:
    """"benign", not the generator's internal "n/a" sentinel: pandas'
    default read_csv NA-value list includes "n/a", so writing that
    literal string into a CSV column would silently turn every benign
    row's subtlety into a real NaN for anyone loading it normally
    (including Task 5's LightGBM step) — found by testing this exact
    load path, not a hypothetical. Pulled out as its own function so the
    fix has a regression test independent of the full join pipeline."""
    return ["benign" if t == "n/a" else t for t in subtlety]


def build_joined_dataset() -> pd.DataFrame:
    dataset = load_cached_dataset()
    if not dataset:
        raise RuntimeError(
            "No generated sessions found. Run `python3 -m generate.generated_sessions` first."
        )

    sessions = score_sessions([d["session"] for d in dataset])
    for s in sessions:
        s.constraint_drift = compute_constraint_drift(s)
        s.ingestion_source_trust_score = compute_ingestion_source_trust_score(s)
    subtlety = [d["subtlety"] for d in dataset]

    transactions = _load_ieee_cis()
    sampled = _sample_legitimate_transactions(transactions, len(sessions))
    # Keep IEEE-CIS's own fraud label for reference only, renamed so it's
    # never mistaken for our actual label (injection_present) downstream.
    sampled = sampled.rename(columns={"isFraud": "ieee_cis_isFraud"})

    subtlety = _relabel_benign_subtlety(subtlety)

    session_rows = pd.DataFrame([s.to_row() for s in sessions])
    session_rows["subtlety"] = subtlety  # outside to_row()'s locked contract, added here

    joined = pd.concat([session_rows, sampled], axis=1)
    return joined


if __name__ == "__main__":
    joined = build_joined_dataset()
    joined.to_csv(OUTPUT_PATH, index=False)
    print(f"Joined {len(joined)} sessions onto real IEEE-CIS transactions -> {OUTPUT_PATH}")
    print(f"Columns: {len(joined.columns)}")
