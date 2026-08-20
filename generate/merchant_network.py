"""
Synthetic shell-merchant network for attack #2 (transaction laundering
via fraudulent merchant network, TEAM_BRIEF.md Part 10 item #48).

Per TEAM_BRIEF.md Sec 2.1, attack #1 and #2 connect on purpose: "the
typosquat storefront the hijacked agent buys from IS a node in this
network." So the merchant nodes here aren't invented placeholder names —
they're the real domains already present in the 210 generated sessions
(generate/generated_sessions.py): the 212 unique malicious/typosquat
domains from hijacked sessions' task_origin_url/content_sources_ingested,
plus the 20 real legitimate domains from narrative_generator.py's
DOMAINS. What IS synthetic (necessarily — no real acquirer/beneficial-
owner registry exists for made-up domains): the Acquirer and Beneficiary
identities and the registration-age numbers, built around those real
domains per the "shared acquirer, shared beneficial owner,
rapid-registration" pattern.

Graph structure: Merchant --uses_acquirer--> Acquirer,
Merchant --owned_by--> Beneficiary. The label `is_fraud_ring_member` is
the node-classification target for defend/gnn.py.

Deliberately NOT a trivially-separable graph: an earlier version gave
every legitimate merchant its own unique Acquirer (never shared with
anyone), which meant "does this merchant's acquirer have degree > 1"
alone perfectly separated the two classes -- the GNN hit AUC=1.000, but
that validated nothing beyond "a real fraud ring is easier to spot than
an intentionally trivial synthetic graph," the same lesson Task 2's
hand-written-vs-generated hijack examples already taught this project.
Fixed to be realistic instead: legitimate merchants are drawn from a
small pool of shared Acquirers (real small businesses genuinely do share
a handful of popular payment processors) but never share a Beneficiary
(coincidental shared ownership isn't plausible) -- so shared-acquirer
alone is a weak, noisy signal for both classes. Larger rings split their
members across two Acquirers (spreading risk, as a real laundering
operation plausibly would) while keeping one shared Beneficiary --
shared beneficial ownership is the harder-to-fake signal that should
carry more of the real discriminating power, matching TEAM_BRIEF.md Part
10 item #48's own framing (shared acquirer *and* beneficial owner
*and* rapid registration together, not any one signal alone).
"""

import json
import os
import random
from urllib.parse import urlparse

import networkx as nx

from generate.generated_sessions import load_cached_dataset
from generate.narrative_generator import DOMAINS

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUTPUT_PATH = os.path.join(_DATA_DIR, "merchant_network.json")

_SEED = 7
_MIN_RING_SIZE, _MAX_RING_SIZE = 3, 15
_LEGIT_REG_AGE_RANGE = (500, 3000)   # days ago
_RING_REG_AGE_RANGE = (1, 60)        # days ago
_RING_WINDOW_DAYS = 14               # ring members registered within this many days of each other
_N_SHARED_LEGIT_ACQUIRERS = 7        # legit merchants drawn from this many popular processors
_SPLIT_RING_SIZE_THRESHOLD = 6       # rings larger than this split across 2 acquirers


def _extract_domain(url: str) -> str:
    netloc = urlparse(url).netloc or urlparse(f"//{url}").netloc
    return netloc.lower().removeprefix("www.")


def _real_legitimate_domains() -> list[str]:
    domains = set()
    for spec in DOMAINS.values():
        domains.update(spec["merchant_allowlist"])
    return sorted(domains)


def _real_malicious_domains() -> list[str]:
    """Domains from hijacked sessions' task_origin_url/content_sources_ingested,
    excluding the real legitimate domains from DOMAINS. A hijacked session
    sometimes references the genuine origin domain alongside the fake one
    in content_sources_ingested (e.g. the agent visited both) — without
    this exclusion, ~13 of the 19 legitimate merchants got miscounted as
    malicious, found by actually checking the set overlap rather than
    trusting the node count."""
    legit = set(_real_legitimate_domains())
    domains = set()
    for d in load_cached_dataset():
        session = d["session"]
        if not session.injection_present:
            continue
        for url in [session.task_origin_url] + session.content_sources_ingested:
            domain = _extract_domain(url)
            if domain and domain not in legit:
                domains.add(domain)
    return sorted(domains)


def _partition_into_rings(domains: list[str], rng: random.Random) -> list[list[str]]:
    rings, remaining = [], list(domains)
    rng.shuffle(remaining)
    while remaining:
        size = min(rng.randint(_MIN_RING_SIZE, _MAX_RING_SIZE), len(remaining))
        rings.append(remaining[:size])
        remaining = remaining[size:]
    return rings


def build_merchant_network() -> nx.Graph:
    rng = random.Random(_SEED)
    graph = nx.Graph()

    shared_acquirers = [f"Acquirer-shared-{i:03d}" for i in range(_N_SHARED_LEGIT_ACQUIRERS)]
    for a in shared_acquirers:
        graph.add_node(a, type="acquirer")

    for i, domain in enumerate(_real_legitimate_domains()):
        acquirer = rng.choice(shared_acquirers)  # coincidental sharing, like real payment processors
        beneficiary = f"Beneficiary-legit-{i:03d}"  # never shared -- ownership doesn't coincide
        reg_age = rng.randint(*_LEGIT_REG_AGE_RANGE)
        graph.add_node(domain, type="merchant", is_fraud_ring_member=False, registered_days_ago=reg_age)
        graph.add_node(beneficiary, type="beneficiary")
        graph.add_edge(domain, acquirer, relation="uses_acquirer")
        graph.add_edge(domain, beneficiary, relation="owned_by")

    rings = _partition_into_rings(_real_malicious_domains(), rng)
    for ring_id, members in enumerate(rings):
        beneficiary = f"Beneficiary-ring-{ring_id:03d}"  # the persistent, harder-to-fake ring signature
        graph.add_node(beneficiary, type="beneficiary")

        # Larger rings split across 2 acquirers (spreading risk) -- shared
        # acquirer alone is then a weak signal for both classes; shared
        # beneficiary is what actually ties the ring together.
        if len(members) > _SPLIT_RING_SIZE_THRESHOLD:
            acquirers = [f"Acquirer-ring-{ring_id:03d}-a", f"Acquirer-ring-{ring_id:03d}-b"]
        else:
            acquirers = [f"Acquirer-ring-{ring_id:03d}-a"]
        for a in acquirers:
            graph.add_node(a, type="acquirer")

        base_age = rng.randint(*_RING_REG_AGE_RANGE)
        for domain in members:
            acquirer = rng.choice(acquirers)
            reg_age = max(1, base_age + rng.randint(-_RING_WINDOW_DAYS, _RING_WINDOW_DAYS))
            graph.add_node(domain, type="merchant", is_fraud_ring_member=True, registered_days_ago=reg_age)
            graph.add_edge(domain, acquirer, relation="uses_acquirer")
            graph.add_edge(domain, beneficiary, relation="owned_by")

    return graph


if __name__ == "__main__":
    graph = build_merchant_network()
    merchants = [n for n, d in graph.nodes(data=True) if d.get("type") == "merchant"]
    fraud_count = sum(1 for n in merchants if graph.nodes[n]["is_fraud_ring_member"])

    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(nx.node_link_data(graph, edges="edges"), f, indent=2)

    print(f"Merchant network: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"  Merchants: {len(merchants)} ({fraud_count} fraud-ring, {len(merchants) - fraud_count} legitimate)")
    print(f"Saved -> {OUTPUT_PATH}")
