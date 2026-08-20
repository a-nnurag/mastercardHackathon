import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate.merchant_network import build_merchant_network, _real_legitimate_domains, _real_malicious_domains


def test_legitimate_domains_not_counted_as_malicious():
    # Regression test: hijacked sessions sometimes reference the real
    # legitimate domain alongside the fake one in content_sources_ingested,
    # which previously caused ~13/19 real merchants to be miscounted as
    # fraud-ring members.
    legit = set(_real_legitimate_domains())
    malicious = set(_real_malicious_domains())
    assert not (legit & malicious)
    assert len(legit) == 19


def test_ring_members_share_acquirer_and_beneficiary():
    graph = build_merchant_network()
    fraud_merchants = [n for n, d in graph.nodes(data=True) if d.get("type") == "merchant" and d["is_fraud_ring_member"]]
    assert fraud_merchants

    sample = fraud_merchants[0]
    neighbors = set(graph.neighbors(sample))
    acquirer = next(n for n in neighbors if graph.nodes[n]["type"] == "acquirer")
    beneficiary = next(n for n in neighbors if graph.nodes[n]["type"] == "beneficiary")

    ring_mate = next(
        n for n in graph.neighbors(acquirer)
        if n != sample and graph.nodes[n]["type"] == "merchant"
    )
    assert beneficiary in graph.neighbors(ring_mate)


def test_legitimate_merchants_never_share_a_beneficiary():
    # Legit merchants DO coincidentally share acquirers (a small pool of
    # popular payment processors, like real small businesses) -- that's
    # deliberate, not a bug. What must never coincide is beneficial
    # ownership: shared beneficiary is the harder-to-fake ring signature.
    graph = build_merchant_network()
    legit_merchants = [n for n, d in graph.nodes(data=True) if d.get("type") == "merchant" and not d["is_fraud_ring_member"]]
    assert len(legit_merchants) == 19

    for merchant in legit_merchants:
        beneficiary = next(n for n in graph.neighbors(merchant) if graph.nodes[n]["type"] == "beneficiary")
        other_merchants_on_beneficiary = [n for n in graph.neighbors(beneficiary) if n != merchant]
        assert other_merchants_on_beneficiary == []


def test_legitimate_acquirers_are_shared_across_a_small_pool():
    # The realism fix: legit merchants draw from a small shared-acquirer
    # pool rather than each getting a unique one (an earlier version made
    # "shared acquirer" a perfect fraud tell, which isn't realistic).
    graph = build_merchant_network()
    legit_acquirers = {
        n for n, d in graph.nodes(data=True) if d.get("type") == "acquirer" and n.startswith("Acquirer-shared-")
    }
    assert 1 < len(legit_acquirers) < 19  # shared pool, smaller than the merchant count


def test_ring_registration_ages_are_recent_and_clustered():
    graph = build_merchant_network()
    fraud_ages = [d["registered_days_ago"] for n, d in graph.nodes(data=True) if d.get("type") == "merchant" and d["is_fraud_ring_member"]]
    legit_ages = [d["registered_days_ago"] for n, d in graph.nodes(data=True) if d.get("type") == "merchant" and not d["is_fraud_ring_member"]]

    assert max(fraud_ages) <= 74  # 60-day range + 14-day window
    assert min(legit_ages) >= 500
