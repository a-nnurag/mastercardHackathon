"""
Defend layer 1 — hard mandate-scope rules, no ML.

Per TEAM_BRIEF.md's Defend stack (Sec 4.6): the first, cheapest layer a
risk analyst sees, before LightGBM (Task 5). Every flag comes with a
concrete, named reason a human could verify by hand — no scores, no
model, just facts about the session.
"""

from typing import Optional

from defend.contracts import SignalResult
from defend.detection_context import DetectionContext
from defend.signals.base import SignalDetector
from generate.session_schema import AgentSession
from defend.constraint_drift import extract_amount_inr, extract_domain


def apply_rules(session: AgentSession) -> tuple[bool, list[str]]:
    """Returns (flagged, reasons). Empty reasons list means no violation
    of any checked rule — not proof of legitimacy, just that these
    specific hard checks found nothing."""
    reasons = []

    amount = extract_amount_inr(session.signed_artifact_text)
    if amount is not None and amount > session.mandate_scope.amount_cap:
        reasons.append(
            f"amount cap breached: signed INR {amount:,.0f} > cap INR {session.mandate_scope.amount_cap:,.0f}"
        )

    origin_domain = extract_domain(session.task_origin_url)
    allowlist = {extract_domain(d) for d in session.mandate_scope.merchant_allowlist}
    if origin_domain not in allowlist:
        reasons.append(f"task_origin_url domain '{origin_domain}' not in merchant allowlist {sorted(allowlist)}")

    if session.agent_registry_status != "valid":
        reasons.append(f"agent_registry_status is '{session.agent_registry_status}', not 'valid'")

    return bool(reasons), reasons


class RuleDetector(SignalDetector):
    name: str = "rules"
    supported_attack_families: tuple[str, ...] = ("prompt_injection",)

    def detect(
        self,
        session: AgentSession,
        context: Optional[DetectionContext] = None,
    ) -> SignalResult:
        flagged, reasons = apply_rules(session)
        return SignalResult(
            name=self.name,
            value=flagged,
            available=True,
            evidence=reasons,
            hard_violation=flagged,
            metadata={
                "violations_count": len(reasons),
                "reasons": reasons,
            },
        )


if __name__ == "__main__":
    from generate.generated_sessions import load_cached_dataset

    dataset = load_cached_dataset()
    sessions = [d["session"] for d in dataset]
    subtlety = [d["subtlety"] for d in dataset]

    flags = [apply_rules(s) for s in sessions]

    print(f"Evaluated {len(sessions)} sessions")
    print(f"Flagged: {sum(f for f, _ in flags)}/{len(flags)}")

    for label in ("n/a", "obvious", "subtle"):
        idx = [i for i, t in enumerate(subtlety) if t == label]
        if not idx:
            continue
        flagged_count = sum(flags[i][0] for i in idx)
        display_label = "benign" if label == "n/a" else label
        print(f"  {display_label}: {flagged_count}/{len(idx)} flagged")

    obvious_idx = [i for i, t in enumerate(subtlety) if t == "obvious"]
    if obvious_idx:
        recall = sum(flags[i][0] for i in obvious_idx) / len(obvious_idx)
        print(f"\nRecall on hijacked-obvious subset: {recall:.3f}")
        if recall == 0.0:
            print("READ: zero recall on the easiest cases — something's wired wrong, per plan.md's own check.")

    print("\nSample reasons (first 3 flagged sessions):")
    shown = 0
    for i, (flagged, reasons) in enumerate(flags):
        if flagged and shown < 3:
            print(f"  {sessions[i].agent_id} ({subtlety[i]}): {reasons}")
            shown += 1
