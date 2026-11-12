"""Material role and parallel-mode ambiguity; never modifies measured geometry."""


def classify_edge(entry, selected, own_best, own_alternatives):
    witnesses = sum(
        bool(row.get("passed") and row.get("outside_plateau"))
        for row in selected.get("holdout_evidence", [])
    )
    own_id = own_best["mode_id"] if own_best else None
    context_sensitive = own_id != selected["mode_id"]
    alternatives = [
        m["mode_id"] for m in own_alternatives if m["mode_id"] != selected["mode_id"]
    ]
    if own_best and own_best["mode_id"] != selected["mode_id"]:
        alternatives.append(own_best["mode_id"])
    return dict(
        state=(
            "REVIEW_AMBIGUOUS"
            if context_sensitive or alternatives
            else "REVIEW_ESTIMATED"
        ),
        own_only_mode_id=own_id,
        own_only_alternative_mode_ids=sorted(set(alternatives)),
        neighbor_context_changes_selection=context_sensitive,
        outside_platform_witness_sections=witnesses,
        material_role="NOT_IDENTIFIED_AS_STEEL",
        platform_absence_is_not_veto=True,
        selected_is_provisional=True,
        external_truth_verified=False,
        steel_certified=False,
        control_eligible=False,
    )
