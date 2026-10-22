from maintenance_copilot.query import normalize_query


def test_query_rewrite_reports_expansions_and_typos():
    rewrite = normalize_query("R/H FWD BAFL SEEAL")

    assert rewrite.original_query == "R/H FWD BAFL SEEAL"
    assert rewrite.normalized_query == "right hand forward baffle seal"
    assert {(item.original, item.replacement) for item in rewrite.expansions_applied} == {
        ("r/h", "right hand"),
        ("fwd", "forward"),
    }
    assert {(item.original, item.replacement) for item in rewrite.typo_corrections_applied} == {
        ("bafl", "baffle"),
        ("seeal", "seal"),
    }


def test_query_rewrite_does_not_guess_unknown_shorthand():
    rewrite = normalize_query("XYZ actuator")

    assert rewrite.normalized_query == "xyz actuator"
    assert rewrite.expansions_applied == []
    assert rewrite.typo_corrections_applied == []
