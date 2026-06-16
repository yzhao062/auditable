from auditable import CompoundReport, Report


def test_compound_preserves_per_stage_breakdown():
    reports = [
        Report("data", "d", 0.9),
        Report("model", "m", 0.1),
        Report("harness", "h", 0.0),
    ]
    cr = CompoundReport.of(reports)
    assert len(cr.reports) == 3
    assert set(cr.by_stage()) == {"data", "model", "harness"}


def test_compound_uncalibrated_score_is_max():
    cr = CompoundReport.of([Report("data", "d", 0.9), Report("model", "m", 0.1)])
    assert cr.uncalibrated_score == 0.9


def test_compound_score_can_be_omitted():
    cr = CompoundReport.of([Report("data", "d", 0.9)], score=False)
    assert cr.uncalibrated_score is None


def test_compound_drops_none_reports():
    cr = CompoundReport.of([Report("data", "d", 0.2), None])
    assert len(cr.reports) == 1


def test_compound_digest_changes_with_content():
    a = CompoundReport.of([Report("data", "d", 0.1)])
    b = CompoundReport.of([Report("data", "d", 0.2)])
    assert a.digest() != b.digest()
