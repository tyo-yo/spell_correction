from spell_correction.metrics import LevenshteinDistance


def test_levenshtein_distance():
    metric = LevenshteinDistance(avg_by_seq_len=False)

    pred, gold = ["cdte"], ["cat"]  # distance: 2

    metric(pred, gold)
    val = metric.get_metric()
    assert val["LevenshteinDistance"] == 2
    assert val["ExactMatch"] == 0

    pred, gold = ["ほge", "fgdx"], ["hoge", "fuga"]  # distance: 2, 3
    metric(pred, gold)
    val = metric.get_metric(reset=True)
    assert val["LevenshteinDistance"] == (2 + 2 + 3) / 3
    assert val["ExactMatch"] == 0

    pred, gold = (
        ["ap", "apple", "apple"],
        ["apple", "apple", "apple"],
    )  # distance: 1, 0, 0
    metric(pred, gold)
    val = metric.get_metric()
    assert val["LevenshteinDistance"] == (3 + 0 + 0) / 3
    assert val["ExactMatch"] == (2 + 0 + 0) / 3

