from spell_correction.metrics import LevenshteinDistance


def test_levenshtein_distance():
    metric = LevenshteinDistance(avg_by_seq_len=False)

    pred, gold = ["cdte"], ["cat"]  # distance: 2

    metric(pred, gold)
    val = metric.get_metric()
    assert val == 2

    pred, gold = ["ほge", "fgdx"], ["hoge", "fuga"]  # distance: 2, 3
    metric(pred, gold)
    val = metric.get_metric(reset=True)
    assert val == (2 + 2 + 3) / 3

    pred, gold = (
        ["appl", "apple", "apple"],
        ["apple", "apple", "apple"],
    )  # distance: 1, 0, 0
    metric(pred, gold)
    val = metric.get_metric()
    assert val == (1 + 0 + 0) / 3

