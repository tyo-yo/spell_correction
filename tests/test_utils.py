from spell_correction.utils import BeginningLevenshtein


def test_beginning_edit_distance_00():
    begginng_levenshtein = BeginningLevenshtein()
    assert begginng_levenshtein.distance("", "sapporo") == 0
    assert begginng_levenshtein.distance("s", "sapporo") == 0
    assert begginng_levenshtein.distance("d", "sapporo") == 1
    assert begginng_levenshtein.distance("a", "sapporo") == 1
    assert begginng_levenshtein.distance("sd", "sapporo") == 1
    assert begginng_levenshtein.distance("sa", "sapporo") == 0
    assert begginng_levenshtein.distance("sta", "sapporo") == 1


def test_beginning_edit_distance_01():
    begginng_levenshtein = BeginningLevenshtein()
    result = begginng_levenshtein("sta", "sapporo")
    assert set(result.keys()) == {"match_ref", "match_len", "distance"}
    assert result["match_ref"] == "sa"
    assert result["match_len"] == 2
    assert result["distance"] == 1


def test_beginning_edit_distance_02():
    begginng_levenshtein = BeginningLevenshtein()
    result = begginng_levenshtein("更", "更新")
    assert result["match_ref"] == "更"
    assert result["match_len"] == 1
    assert result["distance"] == 0
