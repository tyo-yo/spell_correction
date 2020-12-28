from spell_correction.t5_spell_corrector import T5SpellCorrector


def test_t5_spell_corrector():
    spell_corrector = T5SpellCorrector()
    corrected, changes = spell_corrector.fill_masked_text(
        "彼はファヴローに <extra_id_0>西部劇映画を提供した。", original_token="クラッシック"
    )
    assert isinstance(corrected, str)
    assert corrected == "彼はファヴローにクラシック西部劇映画を提供した。"
    assert len(changes) == 1
