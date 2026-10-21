"""Caption processing for cached Text Encoder outputs (ported from kohya-ss/sd-scripts #2455).

The deterministic part of ``process_caption`` (first line of a multi-line caption, prefix / suffix, secondary separator,
textual inversion replacements) must reach the captions the outputs are cached from, and ``process_caption`` picks the
line before adding prefix / suffix.
"""

import logging
from types import SimpleNamespace

from library.train_util import BaseDataset


def make_subset(**overrides):
    params = dict(
        enable_wildcard=False,
        enable_multiline_caption=False,
        multiline_caption_separator="---===---",
        caption_prefix=None,
        caption_suffix=None,
        caption_dropout_rate=0.0,
        caption_dropout_every_n_epochs=0,
        caption_tag_dropout_rate=0.0,
        shuffle_caption=False,
        token_warmup_min=1,
        token_warmup_step=0,
        keep_tokens=0,
        keep_tokens_separator="",
        caption_separator=",",
        secondary_separator=None,
    )
    params.update(overrides)
    return SimpleNamespace(**params)


def make_dataset(replacements=None):
    return SimpleNamespace(
        current_epoch=0, current_step=0, max_train_steps=100, replacements=replacements or {}, has_multiple_caption_candidates=False
    )


def for_cache(subset, caption, replacements=None):
    return BaseDataset.process_caption_for_cache(make_dataset(replacements), subset, caption)


def test_for_cache_applies_prefix_suffix_and_the_first_line():
    assert for_cache(make_subset(caption_prefix="pre", caption_suffix="suf"), "line1\nline2") == "pre line1 suf"


def test_for_cache_applies_secondary_separator_and_replacements():
    subset = make_subset(secondary_separator=";;;")
    assert for_cache(subset, "a;;;b, girl", replacements={"girl": "woman"}) == "a,b, woman"


def test_for_cache_ignores_the_random_options():
    # dropout / shuffle / tag dropout are rejected by is_text_encoder_output_cacheable anyway
    subset = make_subset(caption_dropout_rate=1.0, shuffle_caption=True, caption_tag_dropout_rate=1.0, caption_prefix="pre")
    assert for_cache(subset, "a, b, c") == "pre a, b, c"


def test_for_cache_keeps_a_wildcard_caption_as_it_is():
    assert for_cache(make_subset(enable_wildcard=True, caption_prefix="pre"), "a {x|y}\nb") == "pre a {x|y}\nb"


def test_for_cache_keeps_a_multiline_caption_whole():
    # enable_multiline_caption: the whole caption (normalized), as process_caption trains on it
    subset = make_subset(enable_multiline_caption=True, caption_prefix="pre", caption_suffix="suf")
    assert for_cache(subset, "line1\nline2\n\n") == "pre line1\nline2 suf"
    assert for_cache(subset, "line1\nline2\n\n") == BaseDataset.process_caption(make_dataset(), subset, "line1\nline2\n\n")


def test_for_cache_matches_process_caption_for_plain_captions():
    subset = make_subset(caption_prefix="pre", caption_suffix="suf", secondary_separator=";;;")
    for caption in ("a;;;b, c", "first line\nsecond line"):
        assert BaseDataset.process_caption_for_cache(make_dataset(), subset, caption) == BaseDataset.process_caption(
            make_dataset(), subset, caption
        )


def test_process_caption_puts_the_suffix_on_the_first_line():
    # the suffix was appended to the last line and then cut with it
    assert BaseDataset.process_caption(make_dataset(), make_subset(caption_suffix="suf"), "line1\nline2") == "line1 suf"


def test_wildcard_warns_but_stays_cacheable(caplog):
    dataset = make_dataset()
    dataset.subsets = [make_subset(enable_wildcard=True)]
    with caplog.at_level(logging.WARNING):
        assert BaseDataset.is_text_encoder_output_cacheable(dataset) is True
    assert any("enable_wildcard" in record.getMessage() for record in caplog.records)
    dataset.subsets = [make_subset()]
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert BaseDataset.is_text_encoder_output_cacheable(dataset) is True
    assert not any("enable_wildcard" in record.getMessage() for record in caplog.records)
