from spec_decode.real import ngram_draft


def test_ngram_draft_copies_what_followed_last_match():
    ids = [1, 2, 3, 4, 5, 9, 2, 3]
    assert ngram_draft(ids, 2) == [4, 5]


def test_ngram_draft_empty_when_no_match():
    assert ngram_draft([1, 2, 3], 4) == []
