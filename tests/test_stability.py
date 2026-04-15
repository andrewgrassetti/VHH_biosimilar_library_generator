"""Tests for vhh_library.stability – StabilityScorer class."""

from __future__ import annotations

import pytest

from vhh_library.sequence import VHHSequence
from vhh_library.stability import (
    StabilityScorer,
    _esm2_pll_available,
    _pll_to_predicted_tm,
    _sigmoid_normalize,
)

SAMPLE_VHH = (
    "QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISW"
    "SGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAAGVRAEWDYWGQGTLVTVSS"
)


@pytest.fixture
def scorer() -> StabilityScorer:
    return StabilityScorer()


@pytest.fixture
def vhh() -> VHHSequence:
    return VHHSequence(SAMPLE_VHH)


class TestScoring:
    def test_score_returns_dict(self, scorer: StabilityScorer, vhh: VHHSequence) -> None:
        result = scorer.score(vhh)
        assert "composite_score" in result

    def test_composite_score_range(self, scorer: StabilityScorer, vhh: VHHSequence) -> None:
        result = scorer.score(vhh)
        assert 0.0 <= result["composite_score"] <= 1.0

    def test_pI_range(self, scorer: StabilityScorer, vhh: VHHSequence) -> None:
        result = scorer.score(vhh)
        assert 3.0 <= result["pI"] <= 12.0

    def test_disulfide_score(self, scorer: StabilityScorer, vhh: VHHSequence) -> None:
        result = scorer.score(vhh)
        assert 0.0 <= result["disulfide_score"] <= 1.0

    def test_esm2_result_includes_predicted_tm_and_tm_score(self, vhh: VHHSequence) -> None:
        class _StubESMScorer:
            @staticmethod
            def score_single(sequence: str) -> float:
                return -120.0

        scorer = StabilityScorer(esm_scorer=_StubESMScorer())
        result = scorer.score(vhh)

        assert result["scoring_method"] == "esm2"
        assert "predicted_tm" in result
        assert "tm_score" in result
        assert 0.0 <= result["tm_score"] <= 1.0

    def test_esm2_disulfide_penalty_lowers_composite_score(self, vhh: VHHSequence) -> None:
        class _StubESMScorer:
            @staticmethod
            def score_single(sequence: str) -> float:
                return -120.0

        scorer = StabilityScorer(esm_scorer=_StubESMScorer())
        missing_disulfide = VHHSequence.mutate(VHHSequence.mutate(vhh, 23, "A"), 104, "A")

        intact_result = scorer.score(vhh)
        missing_result = scorer.score(missing_disulfide)

        assert intact_result["disulfide_score"] == 1.0
        assert missing_result["disulfide_score"] == 0.0
        assert missing_result["composite_score"] < intact_result["composite_score"]


class TestMutationEffect:
    def test_predict_mutation_effect(self, scorer: StabilityScorer, vhh: VHHSequence) -> None:
        delta = scorer.predict_mutation_effect(vhh, 1, "A")
        assert isinstance(delta, float)


class TestScoringMethod:
    def test_scoring_method_present(self, scorer: StabilityScorer, vhh: VHHSequence) -> None:
        result = scorer.score(vhh)
        assert result["scoring_method"] in ("legacy", "esm2")

    def test_legacy_fallback(self, vhh: VHHSequence) -> None:
        scorer = StabilityScorer()
        result = scorer.score(vhh)
        assert result["scoring_method"] == "legacy"


class TestAvailability:
    def test_esm2_pll_available_returns_bool(self) -> None:
        assert isinstance(_esm2_pll_available(), bool)


class TestCalibrationHelpers:
    def test_pll_to_predicted_tm_helper(self) -> None:
        assert _pll_to_predicted_tm(-120.0, 120) == pytest.approx(82.5)

    def test_sigmoid_normalize_helper(self) -> None:
        low = _sigmoid_normalize(40.0, 55.0, 80.0)
        mid = _sigmoid_normalize(67.5, 55.0, 80.0)
        high = _sigmoid_normalize(90.0, 55.0, 80.0)

        assert 0.0 < low < mid < high < 1.0
        assert mid == pytest.approx(0.5, abs=1e-9)
