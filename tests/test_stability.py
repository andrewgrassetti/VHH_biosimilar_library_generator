"""Tests for vhh_library.stability – StabilityScorer class."""

from __future__ import annotations

from unittest.mock import MagicMock

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


class TestCalibrationHelpers:
    def test_pll_to_predicted_tm(self) -> None:
        # PLL = -120.0, seq_len = 120 → per_residue = -1.0
        # Tm = 12.5 * (-1.0) + 95.0 = 82.5
        tm = _pll_to_predicted_tm(-120.0, 120)
        assert tm == pytest.approx(82.5)

    def test_pll_to_predicted_tm_zero_length(self) -> None:
        # seq_len=0 should not crash (uses max(seq_len, 1))
        tm = _pll_to_predicted_tm(-10.0, 0)
        assert isinstance(tm, float)

    def test_sigmoid_normalize_midpoint(self) -> None:
        # Midpoint of 55..80 is 67.5; sigmoid at midpoint → ≈0.5
        val = _sigmoid_normalize(67.5, 55.0, 80.0)
        assert val == pytest.approx(0.5, abs=0.01)

    def test_sigmoid_normalize_high(self) -> None:
        # Well above tm_max → ≈1.0
        val = _sigmoid_normalize(120.0, 55.0, 80.0)
        assert val == pytest.approx(1.0, abs=0.01)

    def test_sigmoid_normalize_low(self) -> None:
        # Well below tm_min → ≈0.0
        val = _sigmoid_normalize(0.0, 55.0, 80.0)
        assert val == pytest.approx(0.0, abs=0.01)


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


class TestESM2Scoring:
    def test_predicted_tm_present_with_esm2(self, vhh: VHHSequence) -> None:
        mock_esm = MagicMock()
        mock_esm.score_single.return_value = -100.0
        scorer = StabilityScorer(esm_scorer=mock_esm)
        result = scorer.score(vhh)
        assert isinstance(result["predicted_tm"], float)
        assert "tm_score" in result
        assert result["scoring_method"] == "esm2"

    def test_composite_score_range_with_esm2(self, vhh: VHHSequence) -> None:
        mock_esm = MagicMock()
        mock_esm.score_single.return_value = -100.0
        scorer = StabilityScorer(esm_scorer=mock_esm)
        result = scorer.score(vhh)
        assert 0.0 <= result["composite_score"] <= 1.0

    def test_disulfide_penalty_lowers_score(self) -> None:
        """A construct missing the canonical disulfide should score at least 0.15 lower."""
        mock_esm = MagicMock()
        mock_esm.score_single.return_value = -100.0

        # Normal VHH with both Cys at positions 23 and 104
        vhh_with_cys = VHHSequence(SAMPLE_VHH)
        scorer = StabilityScorer(esm_scorer=mock_esm)
        score_with = scorer.score(vhh_with_cys)

        # Create a mutant that removes both Cys at canonical positions
        # Replace C→A at IMGT position 23 and 104
        mutant = VHHSequence.mutate(vhh_with_cys, 23, "A")
        mutant = VHHSequence.mutate(mutant, 104, "A")
        score_without = scorer.score(mutant)

        assert score_with["composite_score"] - score_without["composite_score"] >= 0.15


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
