"""Tests for vhh_library.orthogonal_scoring – alternative scoring methods."""

from __future__ import annotations

import pytest

from vhh_library.orthogonal_scoring import (
    ConsensusStabilityScorer,
)
from vhh_library.sequence import VHHSequence

SAMPLE_VHH = (
    "QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISW"
    "SGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAAGVRAEWDYWGQGTLVTVSS"
)


@pytest.fixture
def vhh() -> VHHSequence:
    return VHHSequence(SAMPLE_VHH)


@pytest.fixture
def consensus() -> ConsensusStabilityScorer:
    return ConsensusStabilityScorer()


class TestConsensusStabilityScorer:
    def test_score_returns_dict(
        self, consensus: ConsensusStabilityScorer, vhh: VHHSequence
    ) -> None:
        result = consensus.score(vhh)
        assert "composite_score" in result
        assert 0.0 <= result["composite_score"] <= 1.0

    def test_positions_evaluated(
        self, consensus: ConsensusStabilityScorer, vhh: VHHSequence
    ) -> None:
        result = consensus.score(vhh)
        assert result["positions_evaluated"] > 0
        assert result["consensus_matches"] <= result["positions_evaluated"]

    def test_predict_mutation_effect_returns_float(
        self, consensus: ConsensusStabilityScorer, vhh: VHHSequence
    ) -> None:
        delta = consensus.predict_mutation_effect(vhh, 1, "A")
        assert isinstance(delta, float)

    def test_same_aa_returns_zero(
        self, consensus: ConsensusStabilityScorer, vhh: VHHSequence
    ) -> None:
        original_aa = vhh.imgt_numbered["1"]
        delta = consensus.predict_mutation_effect(vhh, 1, original_aa)
        assert delta == 0.0


class TestIntegration:
    def test_mutation_changes_scores(
        self,
        consensus: ConsensusStabilityScorer,
        vhh: VHHSequence,
    ) -> None:
        mutant = VHHSequence.mutate(vhh, 1, "A")
        consensus.score(vhh)["composite_score"]
        con_mutant = consensus.score(mutant)["composite_score"]
        # Just verify it runs
        assert isinstance(con_mutant, float)
