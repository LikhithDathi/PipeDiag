from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from experiments.full_system_validation import (
    HYPOTHESIS_IDS,
    SingleFaultEvidenceProvider,
    build_diagnostic_engine,
    build_single_fault_scenarios,
)
from pipediag.diagnostics.evidence_group import EvidenceGroup


@dataclass(frozen=True)
class MissingEvidenceValidation:
    complete_posterior: dict[str, float]
    missing_posterior: dict[str, float]
    no_evidence_posterior: dict[str, float]
    complete_uncertainty: float
    missing_uncertainty: float
    no_evidence_uncertainty: float

    @property
    def prior(self) -> dict[str, float]:
        return uniform_prior()

    @property
    def missing_h3_support(self) -> float:
        return self.missing_posterior["H3"]

    @property
    def prior_h3_support(self) -> float:
        return self.prior["H3"]

    @property
    def missing_evidence_preserves_h3_support(self) -> bool:
        return self.missing_h3_support >= self.prior_h3_support

    @property
    def no_evidence_matches_prior(self) -> bool:
        return all(
            self.no_evidence_posterior[hypothesis]
            == self.prior[hypothesis]
            for hypothesis in HYPOTHESIS_IDS
        )

    @property
    def no_evidence_has_maximum_uncertainty(self) -> bool:
        return self.no_evidence_uncertainty == 1.0

    @property
    def passed(self) -> bool:
        return (
            self.missing_evidence_preserves_h3_support
            and self.no_evidence_matches_prior
            and self.no_evidence_has_maximum_uncertainty
        )


def _collect_evidence_for_h3():
    scenarios = build_single_fault_scenarios(random_state=42)

    h3_scenario = next(
        scenario
        for scenario in scenarios
        if scenario.expected_hypothesis == "H3"
    )

    provider = SingleFaultEvidenceProvider()

    return provider.collect(h3_scenario)


def _remove_evidence_group(
    evidence_groups: Iterable[EvidenceGroup],
    group_name: str,
) -> list[EvidenceGroup]:
    return [
        evidence
        for evidence in evidence_groups
        if evidence.name != group_name
    ]


def _remove_evidence_groups(
    evidence_groups: Iterable[EvidenceGroup],
    group_names: set[str],
) -> list[EvidenceGroup]:
    return [
        evidence
        for evidence in evidence_groups
        if evidence.name not in group_names
    ]


def run_missing_evidence_validation() -> MissingEvidenceValidation:
    engine = build_diagnostic_engine()

    complete_evidence = _collect_evidence_for_h3()

    complete_result = engine.diagnose(
        evidence_groups=complete_evidence
    )

    missing_g3_evidence = _remove_evidence_group(
        complete_evidence,
        "G3",
    )

    missing_result = engine.diagnose(
        evidence_groups=missing_g3_evidence
    )

    no_evidence_result = engine.diagnose(
        evidence_groups=[]
    )

    return MissingEvidenceValidation(
        complete_posterior=complete_result.posterior,
        missing_posterior=missing_result.posterior,
        no_evidence_posterior=no_evidence_result.posterior,
        complete_uncertainty=complete_result.uncertainty,
        missing_uncertainty=missing_result.uncertainty,
        no_evidence_uncertainty=no_evidence_result.uncertainty,
    )


def run_multiple_missing_evidence_validation() -> MissingEvidenceValidation:
    engine = build_diagnostic_engine()

    complete_evidence = _collect_evidence_for_h3()

    complete_result = engine.diagnose(
        evidence_groups=complete_evidence
    )

    reduced_evidence = _remove_evidence_groups(
        complete_evidence,
        {"G3", "G4", "G5", "G7"},
    )

    reduced_result = engine.diagnose(
        evidence_groups=reduced_evidence
    )

    no_evidence_result = engine.diagnose(
        evidence_groups=[]
    )

    return MissingEvidenceValidation(
        complete_posterior=complete_result.posterior,
        missing_posterior=reduced_result.posterior,
        no_evidence_posterior=no_evidence_result.posterior,
        complete_uncertainty=complete_result.uncertainty,
        missing_uncertainty=reduced_result.uncertainty,
        no_evidence_uncertainty=no_evidence_result.uncertainty,
    )


def uniform_prior() -> dict[str, float]:
    probability = 1.0 / len(HYPOTHESIS_IDS)

    return {
        hypothesis: probability
        for hypothesis in HYPOTHESIS_IDS
    }