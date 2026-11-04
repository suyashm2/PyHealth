"""Next-visit diagnosis prediction tasks for MIMIC-III and MIMIC-IV.

Given the diagnosis codes from a patient's earlier hospital visits, predict the
set of diagnosis codes recorded at the patient's next visit. This is the task
used by Doctor AI (Choi et al., MLHC 2016) and by SHy (Yu et al., CHIL 2025),
which report Recall@k and nDCG@k on it.
"""

from typing import Any, ClassVar

import polars as pl

from .base_task import BaseTask


def _clean_codes(codes: list[Any]) -> list[str]:
    """Removes missing and repeated codes while keeping their original order.

    Args:
        codes: Raw diagnosis codes for one visit.

    Returns:
        The codes as stripped strings, each kept once.
    """
    cleaned_codes: list[str] = []
    seen_codes = set()
    for code in codes:
        if code is None:
            continue
        code = str(code).strip()
        if code == "" or code in seen_codes:
            continue
        cleaned_codes.append(code)
        seen_codes.add(code)
    return cleaned_codes


def _build_next_visit_samples(
    patient_id: str,
    visit_ids: list[str],
    visit_codes: list[list[str]],
    sample_per_visit: bool,
) -> list[dict[str, Any]]:
    """Turns one patient's visits into next-visit diagnosis samples.

    Args:
        patient_id: The patient's ID.
        visit_ids: Admission IDs of the patient's visits, in time order.
        visit_codes: Diagnosis codes of each visit, in the same order.
        sample_per_visit: If False, make one sample per patient: all visits
            except the last are the input and the last visit is the target, as
            in the SHy paper. If True, make one sample for every visit after
            the first, each using all earlier visits as its input.

    Returns:
        A list of samples. Each sample holds the patient ID, the target visit's
        ID, the earlier visits' codes ("conditions", a list of visits), and the
        target visit's codes ("next_conditions").
    """
    if len(visit_codes) < 2:
        return []

    if sample_per_visit:
        target_positions = range(1, len(visit_codes))
    else:
        target_positions = [len(visit_codes) - 1]

    samples = []
    for target_position in target_positions:
        samples.append(
            {
                "patient_id": patient_id,
                "visit_id": visit_ids[target_position],
                "conditions": visit_codes[:target_position],
                "next_conditions": visit_codes[target_position],
            }
        )
    return samples


class NextVisitDiagnosisMIMIC3(BaseTask):
    """Next-visit diagnosis prediction on MIMIC-III.

    Each sample uses the ICD-9 diagnosis codes of a patient's earlier visits to
    predict the ICD-9 diagnosis codes of a later visit (a multilabel target).
    Visits without diagnosis codes are skipped, and patients need at least two
    visits with codes. Codes keep MIMIC's format without a dot (for example,
    "4280"); ``pyhealth.medcode.codes.icd9cm.ICD9CM.standardize`` converts them
    to the dotted form used by the ICD-9 hierarchy.

    Attributes:
        task_name (str): The name of the task.
        input_schema (dict[str, str]): The input field "conditions": the
            diagnosis codes of each earlier visit (a list of visits, each a
            list of codes).
        output_schema (dict[str, str]): The output field "next_conditions":
            the diagnosis codes of the target visit.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import NextVisitDiagnosisMIMIC3
        >>> dataset = MIMIC3Dataset(
        ...     root="/path/to/mimic-iii/1.4",
        ...     tables=["DIAGNOSES_ICD"],
        ... )
        >>> task = NextVisitDiagnosisMIMIC3()
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "NextVisitDiagnosisMIMIC3"
    input_schema: ClassVar[dict[str, str]] = {"conditions": "nested_sequence"}
    output_schema: ClassVar[dict[str, str]] = {"next_conditions": "multilabel"}

    def __init__(self, sample_per_visit: bool = False, **kwargs: Any) -> None:
        """Initializes the task.

        Args:
            sample_per_visit: If False (the default), make one sample per
                patient, with the last visit as the target, as in the SHy
                paper. If True, make one sample for every visit after the
                first.
            **kwargs: Passed on to ``BaseTask``.
        """
        super().__init__(**kwargs)
        self.sample_per_visit = sample_per_visit

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Creates next-visit diagnosis samples for one patient.

        Args:
            patient: A patient with ``admissions`` and ``diagnoses_icd`` events.

        Returns:
            The patient's samples, or an empty list if the patient has fewer
            than two visits with diagnosis codes.
        """
        admissions = patient.get_events(event_type="admissions")
        if len(admissions) < 2:
            return []

        visit_ids = []
        visit_codes = []
        for admission in admissions:
            diagnoses = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
                return_df=True,
            )
            raw_codes = (
                diagnoses.select(pl.col("diagnoses_icd/icd9_code"))
                .to_series()
                .to_list()
            )
            codes = _clean_codes(raw_codes)
            if len(codes) == 0:
                continue
            visit_ids.append(admission.hadm_id)
            visit_codes.append(codes)

        return _build_next_visit_samples(
            patient.patient_id, visit_ids, visit_codes, self.sample_per_visit
        )


class NextVisitDiagnosisMIMIC4(BaseTask):
    """Next-visit diagnosis prediction on MIMIC-IV, using ICD-9 codes only.

    MIMIC-IV records diagnoses in ICD-9 before October 2015 and in ICD-10
    after. This task keeps only the ICD-9 codes, so that models that use the
    ICD-9 hierarchy (such as SHy) can be applied. Visits left without ICD-9
    codes are skipped, and patients need at least two visits with ICD-9 codes.
    Codes keep MIMIC's format without a dot (for example, "4280").

    Attributes:
        task_name (str): The name of the task.
        input_schema (dict[str, str]): The input field "conditions": the
            ICD-9 diagnosis codes of each earlier visit (a list of visits,
            each a list of codes).
        output_schema (dict[str, str]): The output field "next_conditions":
            the ICD-9 diagnosis codes of the target visit.

    Examples:
        >>> from pyhealth.datasets import MIMIC4EHRDataset
        >>> from pyhealth.tasks import NextVisitDiagnosisMIMIC4
        >>> dataset = MIMIC4EHRDataset(
        ...     root="/path/to/mimic-iv/3.1",
        ...     tables=["diagnoses_icd"],
        ... )
        >>> task = NextVisitDiagnosisMIMIC4()
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "NextVisitDiagnosisMIMIC4"
    input_schema: ClassVar[dict[str, str]] = {"conditions": "nested_sequence"}
    output_schema: ClassVar[dict[str, str]] = {"next_conditions": "multilabel"}

    def __init__(self, sample_per_visit: bool = False, **kwargs: Any) -> None:
        """Initializes the task.

        Args:
            sample_per_visit: If False (the default), make one sample per
                patient, with the last visit as the target, as in the SHy
                paper. If True, make one sample for every visit after the
                first.
            **kwargs: Passed on to ``BaseTask``.
        """
        super().__init__(**kwargs)
        self.sample_per_visit = sample_per_visit

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Creates next-visit diagnosis samples for one patient.

        Args:
            patient: A patient with ``admissions`` and ``diagnoses_icd`` events.

        Returns:
            The patient's samples, or an empty list if the patient has fewer
            than two visits with ICD-9 diagnosis codes.
        """
        admissions = patient.get_events(event_type="admissions")
        if len(admissions) < 2:
            return []

        visit_ids = []
        visit_codes = []
        for admission in admissions:
            diagnoses = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
                return_df=True,
            )
            icd9_diagnoses = diagnoses.filter(
                pl.col("diagnoses_icd/icd_version").cast(pl.Utf8) == "9"
            )
            raw_codes = (
                icd9_diagnoses.select(pl.col("diagnoses_icd/icd_code"))
                .to_series()
                .to_list()
            )
            codes = _clean_codes(raw_codes)
            if len(codes) == 0:
                continue
            visit_ids.append(admission.hadm_id)
            visit_codes.append(codes)

        return _build_next_visit_samples(
            patient.patient_id, visit_ids, visit_codes, self.sample_per_visit
        )
