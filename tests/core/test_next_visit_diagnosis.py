"""Tests for the next-visit diagnosis prediction tasks.

The tests use four small synthetic patients built in memory, so they need no
MIMIC files and run in well under a second.
"""

import unittest
from datetime import datetime

import polars as pl

from pyhealth.data import Patient
from pyhealth.datasets import create_sample_dataset
from pyhealth.tasks import NextVisitDiagnosisMIMIC3, NextVisitDiagnosisMIMIC4


def _make_patient(patient_id: str, event_rows: list[dict]) -> Patient:
    """Builds a PyHealth Patient from a list of event rows.

    Args:
        patient_id: The patient's ID.
        event_rows: One dictionary per event, using PyHealth's column names
            (``event_type``, ``timestamp``, and ``<event_type>/<attribute>``).

    Returns:
        A Patient whose events are sorted by time.
    """
    events = pl.DataFrame(event_rows, infer_schema_length=None)
    return Patient(patient_id=patient_id, data_source=events)


def _admission(hadm_id: str, admit_time: datetime) -> dict:
    """Returns an admissions event row."""
    return {
        "event_type": "admissions",
        "timestamp": admit_time,
        "admissions/hadm_id": hadm_id,
    }


def _mimic3_diagnosis(hadm_id: str, admit_time: datetime, code: str) -> dict:
    """Returns a MIMIC-III diagnoses_icd event row."""
    return {
        "event_type": "diagnoses_icd",
        "timestamp": admit_time,
        "diagnoses_icd/hadm_id": hadm_id,
        "diagnoses_icd/icd9_code": code,
    }


def _mimic4_diagnosis(
    hadm_id: str, admit_time: datetime, code: str, icd_version: int
) -> dict:
    """Returns a MIMIC-IV diagnoses_icd event row."""
    return {
        "event_type": "diagnoses_icd",
        "timestamp": admit_time,
        "diagnoses_icd/hadm_id": hadm_id,
        "diagnoses_icd/icd_code": code,
        "diagnoses_icd/icd_version": icd_version,
    }


JAN = datetime(2150, 1, 5)
MAR = datetime(2150, 3, 9)
JUN = datetime(2150, 6, 21)


def _three_visit_patient() -> Patient:
    """Patient P1: three visits, a repeated code, rows given out of time order.

    In time order the visits are {4280, 25000}, {V4581}, {4280, 41401}. The
    first visit lists 4280 twice, which should be kept once.
    """
    return _make_patient(
        "P1",
        [
            _admission("103", JUN),
            _mimic3_diagnosis("103", JUN, "4280"),
            _mimic3_diagnosis("103", JUN, "41401"),
            _admission("101", JAN),
            _mimic3_diagnosis("101", JAN, "4280"),
            _mimic3_diagnosis("101", JAN, "25000"),
            _mimic3_diagnosis("101", JAN, "4280"),
            _admission("102", MAR),
            _mimic3_diagnosis("102", MAR, "V4581"),
        ],
    )


def _one_visit_patient() -> Patient:
    """Patient P2: a single visit, so there is no next visit to predict."""
    return _make_patient(
        "P2",
        [
            _admission("201", JAN),
            _mimic3_diagnosis("201", JAN, "486"),
        ],
    )


def _empty_visit_patient() -> Patient:
    """Patient P3: three admissions, but the middle one has no diagnoses.

    The empty visit should be skipped, leaving {486} followed by {5849}.
    """
    return _make_patient(
        "P3",
        [
            _admission("301", JAN),
            _mimic3_diagnosis("301", JAN, "486"),
            _admission("302", MAR),
            _admission("303", JUN),
            _mimic3_diagnosis("303", JUN, "5849"),
        ],
    )


def _mixed_icd_patient() -> Patient:
    """Patient P4 (MIMIC-IV): ICD-9 and ICD-10 codes mixed.

    Visit 401 has only ICD-9 codes, visit 402 only ICD-10 codes (skipped), and
    visit 403 has both (only the ICD-9 code is kept).
    """
    return _make_patient(
        "P4",
        [
            _admission("401", JAN),
            _mimic4_diagnosis("401", JAN, "4280", 9),
            _mimic4_diagnosis("401", JAN, "25000", 9),
            _admission("402", MAR),
            _mimic4_diagnosis("402", MAR, "I509", 10),
            _admission("403", JUN),
            _mimic4_diagnosis("403", JUN, "41401", 9),
            _mimic4_diagnosis("403", JUN, "E119", 10),
        ],
    )


class TestNextVisitDiagnosisMIMIC3(unittest.TestCase):
    """Tests for NextVisitDiagnosisMIMIC3."""

    def test_task_schema(self):
        self.assertEqual(NextVisitDiagnosisMIMIC3.task_name, "NextVisitDiagnosisMIMIC3")
        self.assertEqual(
            NextVisitDiagnosisMIMIC3.input_schema, {"conditions": "nested_sequence"}
        )
        self.assertEqual(
            NextVisitDiagnosisMIMIC3.output_schema, {"next_conditions": "multilabel"}
        )

    def test_default_is_one_sample_per_patient(self):
        task = NextVisitDiagnosisMIMIC3()
        self.assertFalse(task.sample_per_visit)

        samples = task(_three_visit_patient())

        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["patient_id"], "P1")
        self.assertEqual(sample["visit_id"], "103")
        self.assertEqual(sample["conditions"], [["4280", "25000"], ["V4581"]])
        self.assertEqual(sample["next_conditions"], ["4280", "41401"])

    def test_sample_per_visit(self):
        task = NextVisitDiagnosisMIMIC3(sample_per_visit=True)

        samples = task(_three_visit_patient())

        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0]["visit_id"], "102")
        self.assertEqual(samples[0]["conditions"], [["4280", "25000"]])
        self.assertEqual(samples[0]["next_conditions"], ["V4581"])
        self.assertEqual(samples[1]["visit_id"], "103")
        self.assertEqual(samples[1]["conditions"], [["4280", "25000"], ["V4581"]])
        self.assertEqual(samples[1]["next_conditions"], ["4280", "41401"])

    def test_target_visit_never_in_input(self):
        task = NextVisitDiagnosisMIMIC3(sample_per_visit=True)

        for sample in task(_three_visit_patient()):
            self.assertNotIn(sample["next_conditions"], sample["conditions"])

    def test_single_visit_patient_is_skipped(self):
        task = NextVisitDiagnosisMIMIC3()

        self.assertEqual(task(_one_visit_patient()), [])

    def test_visit_without_codes_is_skipped(self):
        task = NextVisitDiagnosisMIMIC3()

        samples = task(_empty_visit_patient())

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["visit_id"], "303")
        self.assertEqual(samples[0]["conditions"], [["486"]])
        self.assertEqual(samples[0]["next_conditions"], ["5849"])

    def test_samples_fit_pyhealth_schema(self):
        task = NextVisitDiagnosisMIMIC3()
        samples = task(_three_visit_patient()) + task(_empty_visit_patient())

        sample_dataset = create_sample_dataset(
            samples=samples,
            input_schema=task.input_schema,
            output_schema=task.output_schema,
            dataset_name="next_visit_test",
        )

        self.assertEqual(len(sample_dataset), 2)
        first_sample = sample_dataset[0]
        self.assertIn("conditions", first_sample)
        self.assertIn("next_conditions", first_sample)


class TestNextVisitDiagnosisMIMIC4(unittest.TestCase):
    """Tests for NextVisitDiagnosisMIMIC4."""

    def test_task_schema(self):
        self.assertEqual(NextVisitDiagnosisMIMIC4.task_name, "NextVisitDiagnosisMIMIC4")
        self.assertEqual(
            NextVisitDiagnosisMIMIC4.input_schema, {"conditions": "nested_sequence"}
        )
        self.assertEqual(
            NextVisitDiagnosisMIMIC4.output_schema, {"next_conditions": "multilabel"}
        )

    def test_keeps_only_icd9_codes(self):
        task = NextVisitDiagnosisMIMIC4()

        samples = task(_mixed_icd_patient())

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["patient_id"], "P4")
        self.assertEqual(samples[0]["visit_id"], "403")
        self.assertEqual(samples[0]["conditions"], [["4280", "25000"]])
        self.assertEqual(samples[0]["next_conditions"], ["41401"])

    def test_sample_per_visit_skips_icd10_only_visit(self):
        task = NextVisitDiagnosisMIMIC4(sample_per_visit=True)

        samples = task(_mixed_icd_patient())

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["visit_id"], "403")


if __name__ == "__main__":
    unittest.main()
