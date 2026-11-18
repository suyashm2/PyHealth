pyhealth.tasks.next_visit_diagnosis
===================================

Next-visit diagnosis prediction: given the diagnosis codes from a patient's
earlier hospital visits, predict the set of diagnosis codes recorded at the next
visit (a multilabel target). This is the task used by Doctor AI (Choi et al.,
MLHC 2016) and SHy (Yu et al., CHIL 2025).

By default each patient gives one sample, with the last visit as the target and
all earlier visits as the input, as in the SHy paper. Pass
``sample_per_visit=True`` to make one sample for every visit after the first.
The MIMIC-IV task keeps only ICD-9 codes, so that models using the ICD-9
hierarchy can be applied.

.. autoclass:: pyhealth.tasks.next_visit_diagnosis.NextVisitDiagnosisMIMIC3
    :members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.next_visit_diagnosis.NextVisitDiagnosisMIMIC4
    :members:
    :show-inheritance:
