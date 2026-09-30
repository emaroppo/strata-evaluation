"""Every first-party failure mode keeps the promise a plugin's would."""

import pytest

from strata.evaluation.conformance import FailureModeContract, TaskContract
from strata.evaluation.failures import DEFAULT
from strata.evaluation.registry import available, resolve


def test_every_default_mode_is_registered():
    assert set(DEFAULT) <= set(available("failure_mode"))


class TestFirstPartyModes(FailureModeContract):
    @pytest.fixture(params=DEFAULT)
    def mode(self, request):
        return resolve("failure_mode", request.param)[0]


class TestFirstPartyTasks(TaskContract):
    @pytest.fixture(params=["classify", "entities", "mask"])
    def task(self, request):
        return resolve("task", request.param)[0]


OVERLAPPING = '''
from strata.evaluation.failures import FailureMode


class HalfCovered(FailureMode):
    """What `truncated` already measures, under another name."""

    name = "half_covered"
    version = "1"
    side = "gold"
    description = "some of the entity is uncovered"
    example = ("Dear [John Smith|PER],", "Dear [John|PER] Smith,")

    def detect(self, a):
        return {g for g, ps in enumerate(a.gold_to_pred) if ps and a.gold[g].chars - a.covered}
'''


def test_a_mode_that_measures_what_another_does_fails_naming_it(tmp_path):
    (tmp_path / "failures.py").write_text(OVERLAPPING)
    mode, identity = resolve("failure_mode", "failures.py:HalfCovered", root=tmp_path)
    assert identity.source.startswith("file:")
    with pytest.raises(AssertionError, match="also raises truncated"):
        FailureModeContract().test_its_example_raises_no_other_mode(mode)
