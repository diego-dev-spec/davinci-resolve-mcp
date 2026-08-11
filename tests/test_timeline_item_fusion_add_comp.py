"""Tests for timeline_item_fusion(action="add_comp") — count-verified creation.

Background (measured live on Studio 20.2.1.6, pruebas/add-comp-gate-01 in the
produccion-visual-dp lab repo): AddFusionComp() returns None on a "Solid Color"
GENERATOR timeline item while creating the comp anyway, and returns the comp
object on a media-backed clip. The previous implementation judged success by the
return value alone, so it reported a false failure on generators — and its
retryable error invited a retry that appended a SECOND comp.

These tests pin the replacement contract: success is decided ONLY by
count_after == count_before + 1; the native return value and any exception are
diagnostics; identification of the created comp is best-effort and never a
success criterion.

No live Resolve is needed — a fake timeline item drives every branch.
"""

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import server


_RAISE = object()  # sentinel: "this read should raise"


class FakeItem:
    """Timeline item whose comp count follows a scripted sequence.

    `counts` is consumed one entry per GetFusionCompCount() call: the first is
    the baseline read, the second the verification read. An entry of _RAISE
    makes that read raise; None makes it return None (unreadable). Same idea
    for `name_lists` against GetFusionCompNameList().
    """

    def __init__(self, counts, add_result=None, add_raises=None, name_lists=None):
        self._counts = list(counts)
        self._add_result = add_result
        self._add_raises = add_raises
        self._name_lists = list(name_lists) if name_lists is not None else None
        self.add_called = 0

    def GetFusionCompCount(self):
        value = self._counts.pop(0)
        if value is _RAISE:
            raise RuntimeError("GetFusionCompCount exploded")
        return value

    def GetFusionCompNameList(self):
        if self._name_lists is None:
            return []
        value = self._name_lists.pop(0)
        if value is _RAISE:
            raise RuntimeError("GetFusionCompNameList exploded")
        return value

    def AddFusionComp(self):
        self.add_called += 1
        if self._add_raises is not None:
            raise self._add_raises
        return self._add_result


class FakeComp:
    """Stand-in for the composition object Resolve returns on media-backed clips."""


def err_of(result):
    return result.get("error") or {}


class AddCompSuccessTests(unittest.TestCase):
    """delta == +1 is success, whatever the native call returned or raised."""

    def test_generator_case_returns_none_but_count_increments(self):
        # The G0 measurement: Solid Color generator, AddFusionComp -> None.
        item = FakeItem(counts=[0, 1], add_result=None,
                        name_lists=[{}, ["Composition 1"]])
        out = server._add_fusion_comp_verified(item)
        self.assertTrue(out["success"])
        self.assertEqual(out["comp_count_before"], 0)
        self.assertEqual(out["comp_count_after"], 1)
        self.assertEqual(out["delta"], 1)
        self.assertFalse(out["native_returned_object"])
        self.assertEqual(out["created_comp_name"], "Composition 1")
        self.assertNotIn("error", out)

    def test_media_backed_case_returns_object_and_count_increments(self):
        # The B_media measurement: media-backed clip, AddFusionComp -> comp.
        item = FakeItem(counts=[0, 1], add_result=FakeComp(),
                        name_lists=[{}, ["Composition 1"]])
        out = server._add_fusion_comp_verified(item)
        self.assertTrue(out["success"])
        self.assertTrue(out["native_returned_object"])
        self.assertEqual(out["delta"], 1)

    def test_exception_with_verified_increment_is_success(self):
        # Real state outranks the return mechanism: it raised, but a comp exists.
        item = FakeItem(counts=[0, 1], add_raises=RuntimeError("bridge hiccup"),
                        name_lists=[{}, ["Composition 1"]])
        out = server._add_fusion_comp_verified(item)
        self.assertTrue(out["success"])
        self.assertEqual(out["delta"], 1)
        self.assertFalse(out["native_returned_object"])
        self.assertIn("bridge hiccup", out["native_exception"])

    def test_increment_on_item_that_already_had_comps(self):
        item = FakeItem(counts=[2, 3], add_result=FakeComp(),
                        name_lists=[["A", "B"], ["A", "B", "C"]])
        out = server._add_fusion_comp_verified(item)
        self.assertTrue(out["success"])
        self.assertEqual(out["comp_count_before"], 2)
        self.assertEqual(out["comp_count_after"], 3)
        self.assertEqual(out["delta"], 1)
        self.assertEqual(out["created_comp_name"], "C")

    def test_success_response_is_json_serialisable(self):
        # Guards against a PyRemoteObject (or any bridge handle) leaking out.
        item = FakeItem(counts=[0, 1], add_result=FakeComp(),
                        name_lists=[{}, ["Composition 1"]])
        out = server._add_fusion_comp_verified(item)
        json.dumps(out)  # must not raise


class AddCompFailureTests(unittest.TestCase):
    """Anything other than a verified +1 is an error, and never retryable
    once AddFusionComp has actually been invoked."""

    def test_no_object_and_no_increment_is_a_failure(self):
        item = FakeItem(counts=[0, 0], add_result=None, name_lists=[{}, {}])
        out = server._add_fusion_comp_verified(item)
        self.assertNotIn("success", out)
        self.assertEqual(err_of(out)["code"], "ADD_COMP_FAILED")
        self.assertFalse(err_of(out)["retryable"])
        self.assertEqual(err_of(out)["state"]["delta"], 0)

    def test_exception_and_no_increment_is_a_failure(self):
        item = FakeItem(counts=[0, 0], add_raises=RuntimeError("nope"),
                        name_lists=[{}, {}])
        out = server._add_fusion_comp_verified(item)
        self.assertEqual(err_of(out)["code"], "ADD_COMP_FAILED")
        self.assertFalse(err_of(out)["retryable"])
        self.assertIn("nope", err_of(out)["state"]["native_exception"])

    def test_object_returned_without_increment_is_unverified(self):
        # The signals disagree — refuse to call it either way.
        item = FakeItem(counts=[0, 0], add_result=FakeComp(), name_lists=[{}, {}])
        out = server._add_fusion_comp_verified(item)
        self.assertEqual(err_of(out)["code"], "ADD_COMP_UNVERIFIED")
        self.assertFalse(err_of(out)["retryable"])
        self.assertTrue(err_of(out)["state"]["native_returned_object"])

    def test_two_comps_appearing_is_an_unexpected_delta(self):
        item = FakeItem(counts=[0, 2], add_result=FakeComp(),
                        name_lists=[{}, ["Composition 1", "Composition 2"]])
        out = server._add_fusion_comp_verified(item)
        self.assertEqual(err_of(out)["code"], "ADD_COMP_UNEXPECTED_DELTA")
        self.assertFalse(err_of(out)["retryable"])
        self.assertEqual(err_of(out)["state"]["delta"], 2)

    def test_comps_disappearing_is_an_unexpected_delta(self):
        item = FakeItem(counts=[2, 1], add_result=FakeComp(),
                        name_lists=[["A", "B"], ["A"]])
        out = server._add_fusion_comp_verified(item)
        self.assertEqual(err_of(out)["code"], "ADD_COMP_UNEXPECTED_DELTA")
        self.assertFalse(err_of(out)["retryable"])
        self.assertEqual(err_of(out)["state"]["delta"], -1)

    def test_unreadable_baseline_refuses_to_mutate_and_stays_retryable(self):
        item = FakeItem(counts=[_RAISE], add_result=FakeComp())
        out = server._add_fusion_comp_verified(item)
        self.assertEqual(err_of(out)["code"], "ADD_COMP_BASELINE_UNREADABLE")
        self.assertTrue(err_of(out)["retryable"])
        self.assertEqual(item.add_called, 0)  # the whole point: nothing was mutated

    def test_non_numeric_baseline_also_refuses_to_mutate(self):
        item = FakeItem(counts=["not a number"], add_result=FakeComp())
        out = server._add_fusion_comp_verified(item)
        self.assertEqual(err_of(out)["code"], "ADD_COMP_BASELINE_UNREADABLE")
        self.assertEqual(item.add_called, 0)

    def test_unreadable_verification_reports_unknown_state(self):
        item = FakeItem(counts=[0, _RAISE], add_result=FakeComp(), name_lists=[{}])
        out = server._add_fusion_comp_verified(item)
        self.assertEqual(err_of(out)["code"], "ADD_COMP_STATE_UNVERIFIABLE")
        self.assertFalse(err_of(out)["retryable"])
        self.assertIsNone(err_of(out)["state"]["comp_count_after"])
        self.assertEqual(item.add_called, 1)  # it did run; that's why state is unknown


class CreatedCompIdentificationTests(unittest.TestCase):
    """Identification is best-effort and must never gate success."""

    def test_empty_dict_name_list_is_treated_as_no_comps(self):
        # The bridge returns an empty Lua table as {} — measured, not assumed.
        item = FakeItem(counts=[0, 1], add_result=None,
                        name_lists=[{}, ["Composition 1"]])
        out = server._add_fusion_comp_verified(item)
        self.assertEqual(out["created_comp_name"], "Composition 1")

    def test_duplicate_names_are_identified_by_multiset_not_set(self):
        # A set difference would report nothing here even though a comp appeared.
        item = FakeItem(counts=[1, 2], add_result=None,
                        name_lists=[["X"], ["X", "X"]])
        out = server._add_fusion_comp_verified(item)
        self.assertTrue(out["success"])
        self.assertEqual(out["created_comp_name"], "X")

    def test_ambiguous_name_diff_yields_null_name_but_still_succeeds(self):
        item = FakeItem(counts=[2, 3], add_result=None,
                        name_lists=[["A", "B"], ["A", "C", "D"]])
        out = server._add_fusion_comp_verified(item)
        self.assertTrue(out["success"])
        self.assertIsNone(out["created_comp_name"])

    def test_unreadable_name_list_after_still_succeeds(self):
        item = FakeItem(counts=[0, 1], add_result=None, name_lists=[{}, _RAISE])
        out = server._add_fusion_comp_verified(item)
        self.assertTrue(out["success"])
        self.assertIsNone(out["created_comp_name"])

    def test_unreadable_name_list_before_still_succeeds(self):
        item = FakeItem(counts=[0, 1], add_result=None,
                        name_lists=[_RAISE, ["Composition 1"]])
        out = server._add_fusion_comp_verified(item)
        self.assertTrue(out["success"])
        self.assertIsNone(out["created_comp_name"])


class HelperTests(unittest.TestCase):
    """Direct coverage of the normalisation helpers."""

    def test_name_list_normalisation_shapes(self):
        class I:
            def __init__(self, value):
                self._value = value

            def GetFusionCompNameList(self):
                return self._value

        self.assertEqual(server._fusion_comp_name_list(I({})), [])
        self.assertEqual(server._fusion_comp_name_list(I(["a", "b"])), ["a", "b"])
        self.assertEqual(server._fusion_comp_name_list(I(("a",))), ["a"])
        self.assertEqual(server._fusion_comp_name_list(I({1: "a", 2: "b"})), ["a", "b"])
        self.assertIsNone(server._fusion_comp_name_list(I(None)))

    def test_identify_created_comp_edge_cases(self):
        self.assertIsNone(server._fusion_identify_created_comp(None, ["a"]))
        self.assertIsNone(server._fusion_identify_created_comp(["a"], None))
        self.assertIsNone(server._fusion_identify_created_comp(["a"], ["a"]))
        self.assertEqual(server._fusion_identify_created_comp([], ["a"]), "a")
        self.assertIsNone(server._fusion_identify_created_comp([], ["a", "b"]))


class ToolWiringTests(unittest.TestCase):
    """The tool action reaches the verified implementation."""

    def test_add_comp_action_uses_the_verified_path(self):
        item = FakeItem(counts=[0, 1], add_result=None,
                        name_lists=[{}, ["Composition 1"]])

        def fake_get_item(p):
            return None, item, None

        original = server._get_item
        server._get_item = fake_get_item
        try:
            out = server.timeline_item_fusion("add_comp", {})
        finally:
            server._get_item = original

        self.assertTrue(out["success"])
        self.assertEqual(out["delta"], 1)
        self.assertFalse(out["native_returned_object"])
        self.assertEqual(item.add_called, 1)


if __name__ == "__main__":
    unittest.main()
