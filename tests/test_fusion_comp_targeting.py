"""Regression tests for fusion_comp timeline targeting helpers."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import server


class FakeFusion:
    def __init__(self, comp):
        self._comp = comp

    def GetCurrentComp(self):
        return self._comp


class FakeResolve:
    def __init__(self, comp):
        self._fusion = FakeFusion(comp)

    def Fusion(self):
        return self._fusion


class FakeTimelineItem:
    def __init__(self, unique_id, comp_count=1):
        self._unique_id = unique_id
        self._comp_count = comp_count
        self.requested_comp_index = None

    def GetUniqueId(self):
        return self._unique_id

    def GetFusionCompCount(self):
        return self._comp_count

    def GetFusionCompByIndex(self, comp_index):
        self.requested_comp_index = comp_index
        return {"comp_index": comp_index}

    def GetFusionCompByName(self, comp_name):
        return {"comp_name": comp_name}


class FakeTimeline:
    def __init__(self, tracks):
        self._tracks = tracks

    def GetTrackCount(self, track_type):
        return len(self._tracks.get(track_type, {}))

    def GetItemListInTrack(self, track_type, track_index):
        return self._tracks.get(track_type, {}).get(track_index, [])


class FakeBezierSplineModifier:
    """Stand-in for what Input.GetConnectedOutput().GetTool() resolves to once
    AddModifier() attaches a BezierSpline. Doubles as the Output object itself
    (GetTool() returns self) -- these unit tests don't need to exercise the
    Output/Tool indirection, that path is validated natively (Gates A1-A3 in
    produccion-visual-dp).
    """

    def __init__(self, regid="BezierSpline"):
        self._regid = regid
        self.raw_keyframes = {}  # frame(float) -> {1: value}, mirrors GetKeyFrames() shape

    def GetTool(self):
        return self

    def GetAttrs(self):
        return {"TOOLS_RegID": self._regid, "TOOLS_Name": "FakeModifier"}

    def GetKeyFrames(self):
        return dict(self.raw_keyframes)


class FakeFusionInput:
    """Minimal stand-in for a Fusion Input object.

    `inp[time] = value` records a keyframe only conceptually; in real Fusion it
    sets a STATIC value unless a spline modifier is attached first. When a
    FakeBezierSplineModifier is connected, the write also lands in its
    raw_keyframes, mirroring how a real animated write shows up in
    BezierSpline.GetKeyFrames().
    """

    def __init__(self, connected_output=None, keyframe_values=None, write_exception=None, event_log=None):
        self._connected_output = connected_output
        self.assignments = {}
        # frame_position -> value, modelling existing keyframes on the input.
        self.keyframe_values = dict(keyframe_values or {})
        self.write_exception = write_exception
        self.event_log = event_log if event_log is not None else []

    def __bool__(self):
        return True

    def GetConnectedOutput(self):
        return self._connected_output

    def __setitem__(self, time, value):
        if self.write_exception is not None:
            raise self.write_exception
        self.event_log.append(("write", time, value))
        self.assignments[time] = value
        self.keyframe_values[time] = value
        modifier = self._connected_output
        if modifier is not None and hasattr(modifier, "raw_keyframes"):
            modifier.raw_keyframes[float(time)] = {1: value}

    def GetKeyFrames(self):
        # Mirror Fusion: {1-based index: frame_position}, sorted by frame.
        frames = sorted(self.keyframe_values)
        return {i + 1: frame for i, frame in enumerate(frames)} or None


class FakeFusionTool:
    def __init__(self, inputs, addmodifier_exception=None, addmodifier_return=True,
                 addmodifier_regid="BezierSpline", addmodifier_connects=True,
                 addmodifier_extra_seed=None, event_log=None):
        self._inputs = inputs
        self.modifiers_added = []
        self.addmodifier_exception = addmodifier_exception
        self.addmodifier_return = addmodifier_return
        self.addmodifier_regid = addmodifier_regid
        self.addmodifier_connects = addmodifier_connects
        # Optional (frame, value) pre-seeded into the freshly-attached modifier
        # BEFORE the real write happens -- simulates a phantom keyframe left
        # over by a still-buggy native AddModifier, for postcondition tests.
        self.addmodifier_extra_seed = addmodifier_extra_seed
        self.event_log = event_log if event_log is not None else []

    def __getitem__(self, name):
        return self._inputs.get(name)

    def GetInput(self, name, frame):
        inp = self._inputs.get(name)
        return inp.keyframe_values.get(frame) if inp is not None else None

    def AddModifier(self, input_name, modifier_type):
        self.modifiers_added.append((input_name, modifier_type))
        self.event_log.append(("addmodifier", input_name, modifier_type))
        if self.addmodifier_exception is not None:
            raise self.addmodifier_exception
        inp = self._inputs.get(input_name)
        if inp is not None and self.addmodifier_connects:
            modifier = FakeBezierSplineModifier(regid=self.addmodifier_regid)
            if self.addmodifier_extra_seed is not None:
                frame, value = self.addmodifier_extra_seed
                modifier.raw_keyframes[float(frame)] = {1: value}
            inp._connected_output = modifier
        return self.addmodifier_return


class FakeFusionComp:
    """CurrentTime getter/setter support consumable "plan" queues
    (get_currenttime_plan / compn_currenttime_plan) so a test can make a
    SPECIFIC read return an unexpected value or raise, without affecting
    other reads in the same call. Values/exceptions are popped in order;
    once a plan is empty, reads fall back to the real tracked value.
    """

    def __init__(self, tools, current_time=0, event_log=None):
        self._tools = tools
        self.lock_count = 0
        self.unlock_count = 0
        self._current_time = current_time
        self.event_log = event_log if event_log is not None else []
        self.get_currenttime_plan = []
        self.set_currenttime_exception = None
        self.compn_currenttime_plan = []

    def FindTool(self, name):
        return self._tools.get(name)

    def Lock(self):
        self.lock_count += 1
        self.event_log.append("lock")

    def Unlock(self):
        self.unlock_count += 1
        self.event_log.append("unlock")

    @property
    def CurrentTime(self):
        if self.get_currenttime_plan:
            item = self.get_currenttime_plan.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return self._current_time

    @CurrentTime.setter
    def CurrentTime(self, value):
        self.event_log.append(("set_currenttime", value))
        if self.set_currenttime_exception is not None:
            exc, self.set_currenttime_exception = self.set_currenttime_exception, None
            raise exc
        self._current_time = value

    def GetAttrs(self):
        if self.compn_currenttime_plan:
            value = self.compn_currenttime_plan.pop(0)
        else:
            value = self._current_time
        return {"COMPN_CurrentTime": value}


class FusionAddKeyframeTests(unittest.TestCase):
    def _run(self, comp, params):
        with patch.object(server, "_resolve_fusion_comp", return_value=(comp, None)):
            return server.fusion_comp("add_keyframe", params)

    # --- Legacy / out-of-scope paths: byte-for-byte unchanged behavior ---

    def test_attaches_bezierspline_on_virgin_input(self):
        # original_time (comp default 0) == requested_time (0): the new
        # CurrentTime-pin path must not move CurrentTime at all (spec #2).
        inp = FakeFusionInput(connected_output=None)
        tool = FakeFusionTool({"Size": inp})
        comp = FakeFusionComp({"Transform1": tool}, current_time=0)

        result = self._run(comp, {
            "tool_name": "Transform1", "input_name": "Size", "time": 0, "value": 1.0,
        })

        self.assertTrue(result.get("success"), result)
        self.assertEqual(tool.modifiers_added, [("Size", "BezierSpline")])
        self.assertEqual(inp.assignments, {0: 1.0})
        self.assertEqual((comp.lock_count, comp.unlock_count), (1, 1))
        # No CurrentTime write should have happened -- original already matched requested.
        self.assertEqual([e for e in comp.event_log if isinstance(e, tuple) and e[0] == "set_currenttime"], [])
        self.assertEqual(comp.CurrentTime, 0)
        modifier = inp.GetConnectedOutput()
        self.assertEqual(modifier.GetKeyFrames(), {0.0: {1: 1.0}})

    def test_skips_modifier_when_already_animated(self):
        inp = FakeFusionInput(connected_output=object())
        tool = FakeFusionTool({"Size": inp})
        comp = FakeFusionComp({"Transform1": tool}, current_time=42)

        result = self._run(comp, {
            "tool_name": "Transform1", "input_name": "Size", "time": 75, "value": 1.4,
        })

        self.assertTrue(result.get("success"))
        self.assertEqual(tool.modifiers_added, [])
        self.assertEqual(inp.assignments, {75: 1.4})
        # Already-animated path is legacy -- CurrentTime must never be touched.
        self.assertEqual(comp.event_log, ["lock", "unlock"])
        self.assertEqual(comp.CurrentTime, 42)

    def test_honors_custom_modifier_param(self):
        inp = FakeFusionInput(connected_output=None)
        tool = FakeFusionTool({"Center": inp})
        comp = FakeFusionComp({"Transform1": tool}, current_time=42)

        self._run(comp, {
            "tool_name": "Transform1", "input_name": "Center",
            "time": 0, "value": [0.5, 0.5], "modifier": "Path",
        })

        self.assertEqual(tool.modifiers_added, [("Center", "Path")])
        # Path is legacy in v1 -- CurrentTime must never be touched.
        self.assertEqual(comp.event_log, ["lock", "unlock"])
        self.assertEqual(comp.CurrentTime, 42)

    def test_missing_input_returns_error_and_unlocks(self):
        tool = FakeFusionTool({})
        comp = FakeFusionComp({"Transform1": tool})

        result = self._run(comp, {
            "tool_name": "Transform1", "input_name": "Nope", "time": 0, "value": 1.0,
        })

        self.assertIn("error", result)
        self.assertEqual(tool.modifiers_added, [])
        # comp must be unlocked even on the error path.
        self.assertEqual((comp.lock_count, comp.unlock_count), (1, 1))

    # --- New scope: virgin input + BezierSpline, CurrentTime pin strategy ---

    def test_pins_currenttime_when_original_differs_and_restores_it(self):
        # Spec test #1 (happy path) + #14 (critical ordering).
        shared_log = []
        inp = FakeFusionInput(connected_output=None, event_log=shared_log)
        tool = FakeFusionTool({"Size": inp}, event_log=shared_log)
        comp = FakeFusionComp({"Transform1": tool}, current_time=25, event_log=shared_log)

        result = self._run(comp, {
            "tool_name": "Transform1", "input_name": "Size", "time": 0, "value": 0.5,
        })

        self.assertTrue(result.get("success"), result)
        self.assertEqual(tool.modifiers_added, [("Size", "BezierSpline")])
        modifier = inp.GetConnectedOutput()
        self.assertEqual(modifier.GetKeyFrames(), {0.0: {1: 0.5}})
        self.assertEqual(comp.CurrentTime, 25)  # restored
        self.assertEqual((comp.lock_count, comp.unlock_count), (1, 1))

        lock_index = shared_log.index("lock")
        pin_index = shared_log.index(("set_currenttime", 0))
        restore_index = shared_log.index(("set_currenttime", 25))
        unlock_index = shared_log.index("unlock")
        self.assertLess(lock_index, pin_index)
        self.assertLess(pin_index, restore_index)
        self.assertLess(restore_index, unlock_index)

    def test_unreadable_currenttime_blocks_addmodifier(self):
        inp = FakeFusionInput(connected_output=None)
        tool = FakeFusionTool({"Size": inp})
        comp = FakeFusionComp({"Transform1": tool}, current_time=25)
        comp.get_currenttime_plan = [RuntimeError("CurrentTime read boom")]

        result = self._run(comp, {
            "tool_name": "Transform1", "input_name": "Size", "time": 0, "value": 0.5,
        })

        self.assertIn("error", result)
        self.assertEqual(result["error"]["code"], "CURRENTTIME_UNREADABLE")
        self.assertEqual(tool.modifiers_added, [])
        self.assertEqual(inp.assignments, {})
        self.assertEqual((comp.lock_count, comp.unlock_count), (1, 1))

    def test_pin_assignment_exception_blocks_addmodifier(self):
        inp = FakeFusionInput(connected_output=None)
        tool = FakeFusionTool({"Size": inp})
        comp = FakeFusionComp({"Transform1": tool}, current_time=25)
        comp.set_currenttime_exception = RuntimeError("pin assignment boom")

        result = self._run(comp, {
            "tool_name": "Transform1", "input_name": "Size", "time": 0, "value": 0.5,
        })

        self.assertIn("error", result)
        self.assertEqual(result["error"]["code"], "CURRENTTIME_PIN_FAILED")
        self.assertEqual(tool.modifiers_added, [])
        # Cleanup must still restore -- the single-shot exception only hits the pin,
        # so the subsequent restore assignment succeeds and CurrentTime ends correct.
        self.assertEqual(comp.CurrentTime, 25)

    def test_pin_readback_mismatch_blocks_addmodifier(self):
        inp = FakeFusionInput(connected_output=None)
        tool = FakeFusionTool({"Size": inp})
        comp = FakeFusionComp({"Transform1": tool}, current_time=25)
        # The assignment itself "succeeds", but COMPN_CurrentTime disagrees.
        comp.compn_currenttime_plan = [999]

        result = self._run(comp, {
            "tool_name": "Transform1", "input_name": "Size", "time": 0, "value": 0.5,
        })

        self.assertIn("error", result)
        self.assertEqual(result["error"]["code"], "CURRENTTIME_PIN_FAILED")
        self.assertEqual(tool.modifiers_added, [])
        self.assertEqual(comp.CurrentTime, 25)  # restored regardless

    def test_addmodifier_exception_restores_and_errors(self):
        inp = FakeFusionInput(connected_output=None)
        tool = FakeFusionTool({"Size": inp}, addmodifier_exception=RuntimeError("AddModifier boom"))
        comp = FakeFusionComp({"Transform1": tool}, current_time=0)

        result = self._run(comp, {
            "tool_name": "Transform1", "input_name": "Size", "time": 0, "value": 0.5,
        })

        self.assertIn("error", result)
        self.assertEqual(result["error"]["code"], "ADDMODIFIER_FAILED")
        self.assertEqual(inp.assignments, {})  # no write attempted

    def test_addmodifier_falsy_return_with_connected_bezierspline_still_succeeds(self):
        # Corrected contract: AddModifier's return value is diagnostic only.
        inp = FakeFusionInput(connected_output=None)
        tool = FakeFusionTool({"Size": inp}, addmodifier_return=False)
        comp = FakeFusionComp({"Transform1": tool}, current_time=0)

        result = self._run(comp, {
            "tool_name": "Transform1", "input_name": "Size", "time": 0, "value": 0.5,
        })

        self.assertTrue(result.get("success"), result)
        modifier = inp.GetConnectedOutput()
        self.assertEqual(modifier.GetKeyFrames(), {0.0: {1: 0.5}})

    def test_addmodifier_wrong_regid_errors(self):
        inp = FakeFusionInput(connected_output=None)
        tool = FakeFusionTool({"Size": inp}, addmodifier_regid="Path")
        comp = FakeFusionComp({"Transform1": tool}, current_time=0)

        result = self._run(comp, {
            "tool_name": "Transform1", "input_name": "Size", "time": 0, "value": 0.5,
        })

        self.assertIn("error", result)
        self.assertEqual(result["error"]["code"], "ADDMODIFIER_FAILED")
        self.assertEqual(inp.assignments, {})

    def test_addmodifier_does_not_connect_errors(self):
        inp = FakeFusionInput(connected_output=None)
        tool = FakeFusionTool({"Size": inp}, addmodifier_connects=False)
        comp = FakeFusionComp({"Transform1": tool}, current_time=0)

        result = self._run(comp, {
            "tool_name": "Transform1", "input_name": "Size", "time": 0, "value": 0.5,
        })

        self.assertIn("error", result)
        self.assertEqual(result["error"]["code"], "ADDMODIFIER_FAILED")

    def test_keyframe_write_exception_restores_and_errors(self):
        inp = FakeFusionInput(connected_output=None, write_exception=RuntimeError("write boom"))
        tool = FakeFusionTool({"Size": inp})
        comp = FakeFusionComp({"Transform1": tool}, current_time=25)

        result = self._run(comp, {
            "tool_name": "Transform1", "input_name": "Size", "time": 0, "value": 0.5,
        })

        self.assertIn("error", result)
        self.assertEqual(result["error"]["code"], "KEYFRAME_WRITE_FAILED")
        # No destructive rollback -- the modifier stays connected (partial state).
        self.assertEqual(tool.modifiers_added, [("Size", "BezierSpline")])
        self.assertEqual(comp.CurrentTime, 25)  # still restored

    def test_seed_postcondition_mismatch_errors(self):
        # Simulates a still-buggy native AddModifier leaving a phantom keyframe
        # behind -- the read-back must catch it, not report success.
        inp = FakeFusionInput(connected_output=None, write_exception=None)
        tool = FakeFusionTool({"Size": inp}, addmodifier_extra_seed=(99, 42.0))
        comp = FakeFusionComp({"Transform1": tool}, current_time=25)

        result = self._run(comp, {
            "tool_name": "Transform1", "input_name": "Size", "time": 0, "value": 0.5,
        })

        self.assertIn("error", result)
        self.assertEqual(result["error"]["code"], "SEED_POSTCONDITION_MISMATCH")
        self.assertEqual(comp.CurrentTime, 25)  # still restored, no rollback of the phantom

    def test_restore_failure_on_happy_path_never_succeeds(self):
        # 8-10 all succeed, but the final CurrentTime verification disagrees --
        # must never report success (spec #12).
        inp = FakeFusionInput(connected_output=None)
        tool = FakeFusionTool({"Size": inp})
        comp = FakeFusionComp({"Transform1": tool}, current_time=25)
        # First GetAttrs() call is the pre-AddModifier pin check (must match 0);
        # second is the final restore check (deliberately wrong).
        comp.compn_currenttime_plan = [0, 999]

        result = self._run(comp, {
            "tool_name": "Transform1", "input_name": "Size", "time": 0, "value": 0.5,
        })

        self.assertIn("error", result)
        self.assertEqual(result["error"]["code"], "CURRENTTIME_RESTORE_FAILED")
        self.assertNotIn("primary_error_code", result["error"].get("state", {}))
        # The keyframe itself was written correctly -- only the global
        # CurrentTime postcondition failed.
        modifier = inp.GetConnectedOutput()
        self.assertEqual(modifier.GetKeyFrames(), {0.0: {1: 0.5}})

    def test_restore_failure_after_primary_error_keeps_primary_in_state(self):
        # Spec #13: primary failure (pin mismatch) + restore also fails ->
        # CURRENTTIME_RESTORE_FAILED wins, primary preserved in `state`.
        inp = FakeFusionInput(connected_output=None)
        tool = FakeFusionTool({"Size": inp})
        comp = FakeFusionComp({"Transform1": tool}, current_time=25)
        comp.compn_currenttime_plan = [999, 888]  # pin check fails, then restore check fails too

        result = self._run(comp, {
            "tool_name": "Transform1", "input_name": "Size", "time": 0, "value": 0.5,
        })

        self.assertIn("error", result)
        self.assertEqual(result["error"]["code"], "CURRENTTIME_RESTORE_FAILED")
        self.assertEqual(result["error"]["state"]["primary_error_code"], "CURRENTTIME_PIN_FAILED")
        self.assertEqual(tool.modifiers_added, [])


class FusionGetKeyframesTests(unittest.TestCase):
    def test_returns_frame_positions_and_values(self):
        # GetKeyFrames yields {index: frame}; the handler must report the frame
        # position as `time` and the GetInput(frame) result as `value`.
        inp = FakeFusionInput(
            connected_output=object(),
            keyframe_values={0.0: 1.0, 75.0: 1.4},
        )
        tool = FakeFusionTool({"Size": inp})
        comp = FakeFusionComp({"Transform1": tool})

        with patch.object(server, "_resolve_fusion_comp", return_value=(comp, None)):
            result = server.fusion_comp(
                "get_keyframes", {"tool_name": "Transform1", "input_name": "Size"}
            )

        self.assertEqual(
            result["keyframes"],
            [{"time": 0.0, "value": 1.0}, {"time": 75.0, "value": 1.4}],
        )

    def test_no_keyframes_returns_empty_list(self):
        inp = FakeFusionInput(connected_output=None, keyframe_values={})
        tool = FakeFusionTool({"Size": inp})
        comp = FakeFusionComp({"Transform1": tool})

        with patch.object(server, "_resolve_fusion_comp", return_value=(comp, None)):
            result = server.fusion_comp(
                "get_keyframes", {"tool_name": "Transform1", "input_name": "Size"}
            )

        self.assertEqual(result["keyframes"], [])


class FusionCompTargetingTests(unittest.TestCase):
    def test_active_comp_fallback_does_not_require_timeline(self):
        active_comp = object()

        with patch.object(server, "get_resolve", return_value=FakeResolve(active_comp)), patch.object(
            server,
            "_get_tl",
            side_effect=AssertionError("_get_tl should not be called without timeline scope"),
        ):
            comp, err = server._resolve_fusion_comp({})

        self.assertIs(comp, active_comp)
        self.assertIsNone(err)

    def test_bulk_set_inputs_requires_timeline_scope_per_op(self):
        with patch.object(
            server,
            "_resolve_fusion_comp",
            side_effect=AssertionError("_resolve_fusion_comp should not be called for unscoped bulk ops"),
        ):
            result = server._fusion_comp_bulk_set_inputs(
                {"ops": [{"tool_name": "Text1", "input_name": "StyledText", "value": "Hello"}]}
            )

        self.assertEqual(result["op_count"], 1)
        self.assertIn("timeline scope is required", result["results"][0]["error"])

    def test_find_timeline_item_by_id_scans_timeline_tracks(self):
        wanted = FakeTimelineItem("target")
        timeline = FakeTimeline({
            "video": {1: [FakeTimelineItem("video-1")]},
            "audio": {1: [wanted]},
        })

        self.assertIs(server._find_timeline_item_by_id(timeline, "target"), wanted)

    def test_comp_index_defaults_to_first_comp_and_validates_range(self):
        item = FakeTimelineItem("clip-1", comp_count=2)

        comp, err = server._get_fusion_comp_on_timeline_item(item, {})
        self.assertEqual(comp, {"comp_index": 1})
        self.assertIsNone(err)

        comp, err = server._get_fusion_comp_on_timeline_item(item, {"comp_index": 3})
        self.assertIsNone(comp)
        self.assertIn("item has 2 comp(s)", (err["error"].get("message","") if isinstance(err["error"], dict) else err["error"]))


if __name__ == "__main__":
    unittest.main()
