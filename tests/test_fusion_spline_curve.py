"""Tests for fusion_comp(action="get_spline_curve") — read-only BezierSpline
curve introspection (value + LH/RH handles + Flags), complementing the
already-existing get_keyframes (times only, no curve shape).

A fake comp/tool/input/output graph is used so no live Resolve is needed.
Comparisons are semantic (numeric tolerance on floats via assertAlmostEqual),
never string/byte equality against raw Fusion output — floats crossing the
bridge are not guaranteed to be bit-identical.
"""
import copy
import unittest
from unittest import mock

import src.server as s


class FakeOutput:
    """Stand-in for the Output object GetConnectedOutput() returns."""

    def __init__(self, tool=None, raise_on_get_tool=None):
        self._tool = tool
        self._raise = raise_on_get_tool

    def GetTool(self):
        if self._raise is not None:
            raise self._raise
        return self._tool


class FakeModifierTool:
    """Stand-in for the Fusion Tool object behind a modifier (BezierSpline, Path, ...)."""

    def __init__(self, regid, name="Mod1", keyframes=None, raise_on_get_keyframes=None):
        self._regid = regid
        self._name = name
        self._keyframes = keyframes
        self._raise = raise_on_get_keyframes

    def GetAttrs(self):
        return {"TOOLS_RegID": self._regid, "TOOLS_Name": self._name}

    def GetKeyFrames(self):
        if self._raise is not None:
            raise self._raise
        return self._keyframes


class FakeInput:
    def __init__(self, connected_output=None):
        self._connected_output = connected_output

    def __bool__(self):
        return True

    def GetConnectedOutput(self):
        return self._connected_output


class FakeTool:
    def __init__(self, inputs):
        self._inputs = inputs

    def __getitem__(self, name):
        return self._inputs.get(name)

    def GetInput(self, name, time):
        # Only used by delete_keyframe's Path branch, via FakePathInput's
        # own value store, so surviving-frame values reflect any prior write.
        return self._inputs[name].get_value(time)


class FakeComp:
    def __init__(self, tools):
        self._tools = tools

    def FindTool(self, name):
        return self._tools.get(name)

    # No-ops: get_spline_curve never calls these (read-only), but
    # set_spline_handles does (Lock/Unlock around the write, StartUndo/EndUndo
    # wrapping it) — present unconditionally so the same FakeComp works for
    # both action's tests.
    def Lock(self):
        pass

    def Unlock(self):
        pass

    def StartUndo(self, name):
        pass

    def EndUndo(self, keep):
        pass


class FakeWritableModifierTool(FakeModifierTool):
    """FakeModifierTool with a working SetKeyFrames(). set_spline_handles
    always re-reads via GetKeyFrames() immediately after writing, so the
    fake must actually reflect the write for read-back verification to be
    exercised meaningfully."""

    def __init__(self, *args, raise_on_set_keyframes=None, corrupt_after_write=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._raise_on_set = raise_on_set_keyframes
        self._corrupt_after_write = corrupt_after_write
        self.set_keyframes_calls = []

    def SetKeyFrames(self, raw):
        self.set_keyframes_calls.append(raw)
        if self._raise_on_set is not None:
            raise self._raise_on_set
        self._keyframes = self._corrupt_after_write if self._corrupt_after_write is not None else raw
        return None  # empirically confirmed return value — see Gate A/B/C


class FakeDeletableModifierTool(FakeModifierTool):
    """FakeModifierTool with a working DeleteKeyFrames(frame). delete_keyframe
    always re-reads via GetKeyFrames() immediately after deleting, so the
    fake must actually reflect the deletion for read-back verification to be
    exercised meaningfully."""

    def __init__(self, *args, raise_on_delete_keyframes=None, corrupt_after_delete=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._raise_on_delete = raise_on_delete_keyframes
        self._corrupt_after_delete = corrupt_after_delete
        self.delete_keyframes_calls = []

    def DeleteKeyFrames(self, frame):
        self.delete_keyframes_calls.append(frame)
        if self._raise_on_delete is not None:
            raise self._raise_on_delete
        if self._corrupt_after_delete is not None:
            self._keyframes = self._corrupt_after_delete
        else:
            self._keyframes = {k: v for k, v in self._keyframes.items() if k != frame}
        return None  # empirically confirmed return value — see Gate A (delete)


class FakePathModifierTool(FakeModifierTool):
    """Stand-in for the Fusion Tool object behind a Path modifier (animates a
    Point input). GetKeyFrames() returns {index: time} -- NOT {time: value}
    like BezierSpline -- empirically confirmed, see
    pruebas/keyframe-path-delete-lab-01/resultado.md."""

    def __init__(self, times, name="Path1", regid="Path"):
        super().__init__(regid, name=name)
        self._times = sorted(times)

    def GetKeyFrames(self):
        return {i + 1: t for i, t in enumerate(self._times)}


_UNSET = object()  # sentinel: distinguishes "not passed" from an explicit None


class FakePathInput:
    """Stand-in for a Point Input animated via a Path modifier. Supports the
    validated native deletion mechanism (Input[time] = None) -- see
    pruebas/keyframe-path-delete-lab-01/resultado.md. No DeleteKeyFrames()
    equivalent exists on Path (empirically confirmed: calling it raises
    TypeError: 'NoneType' object is not callable), so delete_keyframe's Path
    branch never calls it."""

    def __init__(self, connected_output, values, raise_on_setitem=None,
                 connected_output_after=_UNSET, corrupt_times_after=None,
                 corrupt_value_at=None):
        self._connected_output = connected_output
        # _UNSET (default) means "unchanged" -- most tests want
        # GetConnectedOutput() to keep returning the same output after the
        # write. An explicit None means "lost the connection entirely" and
        # must be distinguishable from "not passed" at all.
        self._connected_output_after = (
            connected_output if connected_output_after is _UNSET else connected_output_after
        )
        self._values = dict(values)
        self._raise_on_setitem = raise_on_setitem
        self._corrupt_times_after = corrupt_times_after
        self._corrupt_value_at = corrupt_value_at
        self.setitem_calls = []
        self._written = False

    def __bool__(self):
        return True

    def GetAttrs(self):
        return {"INPS_DataType": "Point"}

    def GetConnectedOutput(self):
        return self._connected_output_after if self._written else self._connected_output

    def get_value(self, time):
        return self._values.get(time)

    def __setitem__(self, time, value):
        self.setitem_calls.append((time, value))
        if self._raise_on_setitem is not None:
            # Exception happens before any mutation -- must not leave a
            # false partial write behind.
            raise self._raise_on_setitem
        self._written = True
        if value is None:
            if self._corrupt_times_after is not None:
                modifier = self._connected_output.GetTool() if self._connected_output else None
                if modifier is not None:
                    modifier._times = list(self._corrupt_times_after)
            else:
                self._values.pop(time, None)
                modifier = self._connected_output.GetTool() if self._connected_output else None
                if modifier is not None:
                    modifier._times = [t for t in modifier._times if abs(t - time) > 1e-9]
            if self._corrupt_value_at is not None:
                self._values.update(self._corrupt_value_at)


def _dispatch(comp, params):
    with mock.patch.object(s, "_resolve_fusion_comp", return_value=(comp, None)):
        return s.fusion_comp("get_spline_curve", params)


def _dispatch_set(comp, params):
    with mock.patch.object(s, "_resolve_fusion_comp", return_value=(comp, None)):
        return s.fusion_comp("set_spline_handles", params)


def _dispatch_delete(comp, params):
    with mock.patch.object(s, "_resolve_fusion_comp", return_value=(comp, None)):
        return s.fusion_comp("delete_keyframe", params)


# Raw shape captured empirically from BezierSpline.GetKeyFrames() against a
# real 2-keyframe linear curve (Transform_Lab.Size, frame 0=0.5, frame 60=1.5),
# see pruebas/keyframe-easing-lab-01/resultado_validacion_nativa.md:
#   {0.0: {1: 0.5, 'RH': {1: 20.0, 2: 0.333...}},
#    60.0: {1: 1.5, 'LH': {1: -20.0, 2: -0.333...}}}
REAL_RAW_CURVE = {
    0.0: {1: 0.5, "RH": {1: 20.0, 2: 0.33333333333333326}},
    60.0: {1: 1.5, "LH": {1: -20.0, 2: -0.3333333333333335}},
}

# Raw shape captured empirically from a 3-keyframe curve before deletion,
# see pruebas/keyframe-delete-lab-01/resultado_gate_a_native_delete_keyframes.md:
#   Transform_Lab.Size, frame 0=0.5, frame 30=1.0, frame 60=1.5.
THREE_KEYFRAME_RAW_CURVE = {
    0.0: {1: 0.5, "RH": {1: 10.0, 2: 0.16666666666666663}},
    30.0: {1: 1.0, "LH": {1: -10.0, 2: -0.16666666666666674}, "RH": {1: 10.0, 2: 0.16666666666666674}},
    60.0: {1: 1.5, "LH": {1: -10.0, 2: -0.16666666666666674}},
}

# Same curve after Gate A actually deleted frame 30.0 against live Resolve --
# handles on 0/60 legitimately recalculated, values unchanged.
THREE_KEYFRAME_RAW_CURVE_AFTER_DELETE_30 = {
    0.0: {1: 0.5, "RH": {1: 20.0, 2: 0.33333333333333326}},
    60.0: {1: 1.5, "LH": {1: -20.0, 2: -0.3333333333333335}},
}

# Point values captured empirically from a 3-keyframe Path curve before
# deletion, see pruebas/keyframe-path-delete-lab-01/resultado.md:
#   Transform_Lab.Center, frame 0=[0.2,0.2], 30=[0.5,0.8], 60=[0.8,0.2].
PATH_THREE_KEYFRAME_VALUES = {
    0.0: {1: 0.2, 2: 0.2, 3: 0.0},
    30.0: {1: 0.5, 2: 0.8, 3: 0.0},
    60.0: {1: 0.8, 2: 0.2, 3: 0.0},
}


class GetSplineCurveTests(unittest.TestCase):
    def _bezier_tool_input(self, keyframes=None, raise_on_get_keyframes=None,
                            modifier_regid="BezierSpline", modifier_name="Transform_LabSize"):
        modifier = FakeModifierTool(
            modifier_regid, name=modifier_name,
            keyframes=keyframes, raise_on_get_keyframes=raise_on_get_keyframes,
        )
        output = FakeOutput(tool=modifier)
        inp = FakeInput(connected_output=output)
        tool = FakeTool({"Size": inp})
        return FakeComp({"Transform1": tool})

    def test_two_keyframes_with_lh_and_rh(self):
        comp = self._bezier_tool_input(keyframes=REAL_RAW_CURVE)
        out = _dispatch(comp, {"tool_name": "Transform1", "input_name": "Size"})

        self.assertTrue(out["success"])
        self.assertEqual(out["modifier_type"], "BezierSpline")
        self.assertEqual(out["modifier_name"], "Transform_LabSize")
        self.assertNotIn("raw_curve", out)  # never part of the normal success contract
        self.assertEqual(len(out["points"]), 2)

        p0, p1 = out["points"]
        self.assertAlmostEqual(p0["time"], 0.0, places=9)
        self.assertAlmostEqual(p0["value"], 0.5, places=9)
        self.assertIsNone(p0["lh"])
        self.assertIsNotNone(p0["rh"])
        self.assertAlmostEqual(p0["rh"]["time_offset"], 20.0, places=9)
        self.assertAlmostEqual(p0["rh"]["value_offset"], 1.0 / 3.0, places=6)
        self.assertIsNone(p0["flags"])

        self.assertAlmostEqual(p1["time"], 60.0, places=9)
        self.assertAlmostEqual(p1["value"], 1.5, places=9)
        self.assertIsNotNone(p1["lh"])
        self.assertIsNone(p1["rh"])
        self.assertAlmostEqual(p1["lh"]["time_offset"], -20.0, places=9)
        self.assertAlmostEqual(p1["lh"]["value_offset"], -1.0 / 3.0, places=6)
        self.assertIsNone(p1["flags"])

    def test_points_sorted_by_time_regardless_of_dict_order(self):
        unordered = {60.0: REAL_RAW_CURVE[60.0], 0.0: REAL_RAW_CURVE[0.0]}
        comp = self._bezier_tool_input(keyframes=unordered)
        out = _dispatch(comp, {"tool_name": "Transform1", "input_name": "Size"})

        times = [pt["time"] for pt in out["points"]]
        self.assertEqual(times, sorted(times))

    def test_flags_present_is_preserved_verbatim(self):
        keyframes = {
            13.0: {1: 0.0, "RH": {1: 20.0, 2: 0.333}},
            34.0: {1: 1.0, "LH": {1: 27.0, 2: 0.666}, "RH": {1: 56.0, 2: 1.0},
                   "Flags": {"Linear": True}},
        }
        comp = self._bezier_tool_input(keyframes=keyframes)
        out = _dispatch(comp, {"tool_name": "Transform1", "input_name": "Size"})

        self.assertTrue(out["success"])
        by_time = {pt["time"]: pt for pt in out["points"]}
        self.assertEqual(by_time[34.0]["flags"], {"Linear": True})
        self.assertIsNone(by_time[13.0]["flags"])  # Flags absent -> null, not omitted/invented

    def test_handle_absent_is_null_not_omitted(self):
        keyframes = {0.0: {1: 0.5}}  # neither LH nor RH present
        comp = self._bezier_tool_input(keyframes=keyframes)
        out = _dispatch(comp, {"tool_name": "Transform1", "input_name": "Size"})

        self.assertTrue(out["success"])
        point = out["points"][0]
        self.assertIn("lh", point)
        self.assertIn("rh", point)
        self.assertIsNone(point["lh"])
        self.assertIsNone(point["rh"])

    def test_no_keyframes_returns_empty_points(self):
        comp = self._bezier_tool_input(keyframes=None)
        out = _dispatch(comp, {"tool_name": "Transform1", "input_name": "Size"})

        self.assertTrue(out["success"])
        self.assertEqual(out["points"], [])

    def test_missing_tool_returns_error(self):
        comp = FakeComp({})
        out = _dispatch(comp, {"tool_name": "Nope", "input_name": "Size"})
        self.assertIn("error", out)

    def test_missing_input_returns_error(self):
        comp = FakeComp({"Transform1": FakeTool({})})
        out = _dispatch(comp, {"tool_name": "Transform1", "input_name": "Nope"})
        self.assertIn("error", out)

    def test_input_not_animated_returns_error(self):
        inp = FakeInput(connected_output=None)
        tool = FakeTool({"Size": inp})
        comp = FakeComp({"Transform1": tool})
        out = _dispatch(comp, {"tool_name": "Transform1", "input_name": "Size"})

        self.assertIn("error", out)
        self.assertEqual(out["error"]["code"], "NOT_ANIMATED")

    def test_connected_modifier_not_bezierspline_returns_error(self):
        # Real case: Point inputs animated via modifier="Path" connect a Path
        # modifier instead of BezierSpline — must error clearly, not misparse.
        comp = self._bezier_tool_input(keyframes={0.0: {1: [0.5, 0.5]}}, modifier_regid="Path")
        out = _dispatch(comp, {"tool_name": "Transform1", "input_name": "Size"})

        self.assertIn("error", out)
        self.assertEqual(out["error"]["code"], "UNSUPPORTED_MODIFIER")
        self.assertEqual(out["error"]["state"]["modifier_type"], "Path")

    def test_get_tool_returns_none(self):
        output = FakeOutput(tool=None)
        inp = FakeInput(connected_output=output)
        tool = FakeTool({"Size": inp})
        comp = FakeComp({"Transform1": tool})
        out = _dispatch(comp, {"tool_name": "Transform1", "input_name": "Size"})
        self.assertIn("error", out)

    def test_get_keyframes_exception_is_caught(self):
        comp = self._bezier_tool_input(raise_on_get_keyframes=RuntimeError("boom"))
        out = _dispatch(comp, {"tool_name": "Transform1", "input_name": "Size"})

        self.assertIn("error", out)
        self.assertIn("boom", out["error"]["message"])

    def test_unexpected_curve_shape_missing_value_key(self):
        # Entry missing the value key `1` entirely — must not be guessed.
        keyframes = {0.0: {"RH": {1: 20.0, 2: 0.333}}}
        comp = self._bezier_tool_input(keyframes=keyframes)
        out = _dispatch(comp, {"tool_name": "Transform1", "input_name": "Size"})

        self.assertIn("error", out)
        self.assertEqual(out["error"]["code"], "UNPARSEABLE_CURVE")
        self.assertIn("raw_curve", out["error"]["state"])
        self.assertIn(0.0, out["error"]["state"]["raw_curve"])

    def test_unexpected_extra_key_errors(self):
        keyframes = {0.0: {1: 0.5, "SomethingNew": 1}}
        comp = self._bezier_tool_input(keyframes=keyframes)
        out = _dispatch(comp, {"tool_name": "Transform1", "input_name": "Size"})
        self.assertIn("error", out)
        self.assertEqual(out["error"]["code"], "UNPARSEABLE_CURVE")

    def test_handle_missing_subkey_errors(self):
        keyframes = {0.0: {1: 0.5, "RH": {1: 20.0}}}  # missing sub-key 2
        comp = self._bezier_tool_input(keyframes=keyframes)
        out = _dispatch(comp, {"tool_name": "Transform1", "input_name": "Size"})
        self.assertIn("error", out)
        self.assertEqual(out["error"]["code"], "UNPARSEABLE_CURVE")

    def test_missing_required_params_uses_guard(self):
        comp = self._bezier_tool_input(keyframes=REAL_RAW_CURVE)
        with mock.patch.object(s, "_resolve_fusion_comp", return_value=(comp, None)):
            out = s.fusion_comp("get_spline_curve", {"tool_name": "Transform1"})  # no input_name
        self.assertIn("error", out)


class SetSplineHandlesTests(unittest.TestCase):
    """set_spline_handles v1: LH/RH time_offset/value_offset on EXISTING
    keyframes only — never creates/deletes keyframes or handles, never
    touches time/value. Gate A/B/C already validated SetKeyFrames() itself
    empirically against live Resolve; these tests exercise the wrapper's
    own logic (validation-before-write, payload construction from the raw
    structure, read-back verification) against a fake bridge."""

    def _fresh_curve(self):
        return copy.deepcopy(REAL_RAW_CURVE)

    def _writable_bezier(self, keyframes, raise_on_set_keyframes=None,
                          corrupt_after_write=None, modifier_regid="BezierSpline"):
        modifier = FakeWritableModifierTool(
            modifier_regid, name="Transform_LabSize", keyframes=keyframes,
            raise_on_set_keyframes=raise_on_set_keyframes,
            corrupt_after_write=corrupt_after_write,
        )
        output = FakeOutput(tool=modifier)
        inp = FakeInput(connected_output=output)
        tool = FakeTool({"Size": inp})
        comp = FakeComp({"Transform1": tool})
        return comp, modifier

    def test_rh_value_offset_valid(self):
        comp, modifier = self._writable_bezier(self._fresh_curve())
        out = _dispatch_set(comp, {
            "tool_name": "Transform1", "input_name": "Size",
            "handles": [{"frame": 0.0, "side": "RH", "value_offset": 0.0}],
        })
        self.assertTrue(out.get("success"), out)
        self.assertEqual(modifier._keyframes[0.0]["RH"][2], 0.0)
        self.assertEqual(modifier._keyframes[0.0]["RH"][1], 20.0)  # untouched
        by_time = {pt["time"]: pt for pt in out["points"]}
        self.assertAlmostEqual(by_time[0.0]["rh"]["value_offset"], 0.0, places=9)

    def test_rh_time_offset_valid(self):
        comp, modifier = self._writable_bezier(self._fresh_curve())
        out = _dispatch_set(comp, {
            "tool_name": "Transform1", "input_name": "Size",
            "handles": [{"frame": 0.0, "side": "RH", "time_offset": 10.0}],
        })
        self.assertTrue(out.get("success"), out)
        self.assertEqual(modifier._keyframes[0.0]["RH"][1], 10.0)
        self.assertAlmostEqual(modifier._keyframes[0.0]["RH"][2], 0.33333333333333326, places=9)

    def test_rh_both_components(self):
        comp, modifier = self._writable_bezier(self._fresh_curve())
        out = _dispatch_set(comp, {
            "tool_name": "Transform1", "input_name": "Size",
            "handles": [{"frame": 0.0, "side": "RH", "time_offset": 5.0, "value_offset": 0.1}],
        })
        self.assertTrue(out.get("success"), out)
        self.assertEqual(modifier._keyframes[0.0]["RH"][1], 5.0)
        self.assertEqual(modifier._keyframes[0.0]["RH"][2], 0.1)

    def test_lh_value_offset_valid(self):
        comp, modifier = self._writable_bezier(self._fresh_curve())
        out = _dispatch_set(comp, {
            "tool_name": "Transform1", "input_name": "Size",
            "handles": [{"frame": 60.0, "side": "LH", "value_offset": -0.5}],
        })
        self.assertTrue(out.get("success"), out)
        self.assertEqual(modifier._keyframes[60.0]["LH"][2], -0.5)

    def test_lh_time_offset_valid(self):
        comp, modifier = self._writable_bezier(self._fresh_curve())
        out = _dispatch_set(comp, {
            "tool_name": "Transform1", "input_name": "Size",
            "handles": [{"frame": 60.0, "side": "LH", "time_offset": -10.0}],
        })
        self.assertTrue(out.get("success"), out)
        self.assertEqual(modifier._keyframes[60.0]["LH"][1], -10.0)

    def test_multiple_handles_one_call(self):
        comp, modifier = self._writable_bezier(self._fresh_curve())
        out = _dispatch_set(comp, {
            "tool_name": "Transform1", "input_name": "Size",
            "handles": [
                {"frame": 0.0, "side": "RH", "value_offset": 0.0},
                {"frame": 60.0, "side": "LH", "time_offset": -10.0},
            ],
        })
        self.assertTrue(out.get("success"), out)
        self.assertEqual(modifier._keyframes[0.0]["RH"][2], 0.0)
        self.assertEqual(modifier._keyframes[60.0]["LH"][1], -10.0)
        self.assertEqual(len(modifier.set_keyframes_calls), 1)  # exactly one write call

    def test_time_field_rejected(self):
        comp, modifier = self._writable_bezier(self._fresh_curve())
        out = _dispatch_set(comp, {
            "tool_name": "Transform1", "input_name": "Size",
            "handles": [{"frame": 0.0, "side": "RH", "value_offset": 0.0, "time": 5.0}],
        })
        self.assertIn("error", out)
        self.assertEqual(out["error"]["code"], "UNKNOWN_HANDLE_FIELD")
        self.assertEqual(out["error"]["state"]["unknown_fields"], ["time"])
        self.assertEqual(modifier.set_keyframes_calls, [])

    def test_value_field_rejected(self):
        comp, modifier = self._writable_bezier(self._fresh_curve())
        out = _dispatch_set(comp, {
            "tool_name": "Transform1", "input_name": "Size",
            "handles": [{"frame": 0.0, "side": "RH", "value_offset": 0.0, "value": 0.9}],
        })
        self.assertIn("error", out)
        self.assertEqual(out["error"]["code"], "UNKNOWN_HANDLE_FIELD")
        self.assertEqual(out["error"]["state"]["unknown_fields"], ["value"])
        self.assertEqual(modifier.set_keyframes_calls, [])

    def test_arbitrary_unknown_field_rejected(self):
        comp, modifier = self._writable_bezier(self._fresh_curve())
        out = _dispatch_set(comp, {
            "tool_name": "Transform1", "input_name": "Size",
            "handles": [{"frame": 0.0, "side": "RH", "value_offset": 0.0, "flags": {"Linear": True}}],
        })
        self.assertIn("error", out)
        self.assertEqual(out["error"]["code"], "UNKNOWN_HANDLE_FIELD")
        self.assertEqual(out["error"]["state"]["unknown_fields"], ["flags"])
        self.assertEqual(modifier.set_keyframes_calls, [])

    def test_unknown_frame_rejected_before_write(self):
        comp, modifier = self._writable_bezier(self._fresh_curve())
        out = _dispatch_set(comp, {
            "tool_name": "Transform1", "input_name": "Size",
            "handles": [{"frame": 30.0, "side": "RH", "value_offset": 0.0}],
        })
        self.assertIn("error", out)
        self.assertEqual(out["error"]["code"], "UNKNOWN_KEYFRAME")
        self.assertEqual(modifier.set_keyframes_calls, [])  # never wrote

    def test_unknown_handle_rejected_before_write(self):
        comp, modifier = self._writable_bezier(self._fresh_curve())
        out = _dispatch_set(comp, {
            "tool_name": "Transform1", "input_name": "Size",
            "handles": [{"frame": 0.0, "side": "LH", "value_offset": 0.0}],  # frame 0 has no LH
        })
        self.assertIn("error", out)
        self.assertEqual(out["error"]["code"], "UNKNOWN_HANDLE")
        self.assertEqual(modifier.set_keyframes_calls, [])

    def test_modifier_not_bezierspline_rejected(self):
        comp, modifier = self._writable_bezier({0.0: {1: [0.5, 0.5]}}, modifier_regid="Path")
        out = _dispatch_set(comp, {
            "tool_name": "Transform1", "input_name": "Size",
            "handles": [{"frame": 0.0, "side": "RH", "value_offset": 0.0}],
        })
        self.assertIn("error", out)
        self.assertEqual(out["error"]["code"], "UNSUPPORTED_MODIFIER")

    def test_neither_component_present_rejected(self):
        comp, modifier = self._writable_bezier(self._fresh_curve())
        out = _dispatch_set(comp, {
            "tool_name": "Transform1", "input_name": "Size",
            "handles": [{"frame": 0.0, "side": "RH"}],
        })
        self.assertIn("error", out)
        self.assertEqual(out["error"]["code"], "MISSING_HANDLE_COMPONENT")
        self.assertEqual(modifier.set_keyframes_calls, [])

    def test_setkeyframes_exception_is_failure(self):
        comp, modifier = self._writable_bezier(
            self._fresh_curve(), raise_on_set_keyframes=RuntimeError("boom"),
        )
        out = _dispatch_set(comp, {
            "tool_name": "Transform1", "input_name": "Size",
            "handles": [{"frame": 0.0, "side": "RH", "value_offset": 0.0}],
        })
        self.assertIn("error", out)
        self.assertEqual(out["error"]["code"], "SETKEYFRAMES_FAILED")

    def test_readback_mismatch_is_failure(self):
        # Simulate Fusion silently corrupting an untouched keyframe (value
        # changed on frame 60, never requested) — must be caught, not missed.
        corrupted = self._fresh_curve()
        corrupted[60.0][1] = 999.0
        comp, modifier = self._writable_bezier(self._fresh_curve(), corrupt_after_write=corrupted)
        out = _dispatch_set(comp, {
            "tool_name": "Transform1", "input_name": "Size",
            "handles": [{"frame": 0.0, "side": "RH", "value_offset": 0.0}],
        })
        self.assertIn("error", out)
        self.assertEqual(out["error"]["code"], "READBACK_MISMATCH")

    def test_unmentioned_frame_and_handle_preserved(self):
        comp, modifier = self._writable_bezier(self._fresh_curve())
        out = _dispatch_set(comp, {
            "tool_name": "Transform1", "input_name": "Size",
            "handles": [{"frame": 0.0, "side": "RH", "value_offset": 0.0}],
        })
        self.assertTrue(out.get("success"), out)
        # frame 60 / LH untouched
        self.assertEqual(modifier._keyframes[60.0][1], 1.5)
        self.assertEqual(modifier._keyframes[60.0]["LH"][1], -20.0)
        self.assertAlmostEqual(modifier._keyframes[60.0]["LH"][2], -0.3333333333333335, places=9)
        # frame 0 value/RH[1] untouched
        self.assertEqual(modifier._keyframes[0.0][1], 0.5)
        self.assertEqual(modifier._keyframes[0.0]["RH"][1], 20.0)


class DeleteKeyframeTests(unittest.TestCase):
    """delete_keyframe v1: delete one EXISTING keyframe from a BezierSpline
    curve only. Gate A (pruebas/keyframe-delete-lab-01) already validated
    BezierSpline.DeleteKeyFrames() itself empirically against live Resolve,
    including that neighboring LH/RH handles legitimately change; these
    tests exercise the wrapper's own logic (localization, validation-before-
    write, read-back verification that ignores handle drift) against a fake
    bridge."""

    def _fresh_curve(self):
        return copy.deepcopy(THREE_KEYFRAME_RAW_CURVE)

    def _deletable_bezier(self, keyframes, raise_on_delete_keyframes=None,
                           corrupt_after_delete=None, modifier_regid="BezierSpline"):
        modifier = FakeDeletableModifierTool(
            modifier_regid, name="Transform_LabSize", keyframes=keyframes,
            raise_on_delete_keyframes=raise_on_delete_keyframes,
            corrupt_after_delete=corrupt_after_delete,
        )
        output = FakeOutput(tool=modifier)
        inp = FakeInput(connected_output=output)
        tool = FakeTool({"Size": inp})
        comp = FakeComp({"Transform1": tool})
        return comp, modifier

    def test_deletes_intermediate_frame(self):
        comp, modifier = self._deletable_bezier(self._fresh_curve())
        out = _dispatch_delete(comp, {
            "tool_name": "Transform1", "input_name": "Size", "time": 30.0,
        })
        self.assertTrue(out.get("success"), out)
        self.assertEqual(modifier.delete_keyframes_calls, [30.0])
        self.assertNotIn(30.0, modifier._keyframes)
        by_time = {pt["time"]: pt for pt in out["points"]}
        self.assertEqual(set(by_time), {0.0, 60.0})
        self.assertAlmostEqual(by_time[0.0]["value"], 0.5, places=9)
        self.assertAlmostEqual(by_time[60.0]["value"], 1.5, places=9)

    def test_unknown_frame_rejected_before_write(self):
        comp, modifier = self._deletable_bezier(self._fresh_curve())
        out = _dispatch_delete(comp, {
            "tool_name": "Transform1", "input_name": "Size", "time": 45.0,
        })
        self.assertIn("error", out)
        self.assertEqual(out["error"]["code"], "UNKNOWN_KEYFRAME")
        self.assertEqual(modifier.delete_keyframes_calls, [])  # never wrote

    def test_deletekeyframes_exception_is_failure(self):
        comp, modifier = self._deletable_bezier(
            self._fresh_curve(), raise_on_delete_keyframes=RuntimeError("boom"),
        )
        out = _dispatch_delete(comp, {
            "tool_name": "Transform1", "input_name": "Size", "time": 30.0,
        })
        self.assertIn("error", out)
        self.assertEqual(out["error"]["code"], "DELETEKEYFRAMES_FAILED")

    def test_readback_frame_still_present_is_failure(self):
        # Simulate DeleteKeyFrames() being a no-op -- the requested frame is
        # still there afterwards.
        comp, modifier = self._deletable_bezier(
            self._fresh_curve(), corrupt_after_delete=self._fresh_curve(),
        )
        out = _dispatch_delete(comp, {
            "tool_name": "Transform1", "input_name": "Size", "time": 30.0,
        })
        self.assertIn("error", out)
        self.assertEqual(out["error"]["code"], "READBACK_MISMATCH")

    def test_readback_wrong_frame_deleted_is_failure(self):
        # Frame 30 (requested) is still present; frame 0 (not requested)
        # disappeared instead.
        wrong = self._fresh_curve()
        del wrong[0.0]
        comp, modifier = self._deletable_bezier(self._fresh_curve(), corrupt_after_delete=wrong)
        out = _dispatch_delete(comp, {
            "tool_name": "Transform1", "input_name": "Size", "time": 30.0,
        })
        self.assertIn("error", out)
        self.assertEqual(out["error"]["code"], "READBACK_MISMATCH")

    def test_readback_value_changed_on_remaining_frame_is_failure(self):
        # Frame 30 correctly gone, but frame 60's value was corrupted too.
        corrupted = copy.deepcopy(THREE_KEYFRAME_RAW_CURVE_AFTER_DELETE_30)
        corrupted[60.0][1] = 999.0
        comp, modifier = self._deletable_bezier(self._fresh_curve(), corrupt_after_delete=corrupted)
        out = _dispatch_delete(comp, {
            "tool_name": "Transform1", "input_name": "Size", "time": 30.0,
        })
        self.assertIn("error", out)
        self.assertEqual(out["error"]["code"], "READBACK_MISMATCH")

    def test_legitimate_handle_recalculation_does_not_fail(self):
        # This is the Gate A finding: deleting frame 30 legitimately changes
        # the RH of frame 0 and the LH of frame 60. Must NOT be treated as
        # a read-back mismatch.
        comp, modifier = self._deletable_bezier(
            self._fresh_curve(),
            corrupt_after_delete=copy.deepcopy(THREE_KEYFRAME_RAW_CURVE_AFTER_DELETE_30),
        )
        out = _dispatch_delete(comp, {
            "tool_name": "Transform1", "input_name": "Size", "time": 30.0,
        })
        self.assertTrue(out.get("success"), out)
        by_time = {pt["time"]: pt for pt in out["points"]}
        self.assertAlmostEqual(by_time[0.0]["rh"]["time_offset"], 20.0, places=9)
        self.assertAlmostEqual(by_time[60.0]["lh"]["time_offset"], -20.0, places=9)

    def test_modifier_still_rejected_when_neither_bezierspline_nor_path(self):
        # Was originally written with modifier_regid="Path" back when Path
        # was categorically unsupported. Path now has its own branch (see
        # DeleteKeyframePathPointTests below), so this must exercise a
        # genuinely unsupported modifier instead -- distinct from
        # test_unknown_modifier_type_rejected's own placeholder name, so the
        # two tests aren't byte-identical.
        comp, modifier = self._deletable_bezier(
            {0.0: {1: [0.5, 0.5]}, 30.0: {1: [1.0, 1.0]}}, modifier_regid="CustomModifier",
        )
        out = _dispatch_delete(comp, {
            "tool_name": "Transform1", "input_name": "Size", "time": 30.0,
        })
        self.assertIn("error", out)
        self.assertEqual(out["error"]["code"], "UNSUPPORTED_MODIFIER")
        self.assertEqual(modifier.delete_keyframes_calls, [])

    def test_unknown_modifier_type_rejected(self):
        comp, modifier = self._deletable_bezier(self._fresh_curve(), modifier_regid="SomeOtherModifier")
        out = _dispatch_delete(comp, {
            "tool_name": "Transform1", "input_name": "Size", "time": 30.0,
        })
        self.assertIn("error", out)
        self.assertEqual(out["error"]["code"], "UNSUPPORTED_MODIFIER")
        self.assertEqual(out["error"]["state"]["modifier_type"], "SomeOtherModifier")
        self.assertEqual(modifier.delete_keyframes_calls, [])


class DeleteKeyframePathPointTests(unittest.TestCase):
    """delete_keyframe v1 Path branch: delete one EXISTING keyframe from a
    Path modifier animating a Point input, using the natively-validated
    mechanism Input[time] = None (no DeleteKeyFrames() equivalent exists on
    Path -- see pruebas/keyframe-path-delete-lab-01/resultado.md). These
    tests exercise the wrapper's own logic against a fake bridge; Gate A
    already validated the native mechanism itself against live Resolve."""

    def _deletable_path(self, times=None, values=None, input_type="Point",
                         raise_on_setitem=None, connected_output_after=_UNSET,
                         corrupt_times_after=None, corrupt_value_at=None):
        times = sorted(PATH_THREE_KEYFRAME_VALUES) if times is None else times
        values = copy.deepcopy(PATH_THREE_KEYFRAME_VALUES) if values is None else values
        modifier = FakePathModifierTool(times)
        output = FakeOutput(tool=modifier)
        inp = FakePathInput(
            connected_output=output, values=values,
            raise_on_setitem=raise_on_setitem,
            connected_output_after=connected_output_after,
            corrupt_times_after=corrupt_times_after,
            corrupt_value_at=corrupt_value_at,
        )
        if input_type != "Point":
            inp.GetAttrs = lambda: {"INPS_DataType": input_type}
        tool = FakeTool({"Center": inp})
        comp = FakeComp({"Transform1": tool})
        return comp, modifier, inp

    # --- A: PASS ---
    def test_deletes_intermediate_frame(self):
        comp, modifier, inp = self._deletable_path()
        out = _dispatch_delete(comp, {
            "tool_name": "Transform1", "input_name": "Center", "time": 30.0,
        })
        self.assertTrue(out.get("success"), out)
        self.assertEqual(inp.setitem_calls, [(30.0, None)])  # single write
        self.assertEqual(modifier._times, [0.0, 60.0])
        by_time = {pt["time"]: pt for pt in out["points"]}
        self.assertEqual(set(by_time), {0.0, 60.0})
        self.assertEqual(by_time[0.0]["value"], {1: 0.2, 2: 0.2, 3: 0.0})
        self.assertEqual(by_time[60.0]["value"], {1: 0.8, 2: 0.2, 3: 0.0})
        for pt in out["points"]:
            self.assertNotIn("lh", pt)
            self.assertNotIn("rh", pt)
            self.assertNotIn("flags", pt)

    # --- B: frame inexistente ---
    def test_unknown_frame_rejected_before_write(self):
        comp, modifier, inp = self._deletable_path()
        out = _dispatch_delete(comp, {
            "tool_name": "Transform1", "input_name": "Center", "time": 45.0,
        })
        self.assertIn("error", out)
        self.assertEqual(out["error"]["code"], "UNKNOWN_KEYFRAME")
        self.assertEqual(inp.setitem_calls, [])

    # --- C: modifier Path sobre input no-Point ---
    def test_non_point_input_rejected(self):
        comp, modifier, inp = self._deletable_path(input_type="Number")
        out = _dispatch_delete(comp, {
            "tool_name": "Transform1", "input_name": "Center", "time": 30.0,
        })
        self.assertIn("error", out)
        self.assertEqual(out["error"]["code"], "UNSUPPORTED_MODIFIER")
        self.assertEqual(inp.setitem_calls, [])

    # --- D: excepcion durante inp[frame] = None ---
    def test_assignment_exception_is_failure(self):
        comp, modifier, inp = self._deletable_path(raise_on_setitem=RuntimeError("boom"))
        out = _dispatch_delete(comp, {
            "tool_name": "Transform1", "input_name": "Center", "time": 30.0,
        })
        self.assertIn("error", out)
        self.assertEqual(out["error"]["code"], "INPUT_KEYFRAME_DELETE_FAILED")
        self.assertFalse(out.get("success"))
        self.assertEqual(modifier._times, [0.0, 30.0, 60.0])  # untouched

    # --- E: read-back incorrecto ---
    def test_readback_frame_still_present_is_failure(self):
        # Simulate the assignment being a no-op -- times unchanged.
        comp, modifier, inp = self._deletable_path(corrupt_times_after=[0.0, 30.0, 60.0])
        out = _dispatch_delete(comp, {
            "tool_name": "Transform1", "input_name": "Center", "time": 30.0,
        })
        self.assertIn("error", out)
        self.assertEqual(out["error"]["code"], "READBACK_MISMATCH")

    def test_readback_new_frame_appears_is_failure(self):
        # Count is right (2), but the surviving frame at 60 was replaced by
        # an unexpected frame at 45 instead.
        comp, modifier, inp = self._deletable_path(corrupt_times_after=[0.0, 45.0])
        out = _dispatch_delete(comp, {
            "tool_name": "Transform1", "input_name": "Center", "time": 30.0,
        })
        self.assertIn("error", out)
        self.assertEqual(out["error"]["code"], "READBACK_MISMATCH")

    def test_readback_value_changed_on_remaining_frame_is_failure(self):
        # Frame 30 correctly gone, but frame 60's value was corrupted too.
        comp, modifier, inp = self._deletable_path(
            corrupt_value_at={60.0: {1: 999.0, 2: 0.2, 3: 0.0}},
        )
        out = _dispatch_delete(comp, {
            "tool_name": "Transform1", "input_name": "Center", "time": 30.0,
        })
        self.assertIn("error", out)
        self.assertEqual(out["error"]["code"], "READBACK_MISMATCH")

    def test_readback_modifier_disconnected_is_failure(self):
        # Input lost its connected modifier entirely after the write.
        comp, modifier, inp = self._deletable_path(connected_output_after=None)
        out = _dispatch_delete(comp, {
            "tool_name": "Transform1", "input_name": "Center", "time": 30.0,
        })
        self.assertIn("error", out)
        self.assertEqual(out["error"]["code"], "READBACK_MISMATCH")

    def test_readback_modifier_type_changed_is_failure(self):
        # Connected modifier is no longer Path after the write.
        other_modifier = FakeModifierTool("SomeOtherModifier", name="Mod1")
        comp, modifier, inp = self._deletable_path(
            connected_output_after=FakeOutput(tool=other_modifier),
        )
        out = _dispatch_delete(comp, {
            "tool_name": "Transform1", "input_name": "Center", "time": 30.0,
        })
        self.assertIn("error", out)
        self.assertEqual(out["error"]["code"], "READBACK_MISMATCH")


if __name__ == "__main__":
    unittest.main()
