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


def _dispatch(comp, params):
    with mock.patch.object(s, "_resolve_fusion_comp", return_value=(comp, None)):
        return s.fusion_comp("get_spline_curve", params)


def _dispatch_set(comp, params):
    with mock.patch.object(s, "_resolve_fusion_comp", return_value=(comp, None)):
        return s.fusion_comp("set_spline_handles", params)


# Raw shape captured empirically from BezierSpline.GetKeyFrames() against a
# real 2-keyframe linear curve (Transform_Lab.Size, frame 0=0.5, frame 60=1.5),
# see pruebas/keyframe-easing-lab-01/resultado_validacion_nativa.md:
#   {0.0: {1: 0.5, 'RH': {1: 20.0, 2: 0.333...}},
#    60.0: {1: 1.5, 'LH': {1: -20.0, 2: -0.333...}}}
REAL_RAW_CURVE = {
    0.0: {1: 0.5, "RH": {1: 20.0, 2: 0.33333333333333326}},
    60.0: {1: 1.5, "LH": {1: -20.0, 2: -0.3333333333333335}},
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


if __name__ == "__main__":
    unittest.main()
