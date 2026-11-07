import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
EXECUTOR = ROOT / "runs/run_frozen_final_evaluation.py"


class FrozenFinalExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.parent = self.root / "parent.txt"
        self.parent.write_text("bound parent")
        self.logical = self.root / "local/cache"
        self.external = self.root / "external/cache"
        self.output = self.root / "execution"
        self.marker = self.root / "command_ran.txt"
        self.protocol = self.root / "protocol.json"
        self.plan = {
            "status": "FROZEN_BEFORE_FINAL_METRICS",
            "preflight_files": {str(self.parent): hashlib.sha256(self.parent.read_bytes()).hexdigest()},
            "storage_routes": [{"logical": str(self.logical), "external": str(self.external)}],
            "commands": [{"id": "first", "argv": [sys.executable, "-c", f"from pathlib import Path; Path({str(self.marker)!r}).write_text('ran')"],
                          "env": {"PYTHONPATH": str(self.root), "PYTHONDONTWRITEBYTECODE": "1"}, "log": str(self.root / "logs/first.log")}],
            "execution_output": str(self.output),
        }

    def execute(self):
        self.protocol.write_text(json.dumps(self.plan))
        return subprocess.run([sys.executable, str(EXECUTOR), "--protocol", str(self.protocol)],
                              cwd=ROOT, capture_output=True, text=True)

    def assert_no_execution(self):
        self.assertFalse(self.marker.exists())
        self.assertFalse(self.output.exists())
        self.assertFalse(os.path.lexists(self.logical))

    def test_prospective_rejected_before_opening_missing_parent_or_creating_output(self):
        self.plan["status"] = "PROSPECTIVE"
        self.plan["preflight_files"] = {str(self.root / "does_not_exist"): "missing"}
        result = self.execute()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FROZEN_BEFORE_FINAL_METRICS", result.stderr)
        self.assertNotIn("FileNotFoundError", result.stderr)
        self.assert_no_execution()
        self.assertFalse(self.external.exists())

    def test_preflight_mismatch_creates_no_output_or_route(self):
        self.parent.write_text("changed parent")
        result = self.execute()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("preflight", result.stderr)
        self.assert_no_execution()
        self.assertFalse(self.external.exists())

    def test_fresh_external_route_exact_arguments_environment_and_serial_commands(self):
        self.root.joinpath("probe.py").write_text("VALUE = 'from frozen namespace'\n")
        argument = "exact space $value `literal`"
        first = ("import json, os, sys, probe; from pathlib import Path; "
                 "Path(sys.argv[1]).write_text(json.dumps({'argument':sys.argv[2], 'namespace':probe.VALUE, "
                 "'cwd':os.getcwd(), 'no_bytecode':os.environ['PYTHONDONTWRITEBYTECODE'], 'inherited_path':os.environ['PATH']})); "
                 "print('first stdout'); print('first stderr', file=sys.stderr)")
        self.plan["commands"][0]["argv"] = [sys.executable, "-c", first, str(self.logical / "feature.json"), argument]
        second = "from pathlib import Path; import sys; assert Path(sys.argv[1]).exists(); Path(sys.argv[2]).write_text('second')"
        self.plan["commands"].append({"id": "second", "argv": [sys.executable, "-c", second, str(self.logical / "feature.json"), str(self.marker)],
                                      "env": self.plan["commands"][0]["env"], "log": str(self.root / "logs/second.log")})
        result = self.execute()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.logical.is_symlink())
        self.assertEqual(self.logical.resolve(), self.external)
        actual = json.loads((self.external / "feature.json").read_text())
        self.assertEqual(actual, {"argument": argument, "namespace": "from frozen namespace", "cwd": str(ROOT),
                                  "no_bytecode": "1", "inherited_path": os.environ["PATH"]})
        self.assertEqual(self.marker.read_text(), "second")
        self.assertIn("first stdout", (self.root / "logs/first.log").read_text())
        self.assertIn("first stderr", (self.root / "logs/first.log").read_text())
        events = [json.loads(line) for line in (self.output / "commands.jsonl").read_text().splitlines()]
        self.assertEqual([(e["id"], e["event"]) for e in events], [("first", "STARTED"), ("first", "FINISHED"), ("second", "STARTED"), ("second", "FINISHED")])
        completion = json.loads((self.output / "completion.json").read_text())
        self.assertEqual(completion["status"], "COMPLETE")
        self.assertEqual(completion["protocol_sha256"], hashlib.sha256(self.protocol.read_bytes()).hexdigest())
        self.assertEqual(completion["executor_sha256_before"], hashlib.sha256(EXECUTOR.read_bytes()).hexdigest())
        self.assertEqual(completion["preflight_files_before"], completion["preflight_files_after"])

    def test_nonzero_process_is_recorded_and_propagated_without_complete_or_next_command(self):
        self.plan["commands"][0]["argv"] = [sys.executable, "-c", "raise SystemExit(7)"]
        self.plan["commands"].append({**self.plan["commands"][0], "id": "never", "argv": [sys.executable, "-c", f"from pathlib import Path; Path({str(self.marker)!r}).write_text('unexpected')"],
                                      "log": str(self.root / "logs/never.log")})
        result = self.execute()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CalledProcessError", result.stderr)
        events = [json.loads(line) for line in (self.output / "commands.jsonl").read_text().splitlines()]
        self.assertEqual(events[-1]["returncode"], 7)
        self.assertEqual([e["id"] for e in events], ["first", "first"])
        self.assertFalse(self.marker.exists())
        self.assertFalse((self.output / "completion.json").exists())

    def test_existing_leaf_fails_before_any_command_or_receipt(self):
        self.external.mkdir(parents=True)
        result = self.execute()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FileExistsError", result.stderr)
        self.assert_no_execution()

    def test_command_changing_bound_parent_cannot_complete(self):
        self.plan["commands"][0]["argv"] = [sys.executable, "-c", f"from pathlib import Path; Path({str(self.parent)!r}).write_text('changed during run')"]
        result = self.execute()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("changed during", result.stderr)
        self.assertFalse((self.output / "completion.json").exists())


if __name__ == "__main__":
    unittest.main()
