"""Exercise pipeline ordering and device propagation without starting model training."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FAKE_PYTHON = '''#!/usr/bin/env python3
import os, pathlib, sys
args = sys.argv[1:]
with open(os.environ['MOCK_LOG'], 'a') as log:
    log.write(pathlib.Path(args[0]).name + '|' + os.environ['CUDA_VISIBLE_DEVICES'] + '|' + args[args.index('--fsdp-devices') + 1] + '\\n')
if 'stage_a' in args[0]:
    if os.environ.get('MOCK_FAIL') == '1': sys.exit(7)
    if os.environ.get('MOCK_MISSING') != '1':
        output = pathlib.Path(args[args.index('--output-dir') + 1])
        run = args[args.index('--run-name') + 1]
        path = output / run / '10000' / 'params'
        path.mkdir(parents=True)
        (path / '_METADATA').write_text('test')
        (output / run / 'config.json').write_text('{}')
'''


class PipelineTest(unittest.TestCase):
    def run_case(self, task, *, extra=None, env_extra=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / 'python'
            fake.write_text(FAKE_PYTHON)
            fake.chmod(0o755)
            log = root / 'calls'
            env = os.environ | {'NO_PLAN_PYTHON': str(fake), 'NO_PLAN_OUTPUT_ROOT': str(root / 'models'),
                                'MOCK_LOG': str(log)} | (env_extra or {})
            result = subprocess.run(
                [str(ROOT / f'scripts/train_{task}_no_plan.sh'), '--gpus', '2,3', '--fsdp', '2', *(extra or [])],
                env=env, text=True, capture_output=True, check=False,
            )
            return result, log.read_text().splitlines() if log.exists() else []

    def test_three_tasks_run_stage_a_then_b(self):
        for task in ('book', 'sponge', 'bottle'):
            with self.subTest(task=task):
                result, lines = self.run_case(task)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(lines, ['train_vla_stage_a_no_plan.py|2,3|2',
                                         'train_vla_multitask_no_plan.py|2,3|2'])

    def test_failure_and_missing_checkpoint_do_not_start_b(self):
        for env_extra in ({'MOCK_FAIL': '1'}, {'MOCK_MISSING': '1'}):
            result, lines = self.run_case('book', env_extra=env_extra)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(lines, ['train_vla_stage_a_no_plan.py|2,3|2'])

    def test_print_only_and_invalid_fsdp(self):
        result, lines = self.run_case('book', extra=['--dry-run'])
        self.assertEqual(result.returncode, 0)
        self.assertEqual(lines, [])
        result, lines = self.run_case('book', extra=['--fsdp', '3'])
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(lines, [])


if __name__ == '__main__':
    unittest.main()
