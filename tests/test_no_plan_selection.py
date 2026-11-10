import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from tactile_vla.vla.no_plan_selection import select_factual


class NoPlanSelectionTest(unittest.TestCase):
    def test_excludes_need_and_end_counterfactuals_without_reordering(self):
        rows = [
            {'global_index': 10, 'need_variant': 'real'},
            {'global_index': 11, 'need_variant': 'rotation_none', 'need_counterfactual': {'changed': True}},
            {'global_index': 12, 'sample_variant': 'moderately_to_slightly'},
            {'global_index': 13},
        ]
        stream = {'manifest_row_indices': [0, 1, 2, 3], 'global_indices': [10, 11, 12, 13]}
        self.assertEqual(select_factual(rows, stream), ([0, 3], [10, 13]))

    def test_rejects_mismatched_or_empty_streams(self):
        rows = [{'global_index': 10, 'need_variant': 'rotation_none'}]
        with self.assertRaises(ValueError):
            select_factual(rows, {'manifest_row_indices': [0], 'global_indices': [11]})
        with self.assertRaises(ValueError):
            select_factual(rows, {'manifest_row_indices': [0], 'global_indices': [10]})

    def test_full_export_writes_no_delta_or_best(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/train_vla_multitask_no_plan.py'
        tree = ast.parse(script.read_text())
        export = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == 'export_full_params')

        class Checkpointer:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def save(self, path, values, *, force):
                self.asserted_values = values
                path.mkdir()
                (path / '_METADATA').write_text('test')

        namespace = {'base': SimpleNamespace(jax=SimpleNamespace(process_index=lambda: 0)),
                     'ocp': SimpleNamespace(PyTreeCheckpointer=Checkpointer),
                     'json': json, 'TASK': 'book', 'VERSION': 'v9_7'}
        exec(compile(ast.Module(body=[export], type_ignores=[]), str(script), 'exec'), namespace)
        state = SimpleNamespace(params=SimpleNamespace(to_pure_dict=lambda: {'w': 1}))
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            namespace['export_full_params'](root, state, 2000, None)
            self.assertTrue((root / '2000/full_params/_METADATA').is_file())
            self.assertFalse((root / '2000/delta_params').exists())
            self.assertFalse((root / 'best').exists())
            metadata = json.loads((root / '2000/book_v9_7_export.json').read_text())
            self.assertFalse(metadata['resume_supported'])


if __name__ == '__main__':
    unittest.main()
