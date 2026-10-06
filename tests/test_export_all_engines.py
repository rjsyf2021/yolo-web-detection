"""导出计划、日志与文件替换回归测试；不加载 GPU 依赖。"""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import export_all_engines as exporter


class ExportTests(unittest.TestCase):
    def setUp(self):
        # unittest cleanup owns this directory across setUp and the test method.
        self.directory = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.base_patch = patch.object(exporter, 'BASE', self.base)
        self.base_patch.start()
        self.addCleanup(self.base_patch.stop)
        for variant in ('s', 'm'):
            (self.base / f'yolo26{variant}.pt').write_bytes(b'weights')
        self.target = self.base / 'yolo26s_640x640.engine'
        self.target.write_bytes(b'old-engine')

    def cli(self, *args):
        output, errors = io.StringIO(), io.StringIO()

        def preflight(command, **kwargs):
            self.assertEqual(command[2:], ['--check-env'])
            # 执行子进程的真实参数分支，仅替换 GPU 检查。
            with patch.object(sys, 'argv', command[1:]):
                exporter.main()
            return SimpleNamespace(returncode=0)

        with (patch.object(sys, 'argv', ['export_all_engines.py', *args]),
              patch.object(exporter, 'check_environment') as check,
              patch.object(exporter.subprocess, 'run', side_effect=preflight),
              patch.object(exporter, 'run_worker', return_value=0) as worker,
              contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors)):
            code = 0
            try:
                exporter.main()
            except SystemExit as exc:
                code = exc.code
        return code, output.getvalue(), errors.getvalue(), check, worker

    def test_partial_export_reports_skip_once(self):
        code, output, _, check, worker = self.cli()
        self.assertEqual(code, 0)
        self.assertEqual(output.count('[跳过]'), 1)
        check.assert_called_once()
        self.assertEqual(worker.call_args.args[4], 'm')

    def test_all_existing_engines_skip_without_preflight(self):
        code, output, _, check, worker = self.cli('--models', 's')
        self.assertEqual(code, 0)
        self.assertEqual(output.count('[跳过]'), 1)
        check.assert_not_called()
        worker.assert_not_called()

    def test_selected_and_overwrite_exports_do_not_report_unrelated_skips(self):
        for args in (('--models', 'm'), ('--models', 's', '--overwrite')):
            with self.subTest(args=args):
                code, output, _, check, worker = self.cli(*args)
                self.assertEqual(code, 0)
                self.assertNotIn('[跳过]', output)
                check.assert_called_once()
                worker.assert_called_once()

    def test_check_env_does_not_inspect_engine_targets(self):
        self.target.write_bytes(b'')
        with patch.object(exporter, 'engine_file_state', side_effect=AssertionError('scanned targets')):
            code, output, _, check, worker = self.cli('--check-env')
        self.assertEqual(code, 0)
        self.assertEqual(output, '')
        check.assert_called_once()
        worker.assert_not_called()

    def test_empty_target_rejected_by_parent_and_worker(self):
        self.target.write_bytes(b'')
        code, output, errors, check, worker = self.cli('--models', 's')
        self.assertNotEqual(code, 0)
        self.assertNotIn('[跳过]', output)
        self.assertIn('--overwrite', errors)
        check.assert_not_called()
        worker.assert_not_called()
        with self.assertRaisesRegex(ValueError, '空文件'):
            exporter.export_one(640, 640)
        self.assertEqual(self.target.read_bytes(), b'')

    def test_empty_target_can_be_explicitly_rebuilt(self):
        self.target.write_bytes(b'')
        code, output, _, check, worker = self.cli('--models', 's', '--overwrite')
        self.assertEqual(code, 0)
        self.assertNotIn('[跳过]', output)
        check.assert_called_once()
        self.assertTrue(worker.call_args.args[2])
        with patch.object(exporter, 'run_stage', side_effect=self.stage):
            exporter.export_one(640, 640, overwrite=True)
        self.assertEqual(self.target.read_bytes(), b'new-engine')

    def test_directory_target_rejected_even_with_overwrite(self):
        self.target.unlink()
        self.target.mkdir()
        for overwrite in (False, True):
            with self.subTest(overwrite=overwrite):
                args = ['--models', 's'] + (['--overwrite'] if overwrite else [])
                code, _, errors, check, worker = self.cli(*args)
                self.assertNotEqual(code, 0)
                self.assertIn('不是普通文件', errors)
                check.assert_not_called()
                worker.assert_not_called()
                with self.assertRaisesRegex(ValueError, '不是普通文件'):
                    exporter.export_one(640, 640, overwrite=overwrite)

    def test_list_distinguishes_empty_target(self):
        self.target.write_bytes(b'')
        code, output, _, check, worker = self.cli('--list', '--models', 's')
        self.assertEqual(code, 0)
        self.assertIn('无效：空文件', output)
        check.assert_not_called()
        worker.assert_not_called()

    def stage(self, name, directory, *args):
        if name == 'engine':
            (Path(directory) / 'yolo26s.engine').write_bytes(b'new-engine')

    def test_stage_failure_preserves_old_engine_and_cleans_temp_files(self):
        for failed_stage in ('onnx', 'reference', 'engine'):
            with self.subTest(stage=failed_stage):
                def stage(name, directory, *args, failed_stage=failed_stage):
                    self.stage(name, directory, *args)
                    if name == failed_stage:
                        raise RuntimeError('stage failed')
                with patch.object(exporter, 'run_stage', side_effect=stage):
                    with self.assertRaisesRegex(RuntimeError, 'stage failed'):
                        exporter.export_one(640, 640, overwrite=True)
                self.assertEqual(self.target.read_bytes(), b'old-engine')
                self.assertFalse(list(self.base.glob('.yolo-export-*')))


if __name__ == '__main__':
    unittest.main()
