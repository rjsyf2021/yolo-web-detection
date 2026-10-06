"""基准任务调度及报告回归测试；不加载真实 GPU 或引擎。"""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import benchmark_fps as benchmark


class BenchmarkFailureTests(unittest.TestCase):
    def run_benchmark(self, *, failure=None, cuda_error=None, frame_error=False,
                      variants=('s',), sizes=('1280x720', '1920x1080', '960x720', '1440x1080'),
                      cleanup_error=None):
        calls = []
        cuda = SimpleNamespace(
            is_available=lambda: True, get_device_name=lambda _: 'test GPU',
            reset_peak_memory_stats=lambda _: None, max_memory_allocated=lambda _: 0,
            synchronize=Mock(side_effect=cuda_error), empty_cache=Mock(side_effect=cleanup_error),
        )
        factory = Mock(side_effect=lambda path, **kwargs: SimpleNamespace(key=Path(path).stem))

        def frame(source, size):
            if frame_error and size == (1920, 1080):
                raise ValueError('cannot construct source frame')
            width, height = size or (source.shape[1], source.shape[0])
            return SimpleNamespace(shape=(height, width, 3))

        def measure(model, image, *args):
            calls.append((model.key, image.shape[1], image.shape[0]))
            if failure and image.shape[1] == 1920:
                raise failure
            return [10.0], [dict(preprocess=2, inference=6, postprocess=2, drawing=0)]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for variant in variants:
                (root / f'yolo26{variant}_640x640.engine').touch()
            output = root / 'report.json'
            argv = ['benchmark_fps.py', '--dir', directory, '--json', str(output),
                    '--source-sizes', *sizes]
            modules = {'torch': SimpleNamespace(cuda=cuda),
                       'ultralytics': SimpleNamespace(YOLO=factory)}
            with (patch.dict(sys.modules, modules), patch.object(sys, 'argv', argv),
                  patch.object(benchmark, 'load_source', return_value=(SimpleNamespace(shape=(600, 800, 3)), 'local')),
                  patch.object(benchmark, 'source_frame', side_effect=frame),
                  patch.object(benchmark, 'measure', side_effect=measure),
                  contextlib.redirect_stdout(io.StringIO())):
                code = 0
                try:
                    benchmark.main()
                except SystemExit as exc:
                    code = exc.code
            return code, json.loads(output.read_text()), calls, factory.call_count

    def test_size_failure_continues_with_new_model(self):
        code, report, calls, models = self.run_benchmark(failure=RuntimeError('OOM'))
        self.assertEqual(code, 1)
        self.assertEqual([width for _, width, _ in calls], [1280, 1920, 960, 1440])
        self.assertEqual(models, 2)
        rows = report['results']
        self.assertEqual([row['status'] for row in rows], ['ok', 'error', 'ok', 'ok'])
        self.assertEqual((rows[1]['source_w'], rows[1]['source_h']), (1920, 1080))
        self.assertEqual(rows[1]['error'], 'OOM')

    def test_source_construction_failure_has_dimensions(self):
        code, report, calls, _ = self.run_benchmark(frame_error=True)
        self.assertEqual(code, 1)
        self.assertEqual(len(report['results']), 4)
        self.assertEqual(report['results'][1]['source_w'], 1920)
        self.assertEqual([width for _, width, _ in calls], [1280, 960, 1440])

    def test_broken_cuda_skips_remaining_sizes_and_models(self):
        code, report, calls, models = self.run_benchmark(
            failure=RuntimeError('inference failed'), cuda_error=RuntimeError('illegal memory access'),
            variants=('s', 'm'))
        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(models, 1)
        self.assertEqual(len(report['results']), 8)
        self.assertTrue(all(row['status'] == 'skipped' for row in report['results'][2:]))
        self.assertEqual(report['results'][1]['gpu_error'], 'illegal memory access')

    def test_original_dimensions_and_success_exit(self):
        code, report, calls, _ = self.run_benchmark(sizes=('original',))
        self.assertEqual(code, 0)
        self.assertEqual(calls[0][1:], (800, 600))
        self.assertEqual(report['results'][0]['fps_mean'], 100)

    def test_cleanup_failure_preserves_report_and_fails_exit(self):
        code, report, _, _ = self.run_benchmark(sizes=('original',), cleanup_error=RuntimeError('CUDA failure'))
        self.assertEqual(code, 1)
        self.assertEqual(report['results'][0]['status'], 'ok')
        self.assertEqual(report['cleanup_errors'][0]['error'], 'CUDA failure')


if __name__ == '__main__':
    unittest.main()
