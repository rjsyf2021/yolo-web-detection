"""真实编解码与音轨合成回归测试；推理替身不需要 CUDA 或模型。

运行：python -m unittest discover -s tests -p 'test_*.py' -v
依赖：应用的 numpy、opencv-python、av，以及 PATH 中的 ffmpeg。
"""
import ast
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from fractions import Fraction
import math
import os
from pathlib import Path
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import unittest


class PassthroughPool:
    def select(self, *args):
        return dict(key='test', input_w=640, input_h=640)

    def predict(self, image, *args, **kwargs):
        class Result:
            def plot(self):
                return image.copy()
        return Result(), [], dict(infer_ms=0, wait_ms=0, draw_ms=0)


class VideoTimestampTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which('ffmpeg'):
            raise unittest.SkipTest('需要 ffmpeg')
        try:
            import av
            import cv2
            import numpy as np
        except ImportError as exc:
            raise unittest.SkipTest(str(exc)) from exc
        cls.av = av
        # 单文件服务在顶层导入 GPU 软件栈；仅加载原始媒体函数，避免启动服务或模型。
        source = Path(__file__).resolve().parents[1] / 'app_ws-multi.py'
        names = {'oriented_bgr', 'output_size', 'ffmpeg_executable',
                 'open_video_decoder', 'open_video_encoder', 'decoded_prefetch', 'process_video'}
        tree = ast.parse(source.read_text(encoding='utf-8'))
        functions = [node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name in names]
        if {node.name for node in functions} != names:
            raise AssertionError('媒体函数入口发生变化，请更新测试加载器')
        namespace = dict(globals(), cv2=cv2, np=np, deque=deque, ThreadPoolExecutor=ThreadPoolExecutor,
                         contextmanager=contextmanager, Fraction=Fraction, math=math, os=os,
                         queue=queue, threading=threading, time=time)
        # Only execute selected functions from the checked-in application, never user input.
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)  # pylint: disable=exec-used
        cls.process_video = staticmethod(namespace['process_video'])

    def check_timeline(self, video_start, audio_offset):
        with tempfile.TemporaryDirectory(prefix='yolo-timestamps-') as directory:
            source = Path(directory) / 'source.mkv'
            target = Path(directory) / 'result.mp4'
            # PCM 避免输入 AAC 编码延迟；声音持续到视频结束。
            subprocess.run([
                'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                '-f', 'lavfi', '-i', 'color=c=blue:s=160x120:r=10:d=2',
                '-itsoffset', str(audio_offset), '-f', 'lavfi', '-i',
                f'sine=frequency=440:sample_rate=48000:duration={2-audio_offset}',
                '-c:v', 'libx264', '-c:a', 'pcm_s16le',
                '-output_ts_offset', str(video_start), str(source),
            ], check=True, capture_output=True)
            with self.av.open(str(source)) as container:
                video = container.streams.video[0]
                audio = container.streams.audio[0]
                self.assertAlmostEqual(float(video.start_time * video.time_base), video_start, places=2)
                self.assertAlmostEqual(float(audio.start_time * audio.time_base),
                                       video_start + audio_offset, places=2)
            metadata = self.process_video(
                source, target, PassthroughPool(),
                dict(engine='auto', profile='balanced', output='original', conf=.4),
                lambda **values: None,
            )
            self.assertTrue(metadata['audio'])
            self.assertEqual(metadata['frames'], 20)
            with self.av.open(str(target)) as container:
                frames = list(container.decode(video=0))
                self.assertEqual(len(frames), 20)
                self.assertAlmostEqual(float(frames[0].pts * frames[0].time_base), 0, places=2)
                self.assertAlmostEqual(float(frames[-1].pts * frames[-1].time_base), 1.9, places=2)
            with self.av.open(str(target)) as container:
                frames = list(container.decode(audio=0))
                self.assertTrue(frames, '输出音轨不可为空')
                start = float(frames[0].pts * frames[0].time_base)
                end = float(frames[-1].pts * frames[-1].time_base) + frames[-1].samples / frames[-1].sample_rate
                # AAC 每帧 1024 个采样点，允许两个音频帧的编码边界误差。
                self.assertAlmostEqual(start, max(0, audio_offset), delta=.05)
                self.assertAlmostEqual(end, 2, delta=.05)
                self.assertGreater(max(abs(frame.to_ndarray()).max() for frame in frames), .01)

    def test_zero_start(self):
        self.check_timeline(0, 0)

    def test_nonzero_start(self):
        self.check_timeline(5, 0)

    def test_audio_starts_after_video(self):
        self.check_timeline(5, .5)

    def test_audio_starts_before_video(self):
        self.check_timeline(5, -.5)


if __name__ == '__main__':
    unittest.main()
