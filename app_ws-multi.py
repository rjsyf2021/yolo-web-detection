#!/usr/bin/env python3
# YOLO 双流水线服务，单文件独立版本。
# 使用 ~/yolo_env/bin/python 启动；已有 .engine 文件放在本脚本同目录。
# 运行依赖 CUDA、TensorRT 引擎；WebRTC 通道还需要先启动 MediaMTX。

"""共享 GPU 引擎池；文件处理与实时检测分别管理各自的输入、输出流水线。"""
from pathlib import Path
import gc
import math
import re
import threading
import time
from collections import OrderedDict

import numpy as np
import torch
from ultralytics import YOLO


# 共享引擎池：通过同一把锁串行访问模型，使用最近最少使用策略管理缓存。
class EnginePool:
    def __init__(self, directory, cache_size=3):
        self.lock = threading.Lock()
        self.cache = OrderedDict()
        self.cache_size = max(1, cache_size)
        self.engines = {}
        for path in sorted(Path(directory).glob('*.engine')):
            match = re.fullmatch(r'yolo26([nsmxl])_(\d+)x(\d+)\.engine', path.name)
            if not match:
                continue
            variant, width, height = match.groups()
            width, height = int(width), int(height)
            if min(width, height) <= 0:
                continue
            # 从文件名读取标称宽×高，按 32 步长推定输入尺寸；此处未读取引擎内部形状。
            ih, iw = ((height + 31) // 32 * 32, (width + 31) // 32 * 32)
            self.engines[path.stem] = dict(
                key=path.stem, path=str(path.resolve()), model=variant,
                w=width, h=height, input_w=iw, input_h=ih,
                label=f'{variant} · {width}×{height}',
            )
        if not self.engines:
            raise RuntimeError(f'在 {directory} 未找到 yolo26s_1440x1080.engine 这类引擎文件')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA 不可用，请在 GPU 服务器的原 YOLO 环境中启动')
        print('[engine] 发现引擎:', ', '.join(self.engines), flush=True)

    # 对外只暴露引擎描述信息，隐藏服务器上的模型文件路径。
    def catalog(self):
        return [{k: v for k, v in item.items() if k != 'path'}
                for item in self.engines.values()]

    def warm_all(self, runs=5, keep_all=True):
        """启动时预热所有发现的引擎，完成后再接收请求。"""
        if keep_all:
            self.cache_size = len(self.engines)
        report = dict(total=len(self.engines), runs=runs, warmed=[], failed={}, evicted=[])
        started = time.perf_counter()
        def attempt(image, engine):
            # 仅返回错误文本，以释放异常的回溯引用，
            # 避免重试时仍保留加载失败的模型和显存分配。
            try:
                for _ in range(runs):
                    self.predict(image, engine, .4, annotate=True)
                return None
            except Exception as exc:
                return str(exc)
        def evict_oldest():
            with self.lock:
                key, old = self.cache.popitem(last=False)
                del old
            gc.collect()
            torch.cuda.empty_cache()
            report['evicted'].append(key)
            print(f'[warmup] 显存不足/预留空间，移出常驻缓存: {key}', flush=True)
        for index, engine in enumerate(self.engines.values(), 1):
            key = engine['key']
            print(f'[warmup] {index}/{report["total"]} {key}: 开始 {runs} 次预热', flush=True)
            image = np.zeros((engine['h'], engine['w'], 3), dtype=np.uint8)
            while True:
                error = attempt(image, engine)
                if error is None:
                    report['warmed'].append(key)
                    print(f'[warmup] {key}: 完成', flush=True)
                    break
                with self.lock:
                    self.cache.pop(key, None)
                gc.collect()
                torch.cuda.empty_cache()
                out_of_memory = any(word in error.lower() for word in
                                    ('out of memory', 'outofmemory', 'cuda_error_out_of_memory'))
                if out_of_memory and self.cache:
                    evict_oldest()
                    continue
                report['failed'][key] = error
                print(f'[warmup] {key}: 失败（未标为就绪）: {error}', flush=True)
                break
            del image
            # 为 NVDEC/NVENC 和并发文件处理预留显存，
            # 避免空闲模型上下文占满可用空间。
            torch.cuda.empty_cache()
            while len(self.cache) > 1 and torch.cuda.mem_get_info()[0] < 1024**3:
                evict_oldest()
        if report['evicted']:
            self.cache_size = max(1, min(self.cache_size, len(self.cache)))
        report.update(resident=list(self.cache), cache_limit=self.cache_size,
                      elapsed=round(time.perf_counter()-started, 2))
        self.warmup_report = report
        print(f'[warmup] 启动预热结束: 成功 {len(report["warmed"])}/{report["total"]}, '
              f'常驻 {len(self.cache)}, 失败 {len(report["failed"])}, 耗时 {report["elapsed"]}s', flush=True)
        if not report['warmed']:
            raise RuntimeError('所有模型预热失败，请查看以上模型/显存/导出尺寸错误')
        return report

    # 显式指定时直接返回该引擎；自动选择先匹配比例，再按策略比较目标分辨率。
    def select(self, width, height, requested='auto', profile='balanced'):
        if min(width, height) <= 0:
            raise ValueError('媒体宽高无效')
        if profile not in ('speed', 'balanced', 'detail'):
            raise ValueError('无效的检测策略')
        if requested != 'auto':
            if requested not in self.engines:
                raise ValueError('所选引擎不存在，请刷新引擎列表')
            return self.engines[requested]

        # 优先减少等比例缩放后的填充面积；允许覆盖率相差 4 个百分点以容纳步长取整，
        # 同时存在两种比例时，16:9 画面不会与 4:3 引擎并列为最佳比例。
        candidates = []
        for item in self.engines.values():
            scale = min(item['input_w'] / width, item['input_h'] / height)
            coverage = width * height * scale * scale / (item['input_w'] * item['input_h'])
            candidates.append((item, scale, coverage))
        best_coverage = max(c[2] for c in candidates)
        candidates = [c for c in candidates if c[2] >= best_coverage - .04]
        # 文件识别默认使用 detail，不设 1080p 上限；仍按原图大小选择，避免无意义放大。
        cap = {'speed': 720, 'balanced': 1080, 'detail': float('inf')}[profile]
        target_scale = min(1.0, cap / min(width, height))
        priority = {'s': 0, 'n': 1, 'm': 2, 'l': 3, 'x': 4}

        def score(candidate):
            item, scale, coverage = candidate
            ratio = scale / target_scale
            # 优先选择最接近目标的分辨率，并对不必要的放大增加惩罚。
            distance = abs(math.log(ratio)) * (1.3 if ratio > 1 else 1.0)
            return (distance + .15 * (best_coverage - coverage),
                    priority[item['model']], item['input_w'] * item['input_h'], item['key'])
        return min(candidates, key=score)[0]

    # 支持 BGR 数组或已预处理张量；返回图像（延后画框时为 Results，否则可为 None）、检测列表及耗时。
    def predict(self, bgr, engine, conf=.4, annotate=False, defer_plot=False):
        if not math.isfinite(conf) or not 0 <= conf <= 1:
            raise ValueError('置信度必须在 0 到 1 之间')
        key = engine['key']
        wait_start = time.perf_counter()
        with self.lock:
            wait_ms = (time.perf_counter() - wait_start) * 1000
            if key not in self.cache:
                if len(self.cache) >= self.cache_size:
                    _, old = self.cache.popitem(last=False)
                    del old
                    gc.collect()
                    torch.cuda.empty_cache()
                print(f'[engine] 首次加载 {key}', flush=True)
                self.cache[key] = YOLO(engine['path'], task='detect')
            model = self.cache[key]
            self.cache.move_to_end(key)
            started = time.perf_counter()
            try:
                result = model.predict(
                    bgr, imgsz=(engine['input_h'], engine['input_w']), rect=False,
                    conf=conf, device=0, verbose=False, stream=False,
                )[0]
                if annotate or defer_plot:
                    result = result.cpu()
                # 实时 JSON 仅需传回紧凑的检测框数据表，
                # 因此无需为每一帧复制完整的 Results 对象。
                packed_boxes = (result.boxes.data.detach().cpu().numpy()
                                if result.boxes is not None else None)
            except Exception as exc:
                self.cache.pop(key, None)
                raise RuntimeError(
                    f'{key} 推理失败（按文件名推定输入为 '
                    f'{engine["input_w"]}×{engine["input_h"]}，请核对导出尺寸）: {exc}'
                ) from exc
            infer_ms = (time.perf_counter() - started) * 1000
        # BGR 图像输入的检测框由 Ultralytics 映射回原图；张量输入另行还原坐标。
        detections = []
        if packed_boxes is not None:
            for row in packed_boxes:
                idx = int(row[-1])
                detections.append(dict(box=[float(x) for x in row[:4]], conf=float(row[-2]),
                                       cls=idx, name=result.names[idx]))
        draw_started = time.perf_counter()
        drawn = result.plot() if annotate else None
        draw_ms = (time.perf_counter() - draw_started) * 1000 if annotate else 0.
        if drawn is not None and drawn.shape[:2] != bgr.shape[:2]:
            raise RuntimeError('检测输出尺寸与原始帧不一致，已停止输出以避免比例错误')
        # CPU 上的 Results 保留当前图像和检测框，不依赖下一次推理；
        # 只有文件处理请求会把画框延后交给另一工作线程。
        return (result if defer_plot else drawn), detections, dict(infer_ms=round(infer_ms, 2), wait_ms=round(wait_ms, 2), draw_ms=round(draw_ms, 2))


"""完整文件上传流水线：服务器解码 → YOLO 推理 → 服务器画框。

视频采用源文件的显示时间戳（PTS），不以处理速度或实时采集帧率决定播放速度。
"""
from fractions import Fraction
from pathlib import Path
import math
import os
import shutil
import subprocess
import time
import queue
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import cv2
import numpy as np


# 图片处理保持原图输出尺寸；进度回调也用于检查任务是否被取消。
def process_image(source, target, pool, options, progress):
    raw = np.fromfile(source, dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('无法解码图片，请上传 JPEG、PNG 或 WebP 图片')
    height, width = image.shape[:2]
    engine = pool.select(width, height, options['engine'], options['profile'])
    progress(stage='detecting', source_w=width, source_h=height, engine=engine['key'])
    annotated, dets, timing = pool.predict(image, engine, options['conf'], annotate=True)
    ok, encoded = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError('图片编码失败')
    encoded.tofile(target)
    return dict(source_w=width, source_h=height, output_w=width, output_h=height,
                engine=engine['key'], input_w=engine['input_w'], input_h=engine['input_h'],
                detections=len(dets), frames=1, **timing)


# 把视频帧转为连续 BGR 数组，并应用像素宽高比和旋转元数据。
def oriented_bgr(frame, stream):
    image = frame.to_ndarray(format='bgr24')
    # 先校正非正方形像素，再旋转到浏览器显示方向。
    sar = stream.sample_aspect_ratio
    if sar and sar > 0 and sar != 1:
        image = cv2.resize(image, (max(1, round(image.shape[1] * float(sar))), image.shape[0]))
    angle = float(getattr(frame, 'rotation', 0) or 0)
    if not angle and stream.metadata.get('rotate'):
        angle = -float(stream.metadata['rotate'])
    quarters = round(angle / 90)
    if abs(angle - quarters * 90) > .1:
        raise ValueError('暂不支持非 90 度倍数的视频旋转，请先规范化视频方向')
    if quarters % 4:
        image = np.rot90(image, quarters % 4)
    return np.ascontiguousarray(image)


# 返回绘制尺寸和编码尺寸；两者的差异仅用于满足编码器的偶数宽高要求。
def output_size(width, height, engine, mode):
    if mode == 'nearest':
        # 保持源画面宽高比，缩小至选定引擎分辨率内，不放大原图。
        scale = min(1., engine['input_w'] / width, engine['input_h'] / height)
        width, height = max(2, round(width * scale)), max(2, round(height * scale))
    elif mode != 'original':
        raise ValueError('无效的输出尺寸模式')
    # H.264 的 yuv420p 格式要求宽高为偶数：每个方向最多补一像素边缘。
    return width, height, width + width % 2, height + height % 2


def ffmpeg_executable():
    # 优先使用系统 ffmpeg（例如 apt 安装的 /usr/bin/ffmpeg）。
    executable = shutil.which('ffmpeg')
    if executable:
        return executable
    # 仅当系统没有 ffmpeg 时才回退到可选的 imageio-ffmpeg。
    # 动态导入可选依赖；实际走到此分支且包未安装时，仍会抛出下方的明确错误。
    try:
        import importlib
        module = importlib.import_module('imageio_ffmpeg')
        return getattr(module, 'get_ffmpeg_exe')()
    except ImportError as exc:
        raise RuntimeError('保留视频音轨需要 ffmpeg；请在服务器安装 ffmpeg 或 imageio-ffmpeg') from exc


# 实际解码首帧以验证 NVDEC；失败时关闭容器，再尝试 CPU 解码。
def open_video_decoder(av, source):
    notes = []
    for hardware in (True, False):
        container = None
        try:
            kwargs = {}
            if hardware:
                from av.codec.hwaccel import HWAccel
                kwargs['hwaccel'] = HWAccel(device_type='cuda', allow_software_fallback=False)
            container = av.open(str(source), **kwargs)
            if not container.streams.video:
                raise ValueError('文件中没有视频轨道')
            stream = container.streams.video[0]
            if not hardware:
                stream.thread_type = 'AUTO'
            frames = iter(container.decode(stream))
            first = next(frames, None)
            if first is None:
                raise ValueError('视频没有可解码的帧')
            if hardware and not getattr(stream.codec_context, 'is_hwaccel', False):
                raise RuntimeError('PyAV 未确认硬件解码已启用')
            image = oriented_bgr(first, stream)
            return container, stream, frames, first, image, ('NVDEC → CPU BGR' if hardware else 'CPU'), notes
        except Exception as exc:
            if container is not None:
                container.close()
            if not hardware:
                raise
            notes.append('GPU 解码初始化回退：' + str(exc)[:350])
    raise RuntimeError('无法初始化视频解码')


# 优先 NVENC，初始化失败时回退软件编码；两者使用相同的输出时间基。
def open_video_encoder(av, path, fps, width, height, time_base):
    notes = []
    for codec in ('h264_nvenc', 'libx264'):
        output = None
        try:
            output = av.open(path, 'w', options={'movflags': '+faststart'})
            encoder = output.add_stream(codec, rate=fps)
            encoder.width, encoder.height = width, height
            encoder.pix_fmt = 'yuv420p'
            encoder.time_base = time_base
            encoder.codec_context.time_base = time_base
            encoder.codec_context.max_b_frames = 0
            encoder.options = ({'preset': 'p4', 'rc': 'constqp', 'qp': '20'} if codec == 'h264_nvenc'
                               else {'preset': 'veryfast', 'crf': '20', 'tune': 'zerolatency'})
            # 先初始化设备和编码器，确保在接收实际帧前发现初始化失败。
            output.start_encoding()
            return output, encoder, codec, notes
        except Exception as exc:
            if output is not None:
                try:
                    output.close()
                except Exception:
                    pass
            if codec == 'libx264':
                raise
            notes.append('NVENC 初始化回退：' + str(exc)[:350])
    raise RuntimeError('无法初始化视频编码')


# 用容量为 2 的队列预取解码帧；将生产线程异常传回消费方，退出时停止并回收线程。
@contextmanager
def decoded_prefetch(frames, stream):
    ready = queue.Queue(maxsize=2)
    stop = threading.Event()
    def put(item):
        while not stop.is_set():
            try:
                ready.put(item, timeout=.1)
                return
            except queue.Full:
                pass
    def produce():
        try:
            for frame in frames:
                if stop.is_set():
                    break
                put(('frame', (frame, oriented_bgr(frame, stream))))
        except Exception as exc:
            put(('error', exc))
        finally:
            put(('end', None))
    def consume():
        while True:
            kind, value = ready.get()
            if kind == 'end':
                return
            if kind == 'error':
                raise value
            yield value
    worker = threading.Thread(target=produce, name='video-decode', daemon=True)
    worker.start()
    try:
        yield consume()
    finally:
        stop.set()
        worker.join()


# 文件视频逐帧保序处理，优先使用源时间戳；画框、编码分线程执行，有音轨时再合并。
def process_video(source, target, pool, options, progress):
    try:
        import av
    except ImportError as exc:
        raise RuntimeError('请在服务器当前 Python 环境安装 av>=14') from exc
    intermediate = str(Path(target).with_name('annotated-silent.mp4'))
    started = time.perf_counter()
    progress(stage='decoding')
    container, stream, frames, first, first_image, decoder_name, hardware_notes = open_video_decoder(av, source)
    with container:
        fps = stream.average_rate or stream.guessed_rate or Fraction(30, 1)
        if not math.isfinite(float(fps)) or fps <= 0:
            fps = Fraction(30, 1)
        has_audio = bool(container.streams.audio)
        ffmpeg = ffmpeg_executable() if has_audio else None
        height, width = first_image.shape[:2]
        engine = pool.select(width, height, options['engine'], options['profile'])
        draw_w, draw_h, out_w, out_h = output_size(width, height, engine, options['output'])
        progress(stage='warming', engine=engine['key'], frames=0)
        warmup_started = time.perf_counter()
        warm_image = (cv2.resize(first_image, (draw_w, draw_h), interpolation=cv2.INTER_AREA)
                      if (draw_w, draw_h) != (width, height) else first_image)
        for _ in range(5):
            options.get('check_cancel', lambda: None)()
            pool.predict(warm_image, engine, options['conf'], annotate=False)
        warmup_ms = (time.perf_counter() - warmup_started) * 1000
        del warm_image
        time_base = stream.time_base or Fraction(1, 90000)
        origin = (first.pts * first.time_base) if first.pts is not None else Fraction(0)
        # 帧缺少时间戳时按帧率补算时长，并在结果中标记为 fallback。
        default_duration = max(1, round(Fraction(1, 1) / fps / time_base))
        timing_fallback = first.pts is None
        count = 0
        infer_total = 0.
        draw_total = 0.
        encode_total = 0.
        durations = {}
        last_end = 0
        progress(stage='detecting', source_w=width, source_h=height,
                 output_w=out_w, output_h=out_h, engine=engine['key'],
                 source_fps=str(fps), total=stream.frames or 0, frames=0)

        output, encoder, encoder_name, encoder_notes = open_video_encoder(av, intermediate, fps, out_w, out_h, time_base)
        hardware_notes.extend(encoder_notes)
        progress(video_decoder=decoder_name, video_encoder=encoder_name, drawing='CPU', video_pipeline='解码 / 推理 / 画框 / 编码四阶段流水线（有界队列）', hardware_notes=hardware_notes)
        with output, ThreadPoolExecutor(max_workers=1, thread_name_prefix='video-draw') as draw_worker, ThreadPoolExecutor(max_workers=1, thread_name_prefix='video-encode') as encode_worker:
            encode_pending = deque()
            processing_started = time.perf_counter()

            def mux_packets(packets):
                for packet in packets:
                    if packet.pts is not None:
                        stamp = round(packet.pts * packet.time_base / time_base)
                        duration = durations.pop(stamp, None)
                        if duration is not None:
                            packet.duration = max(1, round(duration * time_base / packet.time_base))
                    output.mux(packet)

            def encode_frame(frame, pts, duration, image=None):
                options.get('check_cancel', lambda: None)()
                image = oriented_bgr(frame, stream) if image is None else image
                if image.shape[:2] != (height, width):
                    raise ValueError('视频中途改变了分辨率，暂不支持这种文件')
                if (draw_w, draw_h) != (width, height):
                    image = cv2.resize(image, (draw_w, draw_h), interpolation=cv2.INTER_AREA)
                cpu_result, _, timing = pool.predict(image, engine, options['conf'], defer_plot=True)
                drawn = draw_worker.submit(draw_frame, cpu_result)
                encode_pending.append(encode_worker.submit(write_frame, drawn, pts, duration, timing))
                # 限制尚未完成的画框、编码任务数量，同时保留全部帧及其顺序。
                if len(encode_pending) >= 3:
                    encode_pending.popleft().result()

            def draw_frame(cpu_result):
                options.get('check_cancel', lambda: None)()
                draw_started = time.perf_counter()
                annotated = cpu_result.plot()
                if annotated.shape[:2] != (draw_h, draw_w):
                    raise RuntimeError('画框输出尺寸不一致')
                if (out_w, out_h) != (draw_w, draw_h):
                    annotated = cv2.copyMakeBorder(annotated, 0, out_h - draw_h,
                                                  0, out_w - draw_w, cv2.BORDER_REPLICATE)
                return annotated, (time.perf_counter() - draw_started) * 1000

            def write_frame(drawn, pts, duration, timing):
                nonlocal count, infer_total, draw_total, encode_total, last_end
                options.get('check_cancel', lambda: None)()
                annotated, draw_ms = drawn.result()
                options.get('check_cancel', lambda: None)()
                infer_total += timing['infer_ms']
                draw_total += draw_ms
                encode_started = time.perf_counter()
                encoded_frame = av.VideoFrame.from_ndarray(annotated, format='bgr24')
                encoded_frame.pts, encoded_frame.time_base = pts, time_base
                durations[pts] = duration
                mux_packets(encoder.encode(encoded_frame))
                encode_total += (time.perf_counter() - encode_started) * 1000
                last_end = pts + duration
                count += 1
                if count % 10 == 0 or count == 1:
                    elapsed = time.perf_counter() - processing_started
                    progress(frames=count, processing_fps=round(count / max(elapsed, .001), 2),
                             draw_ms=round(draw_total / count, 2), encode_ms=round(encode_total / count, 2))

            # 向前读取一帧，通过相邻时间戳计算可变帧时长，避免缓存整个视频。
            pending, pending_pts, pending_image = first, 0, first_image
            with decoded_prefetch(frames, stream) as prefetched:
                for frame, decoded_image in prefetched:
                    if frame.pts is None:
                        pts = pending_pts + default_duration
                        timing_fallback = True
                    else:
                        pts = round((frame.pts * frame.time_base - origin) / time_base)
                    if pts <= pending_pts:
                        raise ValueError('视频时间戳不递增，无法保证输出与原视频一致')
                    encode_frame(pending, pending_pts, pts - pending_pts, pending_image)
                    pending, pending_pts, pending_image = frame, pts, decoded_image
            tail_duration = getattr(pending, 'duration', 0) or 0
            tail_duration = (round(tail_duration * pending.time_base / time_base)
                             if tail_duration and pending.time_base else default_duration)
            encode_frame(pending, pending_pts, max(1, tail_duration), pending_image)
            while encode_pending:
                encode_pending.popleft().result()
            def flush_encoder():
                mux_packets(encoder.encode(None))
            encode_worker.submit(flush_encoder).result()
            processing_elapsed = time.perf_counter() - processing_started

    duration_seconds = float(last_end * time_base)
    if has_audio:
        progress(stage='muxing', frames=count)
        # 视频已将首帧时间戳归零；音频应用同样偏移以保持同步。
        command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
                   '-i', intermediate, '-itsoffset', str(-float(origin)), '-i', str(source),
                   '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy',
                   '-c:a', 'aac', '-b:a', '192k', '-t', str(duration_seconds),
                   '-map_metadata', '-1', '-movflags', '+faststart', str(target)]
        completed = subprocess.run(command, capture_output=True, text=True,
                                   encoding='utf-8', errors='replace', timeout=3600)
        if completed.returncode:
            raise RuntimeError('视频音轨合成失败: ' + completed.stderr[-1500:])
        os.unlink(intermediate)
    else:
        os.replace(intermediate, target)
    elapsed = time.perf_counter() - started
    return dict(source_w=width, source_h=height, output_w=out_w, output_h=out_h,
                input_w=engine['input_w'], input_h=engine['input_h'], engine=engine['key'],
                source_fps=str(fps), frames=count, duration=round(duration_seconds, 6),
                video_decoder=decoder_name, video_encoder=encoder_name, drawing='CPU', hardware_notes=hardware_notes,
                video_pipeline='解码 / 推理 / 画框 / 编码四阶段流水线（有界队列）',
                draw_ms=round(draw_total / max(count, 1), 2), encode_ms=round(encode_total / max(count, 1), 2),
                timing='fallback' if timing_fallback else 'source_pts', audio=has_audio,
                processing_fps=round(count / max(processing_elapsed, .001), 2),
                warmup_ms=round(warmup_ms, 2), warmup_runs=5,
                processing_seconds=round(processing_elapsed, 3),
                infer_ms=round(infer_total / max(count, 1), 2), elapsed=round(elapsed, 2))


"""实时 JPEG 解码与推理，独立于文件处理流水线。"""
import threading
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as functional

try:
    from nvidia import nvimgcodec
except ImportError:
    nvimgcodec = None


# 实时 JPEG 通道：客户端画框时返回检测列表与统计信息，服务器画框时还返回标注 JPEG。
class RealtimePipeline:
    def __init__(self, pool):
        self.pool = pool
        self.decode_lock = threading.Lock()
        self.decoder = None
        self.warned = False
        self.retry_decoder_after = 0.

    # GPU 解码失败后进入 30 秒冷却期，期间使用 OpenCV，避免每帧重复初始化失败。
    @torch.inference_mode()
    def process(self, raw, config):
        if len(raw) > 20 * 1024 * 1024:
            raise ValueError('实时 JPEG 过大，请降低采集分辨率或质量')
        if config.get('render') == 'server':
            started = time.perf_counter()
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('实时 JPEG 解码失败')
            height, width = image.shape[:2]
            decode_ms = (time.perf_counter() - started) * 1000
            engine = self.pool.select(width, height, config.get('engine', 'auto'), 'balanced')
            annotated, _, timing = self.pool.predict(image, engine, float(config.get('conf', .4)), annotate=True)
            encode_started = time.perf_counter()
            ok, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise RuntimeError('服务器实时 JPEG 编码失败')
            return dict(type='result', id=config.get('id'), render='server', dets=[],
                        width=width, height=height, engine=engine['key'], decoder='OpenCV CPU',
                        decode_ms=round(decode_ms, 2), server_draw_ms=timing['draw_ms'],
                        server_encode_ms=round((time.perf_counter()-encode_started)*1000, 2),
                        return_bytes=int(jpeg.size), _jpeg=jpeg.tobytes(), **timing)
        started = time.perf_counter()
        tensor = None
        decoder_name = 'OpenCV CPU'
        if nvimgcodec is not None and time.monotonic() >= self.retry_decoder_after:
            try:
                with self.decode_lock:
                    if self.decoder is None:
                        self.decoder = nvimgcodec.Decoder()
                    decoded = self.decoder.decode(raw)
                    tensor = torch.from_dlpack(decoded.cuda())
                    if tensor.ndim != 3:
                        raise ValueError('解码张量维数不支持')
                    if tensor.shape[-1] == 3:
                        height, width = tensor.shape[:2]
                        tensor = tensor.permute(2, 0, 1)
                    elif tensor.shape[0] == 3:
                        height, width = tensor.shape[1:]
                    else:
                        raise ValueError('解码结果不是三通道 RGB')
                    width, height = int(width), int(height)
                    engine = self.pool.select(width, height, config.get('engine', 'auto'), 'balanced')
                    ih, iw = engine['input_h'], engine['input_w']
                    scale = min(iw / width, ih / height)
                    resized_w, resized_h = max(1, round(width * scale)), max(1, round(height * scale))
                    left, top = (iw - resized_w) // 2, (ih - resized_h) // 2
                    padding = (left, iw - resized_w - left, top, ih - resized_h - top)
                    if (resized_h, resized_w) != (height, width):
                        tensor = tensor.to(device='cuda:0', dtype=torch.float32,
                                           memory_format=torch.contiguous_format).div_(255).unsqueeze(0)
                        tensor = functional.interpolate(tensor, size=(resized_h, resized_w),
                                                        mode='bilinear', align_corners=False)
                        if any(padding):
                            tensor = functional.pad(tensor, padding, value=114 / 255)
                    else:
                        # 常见摄像头路径：画面无需缩放，只需按引擎尺寸补边。
                        # 先对字节像素补边，再转浮点数，以减少临时显存占用。
                        if any(padding):
                            tensor = functional.pad(tensor, padding, value=114)
                        tensor = tensor.to(device='cuda:0', dtype=torch.float32,
                                           memory_format=torch.contiguous_format).div_(255).unsqueeze(0)
                    tensor = tensor.contiguous()
                    # 等待当前 CUDA 流完成，之后才允许其他线程复用解码器。
                    torch.cuda.current_stream().synchronize()
                    decoder_name = 'nvImageCodec / DLPack'
            except Exception as exc:
                tensor = None
                self.retry_decoder_after = time.monotonic() + 30
                if not self.warned:
                    print(f'[realtime] nvImageCodec 不可用，回退 OpenCV: {exc}', flush=True)
                    self.warned = True

        if tensor is None:
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('实时 JPEG 解码失败')
            height, width = image.shape[:2]
            engine = self.pool.select(width, height, config.get('engine', 'auto'), 'balanced')
            input_image = image
        else:
            input_image = tensor
        decode_ms = (time.perf_counter() - started) * 1000
        _, detections, timing = self.pool.predict(input_image, engine, float(config.get('conf', .4)))
        if tensor is not None:
            # 张量输入返回补边后引擎坐标系中的检测框，需逆向消除补边与缩放。
            sx, sy = resized_w / width, resized_h / height
            for detection in detections:
                x1, y1, x2, y2 = detection['box']
                detection['box'] = [max(0., min(float(width), (x1 - left) / sx)),
                                    max(0., min(float(height), (y1 - top) / sy)),
                                    max(0., min(float(width), (x2 - left) / sx)),
                                    max(0., min(float(height), (y2 - top) / sy))]
        return dict(type='result', id=config.get('id'), dets=detections,
                    width=width, height=height, engine=engine['key'],
                    decode_ms=round(decode_ms, 2), decoder=decoder_name, **timing)


"""WHIP 信令与本地 RTSP 接收流程，独立于 JPEG 通道，不依赖 aiortc。"""
import asyncio
import math
import os
import queue
import threading
import time
import uuid
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import cv2
import numpy as np
from fastapi import HTTPException, WebSocket


class AnnotatedRTSPPublisher:
    """由单个会话执行器管理，所有编解码调用固定在该工作线程上。"""
    def __init__(self, base, path):
        self.base, self.path = base, path
        self.output = None
        self.epoch = 0
        self.shape = None
        self.codec = None
        self.note = ''
        self.ready = False
        self.force_keyframe = threading.Event()
        self.init_ms = 0.

    def close(self):
        output, self.output = self.output, None
        self.ready = False
        if output is not None:
            # 停止实时流时直接关闭输出，不显式排空编码器中的尾帧。
            output.close()

    def open(self, width, height, bitrate, fps=60):
        import av
        opened_at = time.perf_counter()
        self.close()
        self.epoch += 1
        # 采用现有 MediaMTX 配置允许的路径格式，无需修改 YAML。
        self.output_path = 'yolo-' + uuid.uuid4().hex
        self.shape = (width, height, bitrate, fps)
        self.origin = None
        self.last_pts = -1
        self.note = ''
        for codec in ('h264_nvenc', 'libx264'):
            output = None
            try:
                output = av.open(self.base + '/' + self.output_path, 'w', format='rtsp',
                                 options={'rtsp_transport': 'tcp', 'rw_timeout': '3000000',
                                          'flush_packets': '1', 'max_delay': '0'})
                stream = output.add_stream(codec, rate=fps)
                stream.width, stream.height = width, height
                stream.pix_fmt = 'yuv420p'
                stream.time_base = Fraction(1, 90000)
                stream.codec_context.time_base = stream.time_base
                stream.codec_context.max_b_frames = 0
                stream.codec_context.gop_size = 15
                stream.bit_rate = bitrate
                stream.options = ({'preset': 'p4', 'tune': 'ull', 'rc': 'cbr',
                                   'zerolatency': '1', 'rc-lookahead': '0', 'delay': '0', 'forced-idr': '1',
                                   'bufsize': str(max(100000, bitrate//10)), 'profile': 'baseline'}
                                  if codec == 'h264_nvenc' else
                                  {'preset': 'veryfast', 'tune': 'zerolatency', 'profile': 'baseline',
                                   'x264-params': 'keyint=15:min-keyint=15:scenecut=0:repeat-headers=1'})
                output.start_encoding()
                self.output, self.stream, self.codec = output, stream, codec
                self.init_ms = round((time.perf_counter()-opened_at)*1000, 2)
                return
            except Exception as exc:
                if output is not None:
                    try:
                        output.close()
                    except Exception:
                        pass
                if codec == 'libx264':
                    raise RuntimeError('标注视频发布失败：'+str(exc)) from exc
                self.note = 'NVENC 初始化回退：'+str(exc)[:200]

    def write(self, image, stamp, bitrate, fps=60):
        import av
        height, width = image.shape[:2]
        # YUV420 要求宽高为偶数；通过补边满足要求，不拉伸画面。
        if width % 2 or height % 2:
            image = cv2.copyMakeBorder(image, 0, height % 2, 0, width % 2, cv2.BORDER_REPLICATE)
            height, width = image.shape[:2]
        if self.output is None or self.shape != (width, height, bitrate, fps):
            self.open(width, height, bitrate, fps)
        if self.origin is None:
            self.origin = stamp
        pts = max(self.last_pts + 1, round((stamp-self.origin)*90000))
        frame = av.VideoFrame.from_ndarray(image, format='bgr24')
        frame.pts, frame.time_base = pts, Fraction(1, 90000)
        if self.force_keyframe.is_set():
            self.force_keyframe.clear()
            frame.pict_type = av.video.frame.PictureType.I
        for packet in self.stream.encode(frame):
            self.output.mux(packet)
            self.ready = True
        self.last_pts = pts
        return dict(output_ready=self.ready, output_epoch=self.epoch,
                    output_encoder=self.codec, output_note=self.note, output_path=self.output_path,
                    encoder_init_ms=self.init_ms)


# 注册 WebRTC 信令、结果及回放接口；媒体经 MediaMTX 转为本地 RTSP 处理。
def install_webrtc(app, get_pool):
    sessions = {}
    whip_base = os.environ.get('MEDIAMTX_HTTP', 'http://127.0.0.1:8889').rstrip('/')
    rtsp_base = os.environ.get('MEDIAMTX_RTSP', 'rtsp://127.0.0.1:8554').rstrip('/')

    def configuration(value):
        engine = str(value.get('engine', ''))
        conf = float(value.get('conf', .4))
        if get_pool() is None or engine not in get_pool().engines:
            raise ValueError('模型不存在')
        if not math.isfinite(conf) or not 0 <= conf <= 1:
            raise ValueError('无效置信度')
        width, height = int(value['width']), int(value['height'])
        e = get_pool().engines[engine]
        if min(width, height) < 16 or width * height > e['input_w'] * e['input_h'] * 1.1:
            raise ValueError('无效画面尺寸')
        render = value.get('render', 'client')
        if render not in ('client', 'server'):
            raise ValueError('无效画框位置')
        policy = value.get('queue_policy', 'fifo')
        if policy not in ('fifo', 'latest'):
            raise ValueError('无效实时队列策略')
        bitrate = max(500000, min(50000000, int(value.get('bitrate', 8000000))))
        fps = int(value.get('fps', 60))
        if fps not in (10, 15, 20, 24, 25, 30, 40, 45, 50, 60):
            raise ValueError('无效帧率')
        return dict(fps=fps, engine=engine, conf=conf, revision=int(value.get('revision', 0)),
                    width=width, height=height, render=render, bitrate=bitrate, queue_policy=policy)

    def http_request(url, method, data=None):
        request = urllib.request.Request(url, data=data, method=method,
                                        headers={'Content-Type': 'application/sdp'})
        with urllib.request.urlopen(request, timeout=12) as response:
            return response.read(250000).decode('utf-8'), response.headers.get('Location')

    # 幂等关闭会话：先停止接收，再回收播放连接、发布器及其所属执行线程。
    async def close_session(session):
        if session.get('closing'):
            return
        session['closing'] = True
        session['stop'].set()
        if session.get('location'):
            try:
                await asyncio.to_thread(http_request, session['location'], 'DELETE')
            except Exception:
                pass
        if session.get('worker'):
            await asyncio.to_thread(session['worker'].join)
        for location in list(session['playbacks'].values()):
            if not location:
                continue
            try:
                await asyncio.to_thread(http_request, location, 'DELETE')
            except Exception:
                pass
        executor = session.get('executor')
        if executor is not None:
            try:
                await asyncio.wrap_future(executor.submit(session['publisher'].close))
            except Exception as exc:
                print('[webrtc] close publisher:', str(exc), flush=True)
            finally:
                await asyncio.to_thread(executor.shutdown, wait=True)
        sessions.pop(session['id'], None)

    # 读取 RTSP 帧；先进先出模式等待队列空间，最新帧模式替换积压帧以降低延迟。
    def read_stream(session):
        ready, stop = session['frames'], session['stop']
        def put(value):
            while not stop.is_set():
                try:
                    if session['config']['queue_policy'] == 'latest':
                        ready.put_nowait(value)
                    else:
                        ready.put(value, timeout=.1)
                    return
                except queue.Full:
                    if session['config']['queue_policy'] == 'latest':
                        try:
                            old = ready.get_nowait()
                            if old[0] == 'frame':
                                session['replaced'] += 1
                        except queue.Empty:
                            pass
        decode_opened_at = time.perf_counter()
        deadline = time.monotonic() + 20
        hardware, first, fallback = True, True, ''
        startup_invalid_retries = 0
        while not stop.is_set():
            container = None
            try:
                import av
                kwargs = {}
                if hardware:
                    from av.codec.hwaccel import HWAccel
                    kwargs['hwaccel'] = HWAccel(device_type='cuda', allow_software_fallback=False)
                container = av.open(rtsp_base + '/' + session['path'],
                                    options={'rtsp_transport': 'tcp', 'probesize': '32768',
                                             'analyzeduration': '100000', 'max_delay': '0',
                                             'reorder_queue_size': '0'}, timeout=(3., 3.), **kwargs)
                stream = container.streams.video[0]
                if not hardware:
                    stream.thread_type = 'SLICE'
                    stream.codec_context.thread_count = 2
                for frame in container.decode(stream):
                    if stop.is_set():
                        return
                    if hardware and not getattr(stream.codec_context, 'is_hwaccel', False):
                        raise RuntimeError('PyAV 未确认 NVDEC 已启用')
                    fetched = time.perf_counter()
                    image = frame.to_ndarray(format='bgr24')
                    with session['lock']:
                        config = dict(session['config'])
                    if first:
                        session['decoder_startup_ms'] = round((time.perf_counter()-decode_opened_at)*1000, 2)
                        print(f"[webrtc:{session['id'][:8]}] first frame: "
                              f"decoder={'NVDEC' if hardware else 'CPU'}, "
                              f"size={frame.width}x{frame.height}, "
                              f"startup_ms={session['decoder_startup_ms']}, "
                              f"fallback={fallback or 'none'}", flush=True)
                    session['decoded'] += 1
                    first = False
                    stamp = float(frame.pts * frame.time_base) if frame.pts is not None and frame.time_base else fetched
                    put(('frame', (image, config, fetched, (time.perf_counter()-fetched)*1000,
                         'NVDEC → CPU BGR' if hardware else 'CPU', fallback, stamp)))
                if not stop.is_set():
                    raise RuntimeError('视频流已结束')
                return
            except Exception as exc:
                if stop.is_set():
                    return
                print(f"[webrtc:{session['id'][:8]}] decode error: "
                      f"decoder={'NVDEC' if hardware else 'CPU'}, before_first={first}, "
                      f"exception={type(exc).__name__}: {exc}", flush=True)
                if not first:
                    put(('error', str(exc)))
                    return
                # 压缩输入无效不一定意味着硬件不兼容。
                # 在原始启动期限内，使用同一解码器重试一次 RTSP 初始连接，
                # 避免因错误输入无限重试。
                if (hardware and type(exc).__name__ == 'InvalidDataError'
                        and startup_invalid_retries < 1 and time.monotonic() < deadline - 4):
                    startup_invalid_retries += 1
                    print(f"[webrtc:{session['id'][:8]}] retry startup NVDEC once", flush=True)
                    stop.wait(.3)
                    continue
                if hardware and (container is not None or isinstance(exc, ImportError)
                                 or time.monotonic() >= deadline - 10):
                    hardware = False
                    fallback = str(exc)[:200]
                    print(f"[webrtc:{session['id'][:8]}] switching to CPU: {fallback}", flush=True)
                if time.monotonic() >= deadline:
                    put(('error', 'RTSP 接收失败：' + str(exc)))
                    return
                stop.wait(.3)
            finally:
                if container is not None:
                    container.close()

    def detect(image, config, session, stamp):
        # 与 JPEG 一样等比例容纳完整画面，不裁剪、拉伸或自行推断旋转方向。
        height, width = image.shape[:2]
        w, h = config['width'], config['height']
        scale = min(w / width, h / height)
        dw, dh = max(1, round(width*scale)), max(1, round(height*scale))
        dx, dy = (w-dw)//2, (h-dh)//2
        if (width, height) != (w, h):
            fitted = np.zeros((h, w, 3), dtype=np.uint8)
            fitted[dy:dy+dh, dx:dx+dw] = cv2.resize(image, (dw, dh))
            image = fitted
        engine = get_pool().select(w, h, config['engine'])
        server = config['render'] == 'server'
        drawn, dets, timing = get_pool().predict(image, engine, config['conf'], annotate=server)
        output_info = {}
        if server:
            encode_started = time.perf_counter()
            output_info = session['publisher'].write(drawn, stamp, config['bitrate'], config['fps'])
            output_info['server_encode_ms'] = round((time.perf_counter()-encode_started)*1000, 2)
            dets = []
        # WebRTC 检测框按处理画幅归一化；前端再乘显示宽高，JPEG 通道则直接返回像素坐标。
        for det in dets:
            x1, y1, x2, y2 = det['box']
            det['box'] = [x1/w, y1/h, x2/w, y2/h]
        return dict(type='rtc_result', dets=dets, width=w, height=h,
                    source_width=width, source_height=height,
                    engine=engine['key'], revision=config['revision'], render=config['render'],
                    server_draw_ms=timing['draw_ms'], **output_info, **timing)

    @app.post('/api/webrtc/offer')
    async def offer(params: dict):
        try:
            config = configuration(params.get('config', {}))
            sdp = params.get('sdp')
            if params.get('type') != 'offer' or not isinstance(sdp, str) or len(sdp) > 200000:
                raise ValueError('无效 SDP')
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        if len(sessions) >= 2:
            raise HTTPException(429, '最多两路 WebRTC，请先停止旧连接')
        sid = uuid.uuid4().hex
        session = dict(id=sid, path='yolo-'+sid, config=config, lock=threading.Lock(),
                       stop=threading.Event(), frames=queue.Queue(maxsize=4 if config['queue_policy'] == 'fifo' else 1),
                       last_seen=time.monotonic(), initializing=True, closing=False, attached=False, playbacks={},
                       decoded=0, replaced=0, decoder_startup_ms=None)
        sessions[sid] = session
        url = whip_base + '/' + session['path'] + '/whip'
        try:
            # 在媒体帧到达前创建推理与编码线程，
            # 将 CUDA 初始化、预热及所有编码调用固定到该线程。
            session['executor'] = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rtc-process')
            session['publisher'] = AnnotatedRTSPPublisher(rtsp_base, session['path'])
            def warmup():
                started = time.perf_counter()
                image = np.zeros((config['height'], config['width'], 3), dtype=np.uint8)
                engine = get_pool().select(config['width'], config['height'], config['engine'])
                for _ in range(5):
                    if session['stop'].is_set():
                        raise RuntimeError('实时会话已停止')
                    get_pool().predict(image, engine, config['conf'], annotate=config['render']=='server')
                model_ms = round((time.perf_counter()-started)*1000, 2)
                if config['render'] == 'server' and not session['stop'].is_set():
                    session['publisher'].open(config['width']+config['width']%2,
                                              config['height']+config['height']%2, config['bitrate'], config['fps'])
                return model_ms
            # 先预热实际使用的处理路径，再协商发布连接，
            # 避免模型首次加载期间积压输入帧。
            session['warmup_ms'] = await asyncio.wrap_future(session['executor'].submit(warmup))
            if session['closing']:
                raise RuntimeError('实时会话已关闭')
            exchange = asyncio.create_task(asyncio.to_thread(http_request, url, 'POST', sdp.encode()))
            try:
                answer, location = await asyncio.shield(exchange)
            except asyncio.CancelledError:
                _, location = await exchange
                session['location'] = urllib.parse.urljoin(url, location or '')
                raise
            if not location:
                raise ValueError('MediaMTX 缺少 WHIP Location 响应头')
            location = urllib.parse.urljoin(url, location)
            if urllib.parse.urlsplit(location).netloc != urllib.parse.urlsplit(whip_base).netloc:
                raise ValueError('MediaMTX 返回不同主机的信令地址')
            session['location'] = location
            session['initializing'] = False
            session['last_seen'] = time.monotonic()
            return dict(type='answer', sdp=answer, session_id=sid, warmup_ms=session['warmup_ms'])
        except BaseException as exc:
            await close_session(session)
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise HTTPException(502, 'MediaMTX 信令失败，请先启动 mediamtx：'+str(exc)) from exc

    @app.websocket('/api/webrtc/{session_id}/results')
    async def results(websocket: WebSocket, session_id: str):
        session = sessions.get(session_id)
        if not session or session['closing'] or session['attached']:
            await websocket.close(code=1008)
            return
        session['attached'] = True
        await websocket.accept()
        worker = threading.Thread(target=read_stream, args=(session,), daemon=True, name='rtc-rtsp')
        session['worker'] = worker
        worker.start()
        send_lock = asyncio.Lock()
        async def send(value):
            async with send_lock:
                await asyncio.wait_for(websocket.send_json(value), 5)
        async def receive():
            while True:
                value = await websocket.receive_json()
                session['last_seen'] = time.monotonic()
                if value.get('type') == 'config':
                    config = configuration(value)
                    with session['lock']:
                        session['config'] = config
                elif value.get('type') == 'ping':
                    await send(dict(type='pong', sent_at=value.get('sent_at')))
                elif value.get('type') == 'playback_ready':
                    session['publisher'].force_keyframe.set()
        def next_frame():
            try:
                return session['frames'].get(timeout=.5)
            except queue.Empty:
                return None
        async def infer():
            count = 0
            while not session['stop'].is_set():
                item = await asyncio.to_thread(next_frame)
                if item is None:
                    continue
                kind, payload = item
                if kind == 'error':
                    await send(dict(type='error', error=payload))
                    return
                image, config, arrived, convert_ms, decoder, fallback, stamp = payload
                started = time.perf_counter()
                result = await asyncio.wrap_future(session['executor'].submit(detect, image, config, session, stamp))
                if result.get('output_ready'):
                    session['output_path'] = result.pop('output_path')
                    session['output_epoch'] = result['output_epoch']
                count += 1
                result.update(id=count, convert_ms=round(convert_ms, 2), decoder=decoder,
                              fallback=fallback, pending=session['frames'].qsize(),
                              queue_capacity=session['frames'].maxsize, warmup_ms=session['warmup_ms'],
                              decoder_startup_ms=session['decoder_startup_ms'], decoded=session['decoded'],
                              replaced=session['replaced'], queue_policy=config['queue_policy'],
                              server_ms=round((time.perf_counter()-arrived)*1000, 2),
                              queue_ms=round(max(0., (started-arrived)*1000-convert_ms), 2))
                await send(result)
        tasks = [asyncio.create_task(receive()), asyncio.create_task(infer())]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except Exception as exc:
            try:
                await send(dict(type='error', error=str(exc)))
            except Exception:
                pass
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await close_session(session)
            try:
                await websocket.close()
            except Exception:
                pass

    @app.delete('/api/webrtc/{session_id}')
    async def stop(session_id: str):
        session = sessions.get(session_id)
        if session:
            await close_session(session)
        return dict(closed=True)

    @app.post('/api/webrtc/{session_id}/playback')
    async def playback(session_id: str, params: dict):
        session = sessions.get(session_id)
        if not session or session['closing']:
            raise HTTPException(404, '实时会话已关闭')
        sdp, epoch = params.get('sdp'), params.get('epoch')
        if not isinstance(sdp, str) or len(sdp) > 200000 or params.get('type') != 'offer':
            raise HTTPException(400, '无效播放 SDP')
        if not session.get('output_path') or epoch != session.get('output_epoch'):
            raise HTTPException(409, '标注流尚未就绪或正在切换尺寸')
        if len(session['playbacks']) >= 2:
            raise HTTPException(429, '请先关闭旧播放连接')
        pid = uuid.uuid4().hex
        session['playbacks'][pid] = None  # 在让出执行权前预留名额，防止并发请求超出连接上限。
        url = whip_base + '/' + session['output_path'] + '/whep'
        location = None
        try:
            exchange = asyncio.create_task(asyncio.to_thread(http_request, url, 'POST', sdp.encode()))
            try:
                answer, returned = await asyncio.shield(exchange)
            except asyncio.CancelledError:
                _, returned = await exchange
                location = urllib.parse.urljoin(url, returned) if returned else None
                raise
            location = urllib.parse.urljoin(url, returned) if returned else None
            if not location or urllib.parse.urlsplit(location).netloc != urllib.parse.urlsplit(whip_base).netloc:
                raise ValueError('无效 WHEP Location')
            if session['closing'] or epoch != session.get('output_epoch'):
                raise HTTPException(409, '实时会话或尺寸已经变化')
            session['playbacks'][pid] = location
            return dict(sdp=answer, type='answer', playback_id=pid)
        except BaseException as exc:
            session['playbacks'].pop(pid, None)
            if location:
                try:
                    await asyncio.to_thread(http_request, location, 'DELETE')
                except Exception:
                    pass
            if isinstance(exc, (asyncio.CancelledError, HTTPException)):
                raise
            raise HTTPException(502, '标注视频播放协商失败：'+str(exc)) from exc

    @app.delete('/api/webrtc/{session_id}/playback/{playback_id}')
    async def stop_playback(session_id: str, playback_id: str):
        session = sessions.get(session_id)
        location = session['playbacks'].pop(playback_id, None) if session else None
        if location:
            try:
                await asyncio.to_thread(http_request, location, 'DELETE')
            except Exception:
                pass
        return dict(closed=True)

    async def reap():
        for session in list(sessions.values()):
            if time.monotonic()-session['last_seen'] > (120 if session.get('initializing') else 35):
                await close_session(session)

    async def shutdown():
        await asyncio.gather(*(close_session(s) for s in list(sessions.values())), return_exceptions=True)
    return shutdown, reap


HTML = r'''
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>YOLO · 图片视频与实时检测</title>
<style>
:root{color-scheme:dark;font-family:system-ui,-apple-system,"PingFang SC",sans-serif;background:#0b1019;color:#e6edf7}
*{box-sizing:border-box}body{margin:0;padding:24px;max-width:1600px;margin:auto}h1{font-size:25px;margin:0 0 8px}h2{font-size:16px;margin:0;padding:14px;border-bottom:1px solid #29364c}p{color:#9cabbe;line-height:1.6}.tabs,.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:18px 0}.controls{padding:16px;background:#141e2d;border:1px solid #29364c;border-radius:12px}button,select,input[type=file]{font:inherit}button,select,.download{padding:10px 14px;border:1px solid #364963;border-radius:8px;color:#e6edf7;background:#203149}button{cursor:pointer}button.active,button.primary{background:#2563eb;border-color:#4285ff}button:disabled{opacity:.5;cursor:wait}label{display:flex;gap:8px;align-items:center;font-size:14px;color:#bac8dc}input[type=range]{width:110px}input[type=number]{width:68px;background:#0b1019;color:#fff;padding:8px;border:1px solid #364963;border-radius:6px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:start}.panel{min-width:0;background:#111a28;border:1px solid #29364c;border-radius:12px;overflow:hidden}.viewport{padding:12px;min-height:80px}img,video,canvas{display:block;width:100%;height:auto;background:#05080e;border-radius:6px}img:not([src]),video:not([src]):not(.camera){display:none}.status{padding:12px 0;line-height:1.7;white-space:pre-wrap;color:#8bc8ff;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}.hint{font-size:13px;color:#9cabbe}.download{display:inline-block;text-decoration:none;margin-bottom:12px}progress{width:100%;height:9px;accent-color:#3b82f6}[hidden]{display:none!important}@media(max-width:850px){body{padding:14px}.grid{grid-template-columns:1fr}.controls{gap:14px}h1{font-size:21px}}
.download{margin:12px 0 0}
#batchList{display:grid;gap:10px;max-height:420px;overflow:auto;margin:12px 0}
.batch-item{padding:12px;background:#141e2d;border:1px solid #29364c;border-radius:10px;min-width:0;overflow-wrap:anywhere}
.batch-item button,.batch-item a{margin:8px 8px 0 0}
.batch-detail{white-space:pre-wrap;font-size:13px;line-height:1.5;color:#9cabbe;margin-top:6px}
#liveSection .controls>*{max-width:100%}
#liveSection .controls label{flex-wrap:wrap;min-width:0}
#liveSection select{max-width:100%;min-width:0}
@media (pointer:coarse) and (orientation:portrait){
 body{padding:10px}#liveSection .grid{grid-template-columns:minmax(0,1fr)}
 #liveSection .controls{padding:10px;gap:10px}#liveSection .controls label{flex:1 1 145px}
 #liveSection .status{font-size:12px}#liveSection .viewport{padding:6px}
}
@media (pointer:coarse) and (orientation:landscape){
 body{padding:8px}#liveSection .grid{grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:8px}
 #liveSection .controls{padding:8px;gap:8px;margin:8px 0}
 #liveSection button,#liveSection select{padding:6px 8px;font-size:12px}
 #liveSection label{font-size:12px}#liveSection .status{font-size:11px;line-height:1.35;max-height:7em;overflow:auto}
 #liveSection h2{padding:8px;font-size:13px}#liveSection .viewport{padding:6px}
}
#clearQueue:disabled{cursor:default}
</style>
</head>
<body>
<h1>YOLO 视觉检测</h1>
<p>图片与视频由服务器画框；实时摄像头可选 JPEG 或 WebRTC 视频流。</p>
<nav class="tabs"><button id="mediaTab" class="active">图片 / 视频</button><button id="liveTab">实时摄像头</button></nav>
<section id="mediaSection">
 <div class="controls">
  <input id="file" type="file" accept="image/*,video/*" multiple>
  <label>引擎<select id="mediaEngine"><option value="auto">自动匹配比例与分辨率</option></select></label>
  <label>策略<select id="profile"><option value="detail" selected>细节 · 高分辨率 / 接近原图</option><option value="balanced">均衡 · 1080p 优先</option><option value="speed">速度 · 720p 优先</option></select></label>
  <label>视频输出<select id="output"><option value="original">保留原始分辨率</option><option value="nearest">适配引擎分辨率（保持比例）</option></select></label>
  <label>置信度<input id="mediaConfSlider" aria-label="图片视频置信度滑块" type="range" min="0" max="1" step=".01" value=".40"><input id="mediaConf" aria-label="图片视频置信度数值" type="number" min="0" max="1" step=".01" value=".40"></label>
  <label>同时处理<select id="mediaParallel"><option value="1">1 项</option><option value="2">2 项</option><option value="3" selected>3 项</option><option value="4">4 项</option></select></label>
  <button id="detect" class="primary">批量上传并检测</button>
  <button id="clearQueue" disabled>清空全部记录</button>
  <button id="stopTask" disabled>停止所有进行中任务</button>
 </div>
 <div class="hint">支持混合多选图片和视频，再次选择可追加。可同时处理 1 或 2 项；每项可单独预览和下载。并发提高整体吞吐，单个视频的 FPS 可能降低。停止按钮暂停后续队列并停止所有进行中任务。</div>
 <div class="hint">默认按原图尺寸自动匹配现有模型，支持高于 1080p 的引擎；也可手动选择。高分辨率需要更多显存，处理大图或高分辨率视频时建议同时处理 1 项。</div>
 <div id="batchStatus" class="status" role="status"></div>
 <div id="batchList"></div>
 <div id="mediaStatus" class="status" role="status">请选择图片或视频。</div><progress id="progress" max="100" value="0" hidden></progress>
 <div class="grid">
  <div class="panel"><h2>原始文件</h2><div class="viewport"><img id="sourceImage" alt="原始图片" hidden><video id="sourceVideo" controls playsinline preload="metadata" hidden></video></div></div>
  <div class="panel"><h2>服务器检测结果</h2><div class="viewport"><img id="resultImage" alt="服务器标注图片" hidden><video id="resultVideo" controls playsinline preload="metadata" hidden></video><a id="download" class="download" hidden>下载检测结果</a></div></div>
 </div>
</section>
<section id="liveSection" hidden>
 <div class="controls">
  <button id="start" class="primary" disabled>开始摄像头</button><button id="stop" disabled>停止</button>
  <label>摄像头<select id="camera"><option value="">默认后置摄像头</option></select></label>
  <label>传输<select id="liveTransport"><option value="jpeg">JPEG（原通道）</option><option value="webrtc">WebRTC（MediaMTX）</option></select></label>
  <label>画框位置<select id="liveRender"><option value="client" selected>客户端画框</option><option value="server">服务器画框（同帧）</option></select></label>
  <label>WebRTC 队列<select id="rtcQueue"><option value="fifo" selected>顺序处理（保留待处理帧）</option><option value="latest">低延迟（丢弃待处理旧帧）</option></select></label>
  <label>WebRTC目标帧率<select id="rtcFps"><option value="10">10 FPS</option><option value="15">15 FPS</option><option value="20">20 FPS</option><option value="24">24 FPS</option><option value="25">25 FPS</option><option value="30">30 FPS</option><option value="40">40 FPS</option><option value="45">45 FPS</option><option value="50">50 FPS</option><option value="60" selected>60 FPS</option></select></label>
  <label>WebRTC码率上限<select id="rtcBitrate"><option value="500000">0.5 Mbps</option><option value="1000000">1 Mbps</option><option value="1500000">1.5 Mbps</option><option value="2000000">2 Mbps</option><option value="3000000">3 Mbps</option><option value="4000000">4 Mbps</option><option value="6000000">6 Mbps</option><option value="8000000" selected>8 Mbps</option><option value="10000000">10 Mbps</option><option value="12000000">12 Mbps</option><option value="16000000">16 Mbps</option><option value="20000000">20 Mbps</option><option value="25000000">25 Mbps</option><option value="30000000">30 Mbps</option><option value="40000000">40 Mbps</option><option value="50000000">50 Mbps</option></select></label>
  <span class="hint">分辨率优先；码率是上限，实际帧率受设备与带宽限制。服务器画框回传使用同一档码率目标。</span>
  <label>横竖自适应<select id="liveAspect"><option value="4:3">4:3 / 3:4</option><option value="16:9">16:9 / 9:16</option></select></label>
  <label>画幅方向<select id="liveOrientation"><option value="auto">自动</option><option value="landscape">横屏</option><option value="portrait">竖屏</option></select></label>
  <label>分辨率 / 模型<select id="liveEngine"><option value="">正在读取现有模型…</option></select></label>
  <label>JPEG 质量<input id="quality" type="range" min=".65" max=".98" step=".01" value=".90"><span id="qualityValue">90%</span></label>
  <label>JPEG 编码<select id="jpegMode"><option value="compatible">兼容编码</option><option value="worker">后台编码（试用）</option></select></label>
  <label>发送缓冲<select id="bufferMode"><option value="compatible">兼容 512KB</option><option value="adaptive">低积压</option></select></label>
  <label>置信度<input id="liveConfSlider" aria-label="实时置信度滑块" type="range" min="0" max="1" step=".01" value=".40"><input id="liveConf" aria-label="实时置信度数值" type="number" min="0" max="1" step=".01" value=".40"></label>
 </div>
 <div class="hint">完整画面等比例显示与发送，不裁剪、不拉伸；比例不符时留黑边。自动方向跟随视频帧，手动方向仅改变画幅。手机旋转自动调整页面布局。摄像头需 HTTPS 或 localhost。</div>
 <div id="liveStatus" class="status" role="status">尚未连接。</div>
 <div class="grid">
  <div class="panel"><h2>采集画面（所选比例）</h2><div class="viewport"><video id="cameraVideo" class="camera" autoplay playsinline muted></video></div></div>
  <div class="panel"><h2>实时检测</h2><div class="viewport"><canvas id="liveCanvas"></canvas><video id="serverVideo" class="camera" autoplay playsinline muted hidden></video></div></div>
 </div>
</section>
<script>
'use strict';
const $ = id => document.getElementById(id);
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
let catalog = [], sourceURL = null, selectedFile = null, mediaBusy = false;
let socket = null, cameraStream = null, liveGeneration = 0, liveReady = false;
let sequence = 0, encoding = false, lastVideoTime = -1, lastDets = [], lastDetTime = 0;
let liveRevision = 0, currentWidth = 0, currentHeight = 0, lastStatsAt = performance.now(), received = 0;
let liveGeometry = null, geometryKey = '', lastResultId = 0, firstSentAt = 0, lastResponseAt = 0;
const liveContext = $('liveCanvas').getContext('2d');
const MAX_BUFFERED_BYTES = 512 * 1024;
const capture = document.createElement('canvas');
const captureContext = capture.getContext('2d', {alpha:false});
function message(id, text){ $(id).textContent = text; }
function confidence(id){const value = Number($(id).value);if(!Number.isFinite(value)||value<0||value>1)throw new Error('置信度应在 0 到 1 之间');return value;}
function kindOf(file){if(file.type.startsWith('image/')||/\.(jpe?g|png|webp|bmp|tiff?)$/i.test(file.name))return 'image';if(file.type.startsWith('video/')||/\.(mp4|mov|mkv|avi|webm|m4v|ts)$/i.test(file.name))return 'video';throw new Error('请选择图片或视频文件');}
async function jsonResponse(response){let value;try{value=await response.json();}catch{throw new Error(`服务器响应异常（HTTP ${response.status}）`);}if(!response.ok)throw new Error(typeof value.detail==='string'?value.detail:JSON.stringify(value.detail||value));return value;}
async function loadEngines(){
 const data=await jsonResponse(await fetch('/api/engines'));catalog=data.engines;
 // 文件识别展示全部已发现引擎，按标称像素数降序排列，便于选择高分辨率。
 const mediaEngines=[...catalog].sort((a,b)=>b.w*b.h-a.w*a.h||a.key.localeCompare(b.key));
 for(const engine of mediaEngines){const option=document.createElement('option');option.value=engine.key;option.textContent=engine.label;$('mediaEngine').appendChild(option);}
 for(const option of $('liveAspect').options)option.disabled=!enginesForAspect(option.value).length;
 if(!enginesForAspect($('liveAspect').value).length){const available=[...$('liveAspect').options].find(o=>!o.disabled);if(available)$('liveAspect').value=available.value;}
 refreshLiveEngines();
}
loadEngines().catch(e=>message('mediaStatus','引擎列表加载失败：'+e.message));
$('mediaTab').onclick=()=>{stopLive();$('mediaSection').hidden=false;$('liveSection').hidden=true;$('mediaTab').classList.add('active');$('liveTab').classList.remove('active');};
$('liveTab').onclick=()=>{$('sourceVideo').pause();$('resultVideo').pause();$('mediaSection').hidden=true;$('liveSection').hidden=false;$('liveTab').classList.add('active');$('mediaTab').classList.remove('active');};
function videoHardwareInfo(m){
 if(!m.video_decoder)return '';
 // 流水线名称由服务器上报，不根据编解码器名称推断。
 let text=`\n解码：${m.video_decoder} · 编码：${m.video_encoder||'初始化中'} · 画框：CPU`;
 if(Number.isFinite(m.draw_ms))text+=`\n平均画框 ${m.draw_ms}ms · 编码调用与封装 ${m.encode_ms}ms / 帧`;
 if(m.hardware_notes?.length)text+='\n'+m.hardware_notes.join('\n');
 if(m.video_pipeline)text+='\n'+m.video_pipeline;
 return text;
}
let mediaItems=[],previewItem=null;
let mediaGeneration=0,mediaPoll=null;
const mediaUploads=new Set(),activeMediaJobs=new Set();
let stopBatch=false;
async function stopMediaJobs(ids){
 const outcomes=await Promise.allSettled(ids.map(id=>fetch('/api/jobs/'+id+'/stop',{method:'POST',keepalive:true}).then(jsonResponse)));
 return outcomes.filter(result=>result.status==='rejected');
}
$('stopTask').onclick=async()=>{
 if(!mediaBusy)return;
 const generation=mediaGeneration;stopBatch=true;$('stopTask').disabled=true;
 message('batchStatus','正在停止所有进行中的任务；尚在上传的项目会在上传完成后立即请求停止。');
 const failed=await stopMediaJobs([...activeMediaJobs]);
 if(generation===mediaGeneration&&failed.length){message('batchStatus','部分停止请求失败，请重试。');$('stopTask').disabled=false;}
};
const mediaControlIds=['file','detect','mediaEngine','profile','output','mediaConf','mediaConfSlider','mediaParallel'];
function showMediaItem(item){
 previewItem=item;selectedFile=item.file;
 for(const id of ['sourceVideo','resultVideo']){const el=$(id);el.pause();el.removeAttribute('src');el.load();el.hidden=true;}
 for(const id of ['sourceImage','resultImage']){$(id).removeAttribute('src');$(id).hidden=true;}
 $('download').hidden=true;
 if(sourceURL){URL.revokeObjectURL(sourceURL);sourceURL=null;}
 if(item.kind){
  sourceURL=URL.createObjectURL(item.file);
  const source=$(item.kind==='image'?'sourceImage':'sourceVideo');source.src=sourceURL;source.hidden=false;
 }
 if(item.url){
  const result=$(item.kind==='image'?'resultImage':'resultVideo');result.src=item.url;result.hidden=false;
  $('download').href=item.url;$('download').download=item.downloadName;$('download').hidden=false;
 }
 message('mediaStatus',item.file.name+'\n'+item.detail);
}
function setMediaDetail(item,text){
 item.detail=text;item.statusNode.textContent=text;
 if(previewItem===item)message('mediaStatus',item.file.name+'\n'+text);
}
function updateBatchStatus(){
 const count=state=>mediaItems.filter(item=>item.state===state).length;
 $('clearQueue').disabled=mediaItems.length===0;
 message('batchStatus',`共 ${mediaItems.length} 项 · 完成 ${count('done')} · 失败 ${count('error')} · 待处理 ${count('pending')}${mediaBusy?' · 并行处理中':''}`);
}
$('clearQueue').onclick=()=>{
 mediaGeneration++;
 $('stopTask').disabled=true;
 const oldJobs=[...activeMediaJobs];activeMediaJobs.clear();
 void stopMediaJobs(oldJobs).then(failed=>{if(failed.length)console.warn('部分已清空任务停止失败',failed);});
 for(const xhr of mediaUploads)xhr.abort();mediaUploads.clear();
 if(mediaPoll){mediaPoll.abort();mediaPoll=null;}
 for(const item of mediaItems)item.state='removed';
 mediaItems=[];$('batchList').replaceChildren();previewItem=null;selectedFile=null;
 mediaBusy=false;for(const id of mediaControlIds)$(id).disabled=false;
 $('file').value='';$('progress').hidden=true;$('progress').value=0;
 for(const id of ['sourceVideo','resultVideo']){const el=$(id);el.pause();el.removeAttribute('src');el.load();el.hidden=true;}
 for(const id of ['sourceImage','resultImage']){$(id).removeAttribute('src');$(id).hidden=true;}
 if(sourceURL){URL.revokeObjectURL(sourceURL);sourceURL=null;}
 $('download').hidden=true;$('download').removeAttribute('href');
 message('mediaStatus','全部记录已清空，已向已知的后台任务发送停止请求。');
 updateBatchStatus();
};
$('file').onchange=()=>{
 if(mediaBusy)return;
 let first=null;
 for(const file of $('file').files){
  const item={file,kind:null,state:'pending',detail:'等待上传'};
  try{item.kind=kindOf(file);}catch(error){item.state='error';item.detail=error.message;}
  const row=document.createElement('section');row.className='batch-item';
  item.row=row;
  const name=document.createElement('strong');name.textContent=file.name+' · '+(file.size/1048576).toFixed(1)+' MB';
  item.statusNode=document.createElement('div');item.statusNode.className='batch-detail';item.statusNode.textContent=item.detail;
  const preview=document.createElement('button');preview.textContent='预览';preview.onclick=()=>showMediaItem(item);
  const download=document.createElement('a');download.className='download';download.textContent='下载检测结果';download.hidden=true;item.downloadNode=download;
  const retry=document.createElement('button');retry.textContent='重新排队';retry.hidden=item.state!=='error'||!item.kind;item.retryNode=retry;
  retry.onclick=()=>{if(mediaBusy)return;item.state='pending';retry.hidden=true;setMediaDetail(item,'等待上传');updateBatchStatus();};
  row.append(name,item.statusNode,preview,retry,download);$('batchList').appendChild(row);
  mediaItems.push(item);first??=item;
 }
 $('file').value='';
 if(first)showMediaItem(first);
 updateBatchStatus();
};
function uploadMediaItem(item,form,generation){
 return new Promise((resolve,reject)=>{
  const xhr=new XMLHttpRequest();xhr.open('POST','/api/'+item.kind);xhr.responseType='json';
  mediaUploads.add(xhr);xhr.onloadend=()=>mediaUploads.delete(xhr);
  xhr.upload.onprogress=e=>{if(generation===mediaGeneration&&e.lengthComputable){const percent=Math.round(100*e.loaded/e.total);if(previewItem===item)$('progress').value=percent;setMediaDetail(item,`上传中 ${percent}%`);}};
  xhr.onload=()=>{if(xhr.status>=200&&xhr.status<300&&xhr.response?.job_id)resolve(xhr.response);else reject(new Error(typeof xhr.response?.detail==='string'?xhr.response.detail:`上传失败（HTTP ${xhr.status}）`));};
  xhr.onerror=()=>reject(new Error('上传连接中断'));xhr.onabort=()=>reject(new Error('上传已中止'));xhr.send(form);
 });
}
$('detect').onclick=async()=>{
 if(mediaBusy)return;
 const batch=mediaItems.filter(item=>item.state==='pending');
 if(!batch.length){message('batchStatus','请先选择文件，或将失败项重新排队。');return;}
 let settings;
 try{settings={conf:confidence('mediaConf'),engine:$('mediaEngine').value,profile:$('profile').value,output:$('output').value};}
 catch(error){message('batchStatus',error.message);return;}
 const generation=++mediaGeneration;
 stopBatch=false;activeMediaJobs.clear();$('stopTask').disabled=false;
 const controller=new AbortController();mediaPoll=controller;
 mediaBusy=true;for(const id of mediaControlIds)$(id).disabled=true;
 for(const item of mediaItems)item.retryNode.disabled=true;
 $('progress').hidden=false;updateBatchStatus();
 let cursor=0;
 const parallel=Math.max(1,Math.min(4,Number($('mediaParallel').value)||3));
 async function worker(){
  while(cursor<batch.length){
   const item=batch[cursor++];
   if(generation!==mediaGeneration)return;
   if(stopBatch)break;
   if(item.state!=='pending')continue;
   item.state='uploading';setMediaDetail(item,'开始上传');if(!previewItem)showMediaItem(item);if(previewItem===item)$('progress').value=0;
   updateBatchStatus();
   try{
    const form=new FormData();form.append('file',item.file);
    for(const [key,value] of Object.entries(settings))if(key!=='output'||item.kind==='video')form.append(key,value);
    const {job_id}=await uploadMediaItem(item,form,generation);
    if(generation!==mediaGeneration){void stopMediaJobs([job_id]);return;}
    item.jobId=job_id;item.state='processing';
    activeMediaJobs.add(job_id);
    if(stopBatch)await jsonResponse(await fetch('/api/jobs/'+job_id+'/stop',{method:'POST'}));
    let job;
    const stages={queued:'等待服务器处理',decoding:'服务器解码',warming:'模型预热',detecting:'检测并画框',muxing:'封装音轨'};
    while(true){
     if(generation!==mediaGeneration)return;
     job=await jsonResponse(await fetch('/api/jobs/'+job_id,{cache:'no-store',signal:controller.signal}));
     if(generation!==mediaGeneration)return;
     if(job.status==='error')throw new Error(job.error);
     if(job.status==='cancelled'){item.state='stopped';setMediaDetail(item,'任务已停止，可重新排队从头处理');item.retryNode.hidden=false;break;}
     if(job.status==='done')break;
     let detail=stages[job.stage]||'处理中';
     if(job.frames)detail+=` · ${job.frames}${job.total?'/'+job.total:''} 帧`;
     if(job.processing_fps)detail+=` · ${job.processing_fps} FPS`;
     if(job.engine)detail+='\n实际引擎：'+job.engine;
     setMediaDetail(item,detail+videoHardwareInfo(job));
     if(previewItem===item){if(job.total)$('progress').value=Math.min(99,100*(job.frames||0)/job.total);else $('progress').removeAttribute('value');}
     await sleep(1000);
    }
    if(item.state==='stopped')continue;
    const m=job.metadata;
    item.url='/api/jobs/'+job_id+'/result';
    item.downloadName=item.file.name.replace(/\.[^.]+$/,'')+'-detected'+(item.kind==='image'?'.jpg':'.mp4');
    let summary=`完成 · ${m.source_w}×${m.source_h} → ${m.output_w}×${m.output_h}\n实际引擎：${m.engine} · 输入 ${m.input_w}×${m.input_h}`;
    if(item.kind==='video'){
     summary+=`\n${m.frames} 帧 · 原帧率 ${m.source_fps} · ${m.processing_fps} FPS · 总耗时 ${job.elapsed}s`;
     summary+=m.timing==='source_pts'?'\n按原始时间戳播放。':'\n缺失时间戳使用标称帧率。';
     if(m.warmup_runs)summary+=`\n模型预热 ${m.warmup_runs} 次 / ${m.warmup_ms}ms；处理FPS不含初始化、预热和最后音轨封装。`;
    }else summary+=`\n${m.detections} 个目标 · 总耗时 ${job.elapsed}s`;
    item.state='done';setMediaDetail(item,summary+videoHardwareInfo(m));
    item.downloadNode.href=item.url;item.downloadNode.download=item.downloadName;item.downloadNode.hidden=false;
    if(previewItem===item)showMediaItem(item);
    if(previewItem===item)$('progress').value=100;
   }catch(error){if(generation!==mediaGeneration)return;item.state='error';setMediaDetail(item,'处理失败：'+error.message);item.retryNode.hidden=false;}
   finally{if(item.jobId)activeMediaJobs.delete(item.jobId);if(generation===mediaGeneration)updateBatchStatus();}
   updateBatchStatus();
  }
 }
 try{await Promise.all(Array.from({length:parallel},()=>worker()));}
 finally{
  if(generation===mediaGeneration){
  activeMediaJobs.clear();$('stopTask').disabled=true;
  mediaPoll=null;
  mediaBusy=false;for(const id of mediaControlIds)$(id).disabled=false;
  for(const item of mediaItems)item.retryNode.disabled=false;
  $('progress').hidden=true;updateBatchStatus();
  }
 }
};

$('resultVideo').onerror=()=>{if($('resultVideo').getAttribute('src'))message('mediaStatus',$('mediaStatus').textContent+'\n浏览器无法播放该结果，请下载查看或检查服务器日志。');};
$('quality').oninput=()=>{$('qualityValue').textContent=Math.round(Number($('quality').value)*100)+'%';};
let jpegWorker=null,jpegPending=null,workerFailed=false,lastJpegBytes=64*1024;
let encoderLabel='兼容编码';
const jpegWorkerSource=`
let canvas,context;
self.onmessage=async({data})=>{
 const bitmap=data.bitmap,g=data.geometry;
 try{
  if(bitmap.width!==g.sourceWidth||bitmap.height!==g.sourceHeight)throw new Error('视频位图方向与截图尺寸不一致');
  if(!canvas){canvas=new OffscreenCanvas(g.width,g.height);context=canvas.getContext('2d',{alpha:false});}
  if(canvas.width!==g.width||canvas.height!==g.height){canvas.width=g.width;canvas.height=g.height;}
  const drawStart=performance.now();
  context.fillStyle='#000';context.fillRect(0,0,g.width,g.height);
  context.drawImage(bitmap,0,0,g.sourceWidth,g.sourceHeight,g.dx,g.dy,g.dw,g.dh);bitmap.close();
  const encodeStart=performance.now();
  const blob=await canvas.convertToBlob({type:'image/jpeg',quality:data.quality});
  if(blob.type!=='image/jpeg')throw new Error('JPEG unsupported');
  self.postMessage({blob,workerDrawMs:encodeStart-drawStart,workerEncodeMs:performance.now()-encodeStart});
 }catch(error){bitmap.close();self.postMessage({error:String(error)});}
};`;
function disposeJpegWorker(){
 if(jpegWorker){jpegWorker.terminate();jpegWorker=null;}
 if(jpegPending){const pending=jpegPending;jpegPending=null;pending.reject(new Error('后台编码已停止'));}
}
$('jpegMode').onchange=()=>{disposeJpegWorker();workerFailed=false;};
function bufferLimit(){
 return $('bufferMode').value==='compatible'?MAX_BUFFERED_BYTES:
  Math.min(MAX_BUFFERED_BYTES,Math.max(64*1024,lastJpegBytes));
}
function encodeInWorker(video,quality,geometry){
 return new Promise((resolve,reject)=>{
  let finished=false;
  const timer=setTimeout(()=>finish(new Error('后台编码超时')),1500);
  const finish=(error,blob)=>{
   if(finished)return;finished=true;clearTimeout(timer);
   if(jpegPending===pending)jpegPending=null;
   if(error)reject(error);else resolve(blob);
  };
  const pending={reject:error=>finish(error)};jpegPending=pending;
  try{
   if(!window.Worker||!window.OffscreenCanvas||!window.createImageBitmap)throw new Error('浏览器不支持后台编码');
   if(!jpegWorker){
    const url=URL.createObjectURL(new Blob([jpegWorkerSource],{type:'text/javascript'}));
    try{jpegWorker=new Worker(url);}finally{URL.revokeObjectURL(url);}
   }
   const worker=jpegWorker;
   worker.onmessage=({data})=>finish(data.error?new Error(data.error):null,data);
   worker.onerror=event=>{event.preventDefault();finish(new Error('后台编码失败'));};
   worker.onmessageerror=()=>finish(new Error('后台编码消息失败'));
   createImageBitmap(video).then(bitmap=>{
    if(finished){bitmap.close();return;}
    try{worker.postMessage({bitmap,quality,geometry},[bitmap]);}catch(error){bitmap.close();finish(error);}
   },error=>finish(error));
  }catch(error){finish(error);}
 });
}
for(const id of ['mediaConf','liveConf']){
 const number=$(id),slider=$(id+'Slider');
 const changed=()=>{if(id==='liveConf')invalidateLiveView();};
 slider.oninput=()=>{number.value=Number(slider.value).toFixed(2);changed();};
 number.oninput=()=>{if(number.value==='')return;const value=Number(number.value);if(Number.isFinite(value)&&value>=0&&value<=1){slider.value=String(value);changed();}};
 number.onchange=()=>{const value=Number(number.value);number.value=(number.value!==''&&Number.isFinite(value)?Math.min(1,Math.max(0,value)):Number(slider.value)).toFixed(2);slider.value=number.value;changed();};
}

function enginesForAspect(value){
 const ratio=value==='4:3'?4/3:16/9;
 // 允许标称画幅与按步长对齐的引擎尺寸之间存在少量补边差异。
 return catalog.filter(e=>Math.abs(Math.log((e.w/e.h)/ratio))<.05);
}
function selectedLiveEngine(){return catalog.find(e=>e.key===$('liveEngine').value);}
function refreshLiveEngines(){
 const previous=$('liveEngine').value;
 const candidates=enginesForAspect($('liveAspect').value);
 const order={s:0,n:1,m:2,l:3,x:4};
 candidates.sort((a,b)=>b.h-a.h||b.w-a.w||(order[a.model]??99)-(order[b.model]??99));
 $('liveEngine').replaceChildren();
 for(const engine of candidates){const option=document.createElement('option');option.value=engine.key;option.textContent=`${engine.w}×${engine.h} · ${engine.model} 模型`;$('liveEngine').appendChild(option);}
 if(candidates.some(e=>e.key===previous))$('liveEngine').value=previous;
 else if(candidates.length){const preferred=[...candidates].sort((a,b)=>Math.abs(a.h-1080)-Math.abs(b.h-1080)||(order[a.model]??99)-(order[b.model]??99));$('liveEngine').value=preferred[0].key;}
 else{const option=document.createElement('option');option.value='';option.textContent='服务器没有该比例的模型';$('liveEngine').appendChild(option);}
 $('start').disabled=!!cameraStream||!candidates.length;
 invalidateLiveView();
}
// 配置或画幅变化时递增版本号，用于丢弃旧配置下尚未返回的检测结果。
function invalidateLiveView(){liveRevision++;lastDets=[];lastVideoTime=-1;geometryKey='';liveContext.clearRect(0,0,$('liveCanvas').width,$('liveCanvas').height);}
$('liveAspect').onchange=()=>{refreshLiveEngines();if(cameraStream)void startLive();};
$('liveEngine').onchange=()=>{invalidateLiveView();if(cameraStream)void startLive();};
$('liveOrientation').onchange=invalidateLiveView;
function serverRendering(){return $('liveRender').value==='server';}
function updateResultSurface(){
 const streamed=serverRendering()&&$('liveTransport').value==='webrtc';
 $('liveCanvas').hidden=streamed;$('serverVideo').hidden=!streamed;
}
$('liveRender').onchange=()=>{updateResultSurface();if(cameraStream)void startLive();};
let pendingServerJPEG=null,serverJPEGDecoding=false,paintedJPEGId=0;
// 显示端仅保留最新待解码图片；服务端 FIFO 处理不意味着浏览器逐帧展示所有结果。
function queueServerJPEG(data,jpeg,generation){
 pendingServerJPEG={data,jpeg,generation};
 if(serverJPEGDecoding)return;
 serverJPEGDecoding=true;
 void (async()=>{
  try{
   while(pendingServerJPEG){
    const item=pendingServerJPEG;pendingServerJPEG=null;
    let bitmap,url;
    try{
     const blob=new Blob([item.jpeg],{type:'image/jpeg'});
     if(window.createImageBitmap)bitmap=await createImageBitmap(blob);
     else{
      bitmap=new Image();url=URL.createObjectURL(blob);
      await new Promise((resolve,reject)=>{bitmap.onload=resolve;bitmap.onerror=reject;bitmap.src=url;});
     }
     if(item.generation!==liveGeneration||item.data.revision!==liveRevision||!serverRendering()||item.data.id<=paintedJPEGId)continue;
     if(item.data.width!==currentWidth||item.data.height!==currentHeight)continue;
     liveContext.drawImage(bitmap,0,0,currentWidth,currentHeight);paintedJPEGId=item.data.id;
    }catch(error){if(item.generation===liveGeneration)message('liveStatus','标注图片显示失败，请重新开始。');}
    finally{bitmap?.close?.();if(url)URL.revokeObjectURL(url);}
   }
  }finally{serverJPEGDecoding=false;}
 })();
}

function updateLiveGeometry(){
 const video=$('cameraVideo'),engine=selectedLiveEngine();
 if(!engine||!video.videoWidth||!video.videoHeight)return;
 const sourceWidth=video.videoWidth,sourceHeight=video.videoHeight;
 const aspect=$('liveAspect').value;
 const orientation=$('liveOrientation').value;
 const key=`${sourceWidth}:${sourceHeight}:${aspect}:${engine.key}:${orientation}`;
 if(key===geometryKey)return;
 const [baseW,baseH]=aspect==='4:3'?[4,3]:[16,9];
 // 以视频元素的实际尺寸为准；轨道设置中的宽高可能互换。
 const portrait=orientation==='portrait'||(orientation==='auto'&&sourceHeight>sourceWidth);
 const [rw,rh]=portrait?[baseH,baseW]:[baseW,baseH];
 // 在标称尺寸内按所选比例向下取整，竖屏时交换输出宽高；不旋转源像素。
 // 例如：1920×1088 引擎接收 1920×1080 的 JPEG 后，在 GPU 上补边。
 const unit=Math.max(1,Math.floor(Math.min(engine.w/baseW,engine.h/baseH)));
 const width=rw*unit,height=rh*unit;
 const fit=Math.min(width/sourceWidth,height/sourceHeight);
 const dw=sourceWidth*fit,dh=sourceHeight*fit;
 liveGeometry={dx:(width-dw)/2,dy:(height-dh)/2,dw,dh,
               width,height,sourceWidth,sourceHeight,
               aspectLabel:`${rw}:${rh}`,portrait};
 geometryKey=key;liveRevision++;lastDets=[];lastDetTime=0;lastVideoTime=-1;
 currentWidth=width;currentHeight=height;
 $('liveCanvas').width=width;$('liveCanvas').height=height;
 // 采用与上传 JPEG 相同的等比例容纳布局，显示完整视频。
 video.style.aspectRatio=`${rw} / ${rh}`;video.style.objectFit='contain';
 video.style.background='#000';
}
$('cameraVideo').addEventListener('resize',()=>{
 if(cameraStream)updateLiveGeometry();
});

let liveSentCount=0,lastStatusAt=0,cameraRequestLabel='';
function stopLive(){
 stopRtc();
 pendingServerJPEG=null;paintedJPEGId=0;updateResultSurface();
 liveContext.clearRect(0,0,$('liveCanvas').width,$('liveCanvas').height);
 disposeJpegWorker();workerFailed=false;lastJpegBytes=64*1024;
 liveGeneration++;liveReady=false;encoding=false;lastDets=[];lastVideoTime=-1;
 liveGeometry=null;geometryKey='';currentWidth=0;currentHeight=0;
 lastResultId=0;firstSentAt=0;lastResponseAt=0;
 if(socket){socket.onclose=null;socket.onmessage=null;socket.onerror=null;socket.close();socket=null;}
 if(cameraStream){cameraStream.getTracks().forEach(t=>t.stop());cameraStream=null;}
 $('cameraVideo').srcObject=null;$('start').disabled=!selectedLiveEngine();$('stop').disabled=true;
 message('liveStatus','已停止。');
}
function receiveLiveResult(event,generation){
 if(generation!==liveGeneration)return;
 let data,jpeg=null;
 try{
  if(event.data instanceof ArrayBuffer){
   if(event.data.byteLength<5)return;
   const size=new DataView(event.data).getUint32(0);
   if(size>65536||size+4>=event.data.byteLength)return;
   data=JSON.parse(new TextDecoder().decode(new Uint8Array(event.data,4,size)));
   jpeg=new Uint8Array(event.data,4+size);
  }else data=JSON.parse(event.data);
 }catch{return;}
 if(data.type==='init'){
  if(liveReady)return;
  liveReady=true;$('stop').disabled=false;lastStatsAt=performance.now();received=0;
  liveSentCount=0;lastStatusAt=0;
  void continuousCapture(generation);
  message('liveStatus',cameraRequestLabel+'\n已连接，持续发送最新画面（首次加载引擎可能稍慢）');return;
 }
 if(data.type!=='result'){if(data.error)message('liveStatus',data.error);return;}
 // 监测连接状态，但不让采集流程等待回复。
 const now=performance.now();lastResponseAt=now;
 // 会话编号在入口检查；这里再过滤旧画幅版本及重复、乱序结果。
 if(data.revision!==liveRevision||data.id<=lastResultId)return;
 lastResultId=data.id;
 if(data.error){message('liveStatus','检测失败：'+data.error);return;}
 if(data.width!==currentWidth||data.height!==currentHeight)return;
 const age=typeof data.captured_at==='number'?Math.max(0,now-data.captured_at):Infinity;
 if(data.render==='server'&&jpeg&&serverRendering())queueServerJPEG(data,jpeg,generation);
 lastDets=!serverRendering()&&age<=500?data.dets:[];
 lastDetTime=age<=500?data.captured_at:0;
 received++;
 const elapsed=(now-lastStatsAt)/1000;
 if(now-lastStatusAt<250)return; // 统计信息每秒更新 4 次，检测框仍随每次结果更新。
 lastStatusAt=now;
 const ms=value=>Number.isFinite(value)?value.toFixed(1)+'ms':'—';
 const roundtrip=Number.isFinite(data.sent_at)?now-data.sent_at:NaN;
 const transport=Number.isFinite(data.server_total_ms)?Math.max(0,roundtrip-data.server_total_ms):NaN;
 message('liveStatus',`${liveGeometry?.aspectLabel||$('liveAspect').value} · ${data.width}×${data.height} · JPEG ${Number.isFinite(data.jpeg_quality)?Math.round(data.jpeg_quality*100):'—'}% · ${Math.round(data.jpeg_bytes/1024)} KB\n截图 ${ms(data.draw_ms)} · JPEG 编码 ${ms(data.encode_ms)} · 截图→结果 ${ms(age)}\n发送→结果 ${ms(roundtrip)} · 传输与客户端调度估计 ${ms(transport)} · 发送前积压 ${Math.round((data.buffered_bytes||0)/1024)} KB\n服务器总耗时 ${ms(data.server_total_ms)}（排队 ${ms(data.queue_ms)} + 处理 ${ms(data.server_process_ms)}）\n${data.engine} · 解码/预处理 ${ms(data.decode_ms)} (${data.decoder||'CPU'}) · GPU 锁等待 ${ms(data.wait_ms)} · 预测调用 ${ms(data.infer_ms)}\n${(received/Math.max(elapsed,.01)).toFixed(1)} 检测 FPS · 顺序处理 · 当前待处理 ${data.pending_frames??0}/${data.queue_capacity??4} 帧`);
 const settings=cameraStream?.getVideoTracks()[0]?.getSettings();
 if(data.render==='server')$('liveStatus').textContent+=`\n服务器画框 ${ms(data.server_draw_ms)} · 回传 JPEG 编码 ${ms(data.server_encode_ms)} · 回传 ${Math.round(data.return_bytes/1024)} KB · 同帧标注`;
 $('liveStatus').textContent+=`\n实际发送 ${(liveSentCount/Math.max(elapsed,.01)).toFixed(1)} FPS · 摄像头协商 ${settings?.frameRate?.toFixed(1)??'—'} FPS（非实测采集率）\n${encoderLabel} · 缓冲阈值 ${Math.round(bufferLimit()/1024)} KB\n${cameraRequestLabel} · 视频帧 ${$('cameraVideo').videoWidth}×${$('cameraVideo').videoHeight}`;
 if(elapsed>5){lastStatsAt=now;received=0;liveSentCount=0;}
}
async function startLive(){
 stopLive();const generation=liveGeneration;
 try{
  const engine=selectedLiveEngine();if(!engine)throw new Error('请先选择服务器现有的模型分辨率');
  if(!navigator.mediaDevices?.getUserMedia)throw new Error('当前页面不能访问摄像头，请使用 HTTPS 或 localhost');
  $('start').disabled=true;$('stop').disabled=false;message('liveStatus','正在打开摄像头…');
  const device=$('camera').value;
  const [rw,rh]=$('liveAspect').value==='4:3'?[4,3]:[16,9];
  const unit=Math.max(1,Math.floor(Math.min(engine.w/rw,engine.h/rh)));
  // 初始设备方向仅作为请求提示；打开后以 videoWidth/videoHeight 为准。
  const orientation=$('liveOrientation').value;
  const portraitHint=orientation==='portrait'||(orientation==='auto'&&window.matchMedia('(pointer: coarse)').matches&&
   (screen.orientation?.type?.startsWith('portrait')??window.matchMedia('(orientation: portrait)').matches));
  const requestedWidth=(portraitHint?rh:rw)*unit,requestedHeight=(portraitHint?rw:rh)*unit;
  const targetFps=$('liveTransport').value==='webrtc'?Number($('rtcFps').value):60;
  const constraints={width:{exact:requestedWidth},height:{exact:requestedHeight},frameRate:{exact:targetFps}};
  if(device)constraints.deviceId={exact:device};else constraints.facingMode={ideal:'environment'};
  let acquired,fallback=false;
  const relaxed={...constraints,frameRate:{ideal:targetFps,max:targetFps}};
  const swapped={...constraints,width:{exact:requestedHeight},height:{exact:requestedWidth}};
  // 先尝试精确尺寸和交换宽高，再放宽帧率、尺寸；ideal 回退不保证分辨率更低。
  const attempts=[constraints,swapped,relaxed,{...swapped,frameRate:relaxed.frameRate},
   {...relaxed,width:{ideal:requestedWidth},height:{ideal:requestedHeight}}];
  for(let index=0;index<attempts.length;index++){
   if(generation!==liveGeneration)return;
   try{acquired=await navigator.mediaDevices.getUserMedia({video:attempts[index],audio:false});fallback=index>0;break;}
   catch(error){if(error.name!=='OverconstrainedError'||index===attempts.length-1)throw error;}
  }
  if(generation!==liveGeneration){acquired.getTracks().forEach(t=>t.stop());return;}
  cameraStream=acquired;$('cameraVideo').srcObject=acquired;
  const track=acquired.getVideoTracks()[0];
  let rateNote='';
  if($('liveTransport').value==='webrtc'){try{track.contentHint='detail';}catch{}}
  const actual=track.getSettings();
  let caps={};try{caps=track.getCapabilities?.()||{};}catch{}
  const reportedFps=Number.isFinite(actual.frameRate)?actual.frameRate.toFixed(1):'未提供';
  cameraRequestLabel=`请求 ${requestedWidth}×${requestedHeight} / ${targetFps} FPS · 轨道报告 ${actual.width??'?'}×${actual.height??'?'} / ${reportedFps} FPS${fallback?'（已使用兼容协商）':''}${rateNote}`;
  if(Number.isFinite(caps.frameRate?.max))cameraRequestLabel+=` · 设备声明帧率上限 ${caps.frameRate.max}（不代表此尺寸可用）`;
  message('liveStatus',cameraRequestLabel+'\n正在连接服务器…');
  await $('cameraVideo').play();if(generation!==liveGeneration)return;
  const devices=await navigator.mediaDevices.enumerateDevices();if(generation!==liveGeneration)return;
  const selected=acquired.getVideoTracks()[0].getSettings().deviceId;
  $('camera').replaceChildren();
  devices.filter(d=>d.kind==='videoinput').forEach((d,i)=>{const o=document.createElement('option');o.value=d.deviceId;o.textContent=d.label||'摄像头 '+(i+1);$('camera').appendChild(o);});
  if(selected)$('camera').value=selected;
  if($('liveTransport').value==='webrtc'){
   updateLiveGeometry();
   await startRtc(acquired,generation);
   return;
  }
  const connection=new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws');
  connection.binaryType='arraybuffer';
  socket=connection;
  connection.onmessage=e=>receiveLiveResult(e,generation);
  connection.onclose=()=>{if(generation===liveGeneration){stopLive();message('liveStatus','连接已断开，请重新开始。');}};
  connection.onerror=()=>{if(generation===liveGeneration){stopLive();message('liveStatus','WebSocket 连接失败。');}};
 }catch(e){if(generation===liveGeneration){stopLive();message('liveStatus',e.message);}}
}
$('start').onclick=startLive;$('stop').onclick=stopLive;
$('camera').onchange=()=>{if(cameraStream)startLive();};

function drawBoxes(){
 const font=Math.max(22,Math.round(currentWidth/45));
 liveContext.font=`700 ${font}px sans-serif`;liveContext.lineWidth=Math.max(3,currentWidth/320);
 for(const d of lastDets){
  const [x1,y1,x2,y2]=d.box,color=`hsl(${d.cls*137.508%360} 90% 60%)`;
  liveContext.strokeStyle=color;liveContext.strokeRect(x1,y1,x2-x1,y2-y1);
  const label=`${d.name} ${(d.conf*100).toFixed(0)}%`,y=Math.max(font+6,y1);
  liveContext.fillStyle=color;liveContext.fillRect(x1,y-font-6,liveContext.measureText(label).width+10,font+6);
  liveContext.fillStyle='#07101d';liveContext.fillText(label,x1+5,y-4);
 }
}
async function sendCapture(){
 if(!liveReady||encoding||!liveGeometry||!socket||socket.readyState!==WebSocket.OPEN||socket.bufferedAmount>bufferLimit())return;
 const video=$('cameraVideo');if(video.readyState<2||video.currentTime===lastVideoTime)return;
 const connection=socket,generation=liveGeneration,revision=liveRevision,geometry=liveGeometry;
 const engine=$('liveEngine').value,quality=Number($('quality').value),capturedAt=performance.now();
 encoding=true;lastVideoTime=video.currentTime;
 try{
  const conf=confidence('liveConf');
  let encodeStarted=performance.now(),drawMs=0,blob;
  if($('jpegMode').value==='worker'&&!workerFailed){
   try{
    const result=await encodeInWorker(video,quality,geometry);blob=result.blob;
    encoderLabel=`后台直取视频 · 后台绘图 ${result.workerDrawMs.toFixed(1)}ms · JPEG ${result.workerEncodeMs.toFixed(1)}ms`;
   }
   catch(error){
    if(generation!==liveGeneration)return;
    disposeJpegWorker();workerFailed=true;encoderLabel='后台不可用，已回退兼容编码';
   }
  }
  if(generation!==liveGeneration)return;
  if(!blob){
   if(!workerFailed)encoderLabel='兼容编码';
   // 回退时保留原有画布流程；工作线程模式避免重复绘制。
   if(capture.width!==geometry.width||capture.height!==geometry.height){capture.width=geometry.width;capture.height=geometry.height;}
   const drawStart=performance.now();
   captureContext.fillStyle='#000';captureContext.fillRect(0,0,geometry.width,geometry.height);
   captureContext.drawImage(video,0,0,geometry.sourceWidth,geometry.sourceHeight,
                           geometry.dx,geometry.dy,geometry.dw,geometry.dh);
   drawMs=performance.now()-drawStart;
   encodeStarted=performance.now();
   blob=await new Promise(resolve=>capture.toBlob(resolve,'image/jpeg',quality));
  }
  if(!blob)return;
  const encodeMs=performance.now()-encodeStarted;
  if(generation!==liveGeneration||revision!==liveRevision||connection!==socket||connection.readyState!==WebSocket.OPEN||connection.bufferedAmount>bufferLimit())return;
  lastJpegBytes=blob.size;
  const id=++sequence;
  // 不设置未确认帧计数或确认应答门槛，每张 JPEG 都随附对应元数据。
  connection.send(JSON.stringify({type:'frame',id,revision,captured_at:capturedAt,
   sent_at:performance.now(),draw_ms:drawMs,encode_ms:encodeMs,
   jpeg_quality:quality,buffered_bytes:connection.bufferedAmount,engine,conf,render:$('liveRender').value}));
  connection.send(blob); // WebSocket 可直接发送 Blob，省去复制到 ArrayBuffer 的步骤。
  liveSentCount++;
  if(!firstSentAt)firstSentAt=performance.now();
  return true;
 }catch(e){if(generation===liveGeneration)message('liveStatus',e.message);}
 finally{if(generation===liveGeneration)encoding=false;}
}
async function continuousCapture(generation){
 while(generation===liveGeneration&&liveReady){
  updateLiveGeometry();
  const sent=await sendCapture();
  // 编码时已让出执行权给浏览器；成功发送一帧后无需额外定时等待。
  // 没有新帧或连接发送积压时，短暂等待，避免空转。
  if(!sent)await sleep(4);
 }
}
let renderedSignature='';
function render(){
 if(liveReady&&cameraStream&&$('cameraVideo').readyState>=2){
  updateLiveGeometry();
  if(rtcPeer)syncRtcConfig();
  if(liveGeometry&&!serverRendering()){
   // 客户端将近期检测框叠加到当前摄像头画面；严格同帧标注需使用服务器画框模式。
   const g=liveGeometry,video=$('cameraVideo'),fresh=performance.now()-lastDetTime<500;
   const signature=`${liveGeneration}:${liveRevision}:${video.currentTime}:${lastDetTime}:${fresh}`;
   if(signature!==renderedSignature){
    renderedSignature=signature;
    liveContext.fillStyle='#000';liveContext.fillRect(0,0,g.width,g.height);
    liveContext.drawImage(video,0,0,g.sourceWidth,g.sourceHeight,g.dx,g.dy,g.dw,g.dh);
    if(fresh)drawBoxes();
   }
  }
 }
 if(liveReady){
  const since=lastResponseAt||firstSentAt;
  if(since&&performance.now()-since>60000){stopLive();message('liveStatus','服务器超过 60 秒没有返回，请检查日志后重新连接。');}

 }
 requestAnimationFrame(render);
}
requestAnimationFrame(render);
window.addEventListener('pagehide',()=>{stopLive();if(sourceURL)URL.revokeObjectURL(sourceURL);});

function rtcTargetFps(){return Number($('rtcFps').value)||60;}
let rtcPeer=null,rtcChannel=null,rtcSession=null,rtcTimer=null,rtcFetch=null;
let rtcRevision=-1,rtcRtt=null,rtcStatsText='',rtcPrevious=null,rtcStatsBusy=false;
let rtcSenderNote='',rtcPreviousSource=null;
let rtcStartAt=0,rtcFirstResultMs=null,rtcFirstVideoMs=null;
let rtcPreparedReceiver=null;
function prepareRtcReceiver(generation){
 const peer=new RTCPeerConnection({iceServers:[]});
 const transceiver=peer.addTransceiver('video',{direction:'recvonly'});
 preferLowDelay(transceiver.receiver);
 const codecs=RTCRtpReceiver.getCapabilities?.('video')?.codecs?.filter(c=>c.mimeType.toLowerCase()==='video/h264');
 if(codecs?.length&&transceiver.setCodecPreferences)transceiver.setCodecPreferences(codecs);
 const ready=(async()=>{await peer.setLocalDescription(await peer.createOffer());await iceReady(peer);})();
 ready.catch(()=>{}); // 此处先接住拒绝状态，后续使用预建接收器时再处理错误。
 rtcPreparedReceiver={peer,transceiver,ready,generation};
}
function preferLowDelay(receiver){
 try{if('jitterBufferTarget' in receiver)receiver.jitterBufferTarget=0;}catch{}
}
async function configureRtcSender(sender){
 const apply=async withPreference=>{
  const p=sender.getParameters();
  if(!p.encodings?.length)throw new Error('浏览器未提供发送编码参数');
  for(const e of p.encodings){e.maxFramerate=rtcTargetFps();e.maxBitrate=Number($('rtcBitrate').value);e.scaleResolutionDownBy=1;}
  if(withPreference)p.degradationPreference='maintain-resolution';
  await sender.setParameters(p);
 };
 let preference=true;
 try{await apply(true);}
 catch(error){
  preference=false;
  try{await apply(false);}
  catch(second){rtcSenderNote='浏览器拒绝所选帧率发送参数：'+second.message;return;}
 }
 const actual=sender.getParameters();
 const confirmed=actual.degradationPreference==='maintain-resolution';
 rtcSenderNote=`发送帧率上限 ${actual.encodings?.[0]?.maxFramerate??'浏览器未回报'} FPS · ${confirmed?'分辨率优先（带宽不足时可能降帧率）':preference?'已请求分辨率优先，浏览器未确认':'浏览器使用默认自适应策略'}`;
}
let rtcPlayback=null,rtcPlaybackStatus='等待服务器标注流';
function stopRtcPlayback(){
 const old=rtcPlayback;rtcPlayback=null;
 const video=$('serverVideo');video.pause();video.srcObject=null;
 video.onplaying=null;
 if(old){
  old.controller.abort();clearInterval(old.timer);
  old.peer.ontrack=null;old.peer.onconnectionstatechange=null;old.peer.close();
  if(old.id)void fetch(`/api/webrtc/${old.session}/playback/${old.id}`,{method:'DELETE',keepalive:true}).catch(()=>{});
 }
 rtcPlaybackStatus='等待服务器标注流';
}
function ensureRtcPlayback(epoch,generation){
 if(!serverRendering()||!rtcSession||generation!==liveGeneration)return;
 if(rtcPlayback?.epoch===epoch)return;
 stopRtcPlayback();
 const prepared=rtcPreparedReceiver?.generation===generation?rtcPreparedReceiver:null;
 if(prepared)rtcPreparedReceiver=null;
 const peer=prepared?.peer||new RTCPeerConnection({iceServers:[]});
 const state={peer,epoch,generation,session:rtcSession,controller:new AbortController(),id:null,
              started:performance.now(),lastFrame:performance.now(),previous:null,busy:false,timer:null};
 rtcPlayback=state;rtcPlaybackStatus='正在连接服务器标注视频…';
 const current=()=>rtcPlayback===state&&generation===liveGeneration;
 const fail=error=>{if(current()){stopLive();message('liveStatus','标注视频回传失败：'+error.message+'；可切换客户端画框。');}};
 peer.ontrack=event=>{
  if(!current())return;
  preferLowDelay(event.receiver);
  const video=$('serverVideo');video.srcObject=event.streams[0]||new MediaStream([event.track]);
  video.onplaying=()=>{if(current()&&rtcFirstVideoMs===null)rtcFirstVideoMs=performance.now()-rtcStartAt;};
  video.style.objectFit='contain';
  void video.play().catch(fail);
 };
 peer.onconnectionstatechange=()=>{
  if(current()&&peer.connectionState==='connected'&&rtcChannel?.readyState===WebSocket.OPEN)
   rtcChannel.send(JSON.stringify({type:'playback_ready'}));
  if(current()&&['failed','closed'].includes(peer.connectionState))fail(new Error('WebRTC播放连接断开'));
 };
 const transceiver=prepared?.transceiver||peer.addTransceiver('video',{direction:'recvonly'});
 preferLowDelay(transceiver.receiver);
 const codecs=RTCRtpReceiver.getCapabilities?.('video')?.codecs?.filter(c=>c.mimeType.toLowerCase()==='video/h264');
 if(!prepared&&codecs?.length&&transceiver.setCodecPreferences)transceiver.setCodecPreferences(codecs);
 state.timer=setInterval(async()=>{
  if(!current()||state.busy)return;
  if(performance.now()-state.started>25000&&(peer.connectionState!=='connected'||$('serverVideo').readyState<2)){
   fail(new Error('播放建立超时，请检查UDP 8189和服务器日志'));return;
  }
  state.busy=true;
  try{
   const stats=await peer.getStats();if(!current())return;
   for(const report of stats.values()){
    if(report.type!=='inbound-rtp'||(report.kind||report.mediaType)!=='video')continue;
    if(!Number.isFinite(report.framesDecoded)){rtcPlaybackStatus='当前浏览器不提供回传解码 FPS';continue;}
    const previous=state.previous;state.previous=report;
    if(previous&&report.timestamp>previous.timestamp){
     const frames=(report.framesDecoded??0)-(previous.framesDecoded??0);
     const seconds=(report.timestamp-previous.timestamp)/1000;
     if(frames>0)state.lastFrame=performance.now();
     const emitted=report.jitterBufferEmittedCount-previous.jitterBufferEmittedCount;
     const jitterMs=emitted>0&&(Number.isFinite(report.jitterBufferDelay)&&Number.isFinite(previous.jitterBufferDelay))?
      ((report.jitterBufferDelay-previous.jitterBufferDelay)*1000/emitted).toFixed(1)+'ms':'未提供';
     rtcPlaybackStatus=`回传视频解码 ${(frames/seconds).toFixed(1)} FPS · ${report.frameWidth||'?'}×${report.frameHeight||'?'} · 播放抖动缓冲 ${jitterMs} · 目标 ${rtcTargetFps()} FPS`;
     if(performance.now()-state.lastFrame>20000)fail(new Error('回传视频20秒没有新帧'));
    }
   }
  }catch(error){if(current())rtcPlaybackStatus='浏览器暂未提供回传帧率统计';}
  finally{state.busy=false;}
 },1000);
 void (async()=>{
  try{
   if(prepared)await prepared.ready;
   else{await peer.setLocalDescription(await peer.createOffer());await iceReady(peer);}
   if(!current())return;
   const timeout=setTimeout(()=>state.controller.abort(),15000);
   let answer;
   try{
    const response=await fetch(`/api/webrtc/${state.session}/playback`,{
     method:'POST',headers:{'Content-Type':'application/json'},signal:state.controller.signal,
     body:JSON.stringify({type:'offer',sdp:peer.localDescription.sdp,epoch})});
    if(response.status===409){if(current())stopRtcPlayback();return;}
    answer=await jsonResponse(response);
   }finally{clearTimeout(timeout);}
   if(!current()){
    void fetch(`/api/webrtc/${state.session}/playback/${answer.playback_id}`,{method:'DELETE',keepalive:true}).catch(()=>{});return;
   }
   state.id=answer.playback_id;
   await peer.setRemoteDescription({type:answer.type,sdp:answer.sdp});
  }catch(error){fail(error);}
 })();
}
function stopRtc(){
 stopRtcPlayback();
 if(rtcPreparedReceiver){rtcPreparedReceiver.peer.close();rtcPreparedReceiver=null;}
 if(rtcTimer){clearInterval(rtcTimer);rtcTimer=null;}
 if(rtcFetch){rtcFetch.abort();rtcFetch=null;}
 if(rtcChannel){rtcChannel.onmessage=null;rtcChannel.onclose=null;rtcChannel.onerror=null;rtcChannel.close();rtcChannel=null;}
 if(rtcPeer){rtcPeer.onconnectionstatechange=null;rtcPeer.close();rtcPeer=null;}
 if(rtcSession){void fetch('/api/webrtc/'+rtcSession,{method:'DELETE',keepalive:true}).catch(()=>{});rtcSession=null;}
 rtcRevision=-1;rtcRtt=null;rtcStatsText='等待浏览器编码统计';rtcPrevious=null;rtcStatsBusy=false;
 rtcPreviousSource=null;rtcSenderNote='';
 rtcStartAt=0;rtcFirstResultMs=null;rtcFirstVideoMs=null;
}
function syncRtcConfig(){
 if(liveGeometry&&rtcChannel?.readyState===WebSocket.OPEN&&rtcRevision!==liveRevision){
  rtcChannel.send(JSON.stringify({type:'config',revision:liveRevision,engine:$('liveEngine').value,conf:confidence('liveConf'),width:liveGeometry.width,height:liveGeometry.height,render:$('liveRender').value,bitrate:Number($('rtcBitrate').value),fps:rtcTargetFps(),queue_policy:$('rtcQueue').value}));
  rtcRevision=liveRevision;
 }
}
function updateTransportControls(){
 const rtc=$('liveTransport').value==='webrtc';
 for(const id of ['quality','jpegMode','bufferMode'])$(id).disabled=rtc;
 $('rtcBitrate').disabled=!rtc;
 $('rtcFps').disabled=!rtc;
 $('rtcQueue').disabled=!rtc;
 updateResultSurface();
}
$('liveTransport').onchange=()=>{if(cameraStream)stopLive();updateTransportControls();};
updateTransportControls();
$('rtcFps').onchange=()=>{if(cameraStream&&rtcPeer)void startLive();};
$('rtcBitrate').onchange=()=>{if(cameraStream&&rtcPeer)void startLive();};
$('rtcQueue').onchange=()=>{if(cameraStream&&rtcPeer)void startLive();};
function iceReady(peer){
 if(peer.iceGatheringState==='complete')return Promise.resolve();
 return new Promise((resolve,reject)=>{
  const finish=error=>{clearTimeout(timer);peer.removeEventListener('icegatheringstatechange',changed);error?reject(error):resolve();};
  const changed=()=>{if(peer.iceGatheringState==='complete')finish();};
  const timer=setTimeout(()=>finish(new Error('ICE收集超时，请切回JPEG或检查网络')),10000);
  peer.addEventListener('icegatheringstatechange',changed);
 });
}
async function rtcStatistics(peer,generation){
 if(rtcStatsBusy)return;rtcStatsBusy=true;
 try{
  const stats=await peer.getStats();if(generation!==liveGeneration||peer!==rtcPeer)return;
  for(const report of stats.values()){
   if(report.type!=='outbound-rtp'||(report.kind||report.mediaType)!=='video')continue;
   const previous=rtcPrevious;rtcPrevious=report;
   const source=stats.get(report.mediaSourceId)||[...stats.values()].find(s=>s.type==='media-source'&&s.kind==='video');
   let sourceFps=source?.framesPerSecond;
   if(source&&rtcPreviousSource?.id===source.id&&source.timestamp>rtcPreviousSource.timestamp&&Number.isFinite(source.frames)&&Number.isFinite(rtcPreviousSource.frames))
    sourceFps=(source.frames-rtcPreviousSource.frames)*1000/(source.timestamp-rtcPreviousSource.timestamp);
   rtcPreviousSource=source;
   if(previous&&report.timestamp>previous.timestamp){
    const seconds=(report.timestamp-previous.timestamp)/1000;
    const frames=report.framesEncoded-previous.framesEncoded;
    const sentFps=(report.framesSent-previous.framesSent)/seconds;
    const encoding=Number.isFinite(report.totalEncodeTime)&&Number.isFinite(previous.totalEncodeTime)&&frames>0?
     ((report.totalEncodeTime-previous.totalEncodeTime)*1000/frames).toFixed(1)+'ms':'—';
    const codec=stats.get(report.codecId)?.mimeType||'协商中';
    const f=value=>Number.isFinite(value)?value.toFixed(1):'未提供';
    const reasons={none:'无',cpu:'设备计算/编码能力',bandwidth:'带宽',other:'其他'};
    const limitation=reasons[report.qualityLimitationReason]||'浏览器未提供';
    rtcStatsText=`视频源 ${f(sourceFps)} FPS · 编码 ${f(frames/seconds)} FPS · 实际发送 ${f(sentFps)} FPS\n${codec} · 编码平均 ${encoding} · 发送 ${((report.bytesSent-previous.bytesSent)*8/seconds/1e6).toFixed(2)} Mbps · ${report.frameWidth||'?'}×${report.frameHeight||'?'}\n浏览器报告限制：${limitation}${report.encoderImplementation?' · '+report.encoderImplementation:''}\n${rtcSenderNote}\n${cameraRequestLabel}`;
    const transport=stats.get(report.transportId),pair=stats.get(transport?.selectedCandidatePairId);
    if(pair)rtcStatsText+=`\n上行可用带宽估计 ${Number.isFinite(pair.availableOutgoingBitrate)?(pair.availableOutgoingBitrate/1e6).toFixed(2)+' Mbps':'未提供'} · WebRTC往返 ${Number.isFinite(pair.currentRoundTripTime)?(pair.currentRoundTripTime*1000).toFixed(1)+'ms':'未提供'}`;
    const currentTrack=peer.getSenders().find(s=>s.track?.kind==='video')?.track?.getSettings();
    if(currentTrack)rtcStatsText+=`\n当前轨道协商 ${currentTrack.width??'?'}×${currentTrack.height??'?'} / ${f(currentTrack.frameRate)} FPS`;
   }
  }
 }catch{}finally{if(generation===liveGeneration)rtcStatsBusy=false;}
}
async function startRtc(stream,generation){
 if(!window.RTCPeerConnection)throw new Error('浏览器不支持WebRTC，请选择JPEG');
 if(!liveGeometry)throw new Error('摄像头尚未提供视频尺寸，请重新开始');
 rtcStartAt=performance.now();rtcFirstResultMs=null;rtcFirstVideoMs=null;
 if(serverRendering())prepareRtcReceiver(generation);
 const peer=new RTCPeerConnection({iceServers:[]});rtcPeer=peer;
 let channel;
 const track=stream.getVideoTracks()[0];
 try{track.contentHint='detail';}catch{}
 let transceiver;
 try{transceiver=peer.addTransceiver(track,{direction:'sendonly',sendEncodings:[{maxFramerate:rtcTargetFps(),maxBitrate:Number($('rtcBitrate').value),scaleResolutionDownBy:1}]});}
 catch(error){transceiver=peer.addTransceiver(track,{direction:'sendonly'});}
 const codecs=RTCRtpSender.getCapabilities?.('video')?.codecs;
 if(codecs&&transceiver.setCodecPreferences){
  const h264=codecs.filter(codec=>codec.mimeType.toLowerCase()==='video/h264');
  if(!h264.length)throw new Error('浏览器没有H.264编码器，请选择JPEG');
  transceiver.setCodecPreferences(h264);
 }
 if(generation!==liveGeneration){peer.close();return;}
 peer.onconnectionstatechange=()=>{
  if(generation===liveGeneration&&['failed','closed'].includes(peer.connectionState)){
   stopLive();message('liveStatus','WebRTC连接失败或关闭，请切回JPEG；局域网需允许UDP互通。');
  }
 };
 const offer=await peer.createOffer();if(generation!==liveGeneration)return;
 // 不协商 RTP 摄像头方向扩展，因为 RTSP 无法保留该元数据。
 offer.sdp=offer.sdp.split('\r\n').filter(line=>!(line.startsWith('a=extmap:')&&line.includes('urn:3gpp:video-orientation'))).join('\r\n');
 await peer.setLocalDescription(offer);await iceReady(peer);if(generation!==liveGeneration)return;
 const controller=new AbortController();rtcFetch=controller;
 message('liveStatus',cameraRequestLabel+'\n正在预热所选画幅并连接视频流，预热期间尚未上传视频…');
 const timeout=setTimeout(()=>controller.abort(),60000);
 let answer;
 try{
  answer=await jsonResponse(await fetch('/api/webrtc/offer',{method:'POST',headers:{'Content-Type':'application/json'},signal:controller.signal,
   body:JSON.stringify({sdp:peer.localDescription.sdp,type:peer.localDescription.type,config:{engine:$('liveEngine').value,conf:confidence('liveConf'),revision:liveRevision,width:liveGeometry.width,height:liveGeometry.height,render:$('liveRender').value,bitrate:Number($('rtcBitrate').value),fps:rtcTargetFps(),queue_policy:$('rtcQueue').value}})}));
 }finally{clearTimeout(timeout);if(rtcFetch===controller)rtcFetch=null;}
 if(generation!==liveGeneration){void fetch('/api/webrtc/'+answer.session_id,{method:'DELETE'}).catch(()=>{});return;}
 rtcSession=answer.session_id;await peer.setRemoteDescription({sdp:answer.sdp,type:answer.type});
 if(generation!==liveGeneration)return;
 await configureRtcSender(transceiver.sender);
 if(generation!==liveGeneration)return;
 channel=new WebSocket(`${location.protocol==='https:'?'wss:':'ws:'}//${location.host}/api/webrtc/${rtcSession}/results`);rtcChannel=channel;
 channel.onopen=()=>{
  if(generation!==liveGeneration)return;
  liveReady=true;lastStatsAt=performance.now();received=0;lastResponseAt=performance.now();
  syncRtcConfig();message('liveStatus','WebRTC已连接：原生视频轨道发送，JPEG设置不参与。');
 };
 channel.onmessage=event=>{
  if(generation!==liveGeneration)return;
  let data;try{data=JSON.parse(event.data);}catch{return;}
  const now=performance.now();lastResponseAt=now;
  if(data.type==='error'){stopLive();message('liveStatus',data.error);return;}
  if(data.type==='pong'){if(Number.isFinite(data.sent_at))rtcRtt=now-data.sent_at;return;}
  if(data.type!=='rtc_result'||data.revision!==liveRevision||data.id<=lastResultId)return;
  const g=liveGeometry;if(!g)return;
  // 不根据轨道设置旋转检测框；画面方向不匹配时拒绝使用结果。
  if(Math.abs(Math.log((data.source_width/data.source_height)/(g.sourceWidth/g.sourceHeight)))>.03){
   lastDets=[];
   message('liveStatus',`等待视频方向同步：本地 ${g.sourceWidth}×${g.sourceHeight}，服务器 ${data.source_width}×${data.source_height}。若持续不一致，请停止后重新开始或切换 JPEG。`);
   return;
  }
  lastResultId=data.id;
  if(rtcFirstResultMs===null)rtcFirstResultMs=now-rtcStartAt;
  if(data.render==='server'&&data.output_ready)ensureRtcPlayback(data.output_epoch,generation);
  lastDets=serverRendering()?[]:data.dets.map(d=>({...d,box:[d.box[0]*g.width,d.box[1]*g.height,d.box[2]*g.width,d.box[3]*g.height]}));
  lastDetTime=now;received++;
  if(now-lastStatusAt<250)return;lastStatusAt=now;
  const elapsed=Math.max(.01,(now-lastStatsAt)/1000);
  message('liveStatus',`WebRTC / MediaMTX · ${g.aspectLabel} · 显示 ${g.width}×${g.height} · 接收视频 ${data.source_width}×${data.source_height}\n${rtcStatsText}\n检测 ${(received/elapsed).toFixed(1)} FPS · 预测 ${data.infer_ms}ms · BGR转换 ${data.convert_ms}ms · 锁等待 ${data.wait_ms}ms\n解码取帧后耗时 ${data.server_ms}ms · 应用排队 ${data.queue_ms}ms · 结果通道往返 ${rtcRtt===null?'—':rtcRtt.toFixed(1)+'ms'}\n${data.decoder} · ${data.queue_policy==='latest'?'低延迟待处理':'FIFO待处理'} ${data.pending}/${data.queue_capacity??1} 帧${data.queue_policy==='latest'?' · 已丢弃待处理旧帧 '+(data.replaced??0):''}${data.fallback?" · 解码回退："+data.fallback:""}\n连接前画幅预热 ${data.warmup_ms??'—'}ms（未计入视频延迟）\nJPEG质量/缓冲设置不参与；往返时间不是视频端到端延迟。`);
  if(elapsed>5){lastStatsAt=now;received=0;}
  $('liveStatus').textContent+=`\n启动：解码器首帧 ${data.decoder_startup_ms??'—'}ms · 编码器初始化 ${data.encoder_init_ms??'不使用'}ms · WebRTC启动→首结果 ${rtcFirstResultMs?.toFixed(0)??'—'}ms · →首播放 ${rtcFirstVideoMs?.toFixed(0)??'等待中'}ms`;
  if(data.render==='server')$('liveStatus').textContent+=`\n服务器同帧画框 ${data.server_draw_ms}ms · ${data.output_encoder||'初始化'} 编码/发布 ${data.server_encode_ms??'—'}ms · 回传编码目标 ${rtcTargetFps()} FPS\n${rtcPlaybackStatus}${data.output_note?' · '+data.output_note:''}`;
 };
 channel.onclose=()=>{if(generation===liveGeneration){stopLive();message('liveStatus','WebRTC结果通道关闭，请重新开始。');}};
 channel.onerror=()=>{if(generation===liveGeneration){stopLive();message('liveStatus','WebRTC结果通道连接失败。');}};
 const connectedAt=performance.now();
 rtcTimer=setInterval(()=>{
  if(generation!==liveGeneration)return;
  if((!liveReady||peer.connectionState!=='connected')&&performance.now()-connectedAt>20000){stopLive();message('liveStatus','WebRTC建立超时，请检查UDP互通或使用JPEG。');return;}
  if(channel.readyState===WebSocket.OPEN){
   syncRtcConfig();channel.send(JSON.stringify({type:'ping',sent_at:performance.now()}));
   void rtcStatistics(peer,generation);
  }
 },1000);
}

</script>
</body>
</html>

'''

"""YOLO 双流水线服务器，在 GPU 服务器运行：python app_ws-multi.py。

检测逻辑、媒体流水线与网页已内嵌在本文件中，无需额外的拆分模块。
ENGINE_DIR 默认指向脚本目录，可通过环境变量指定已有引擎的存放目录。
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
import json
import os
import shutil
import tempfile
import threading
import time
import traceback
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from starlette.background import BackgroundTask
import uvicorn


BASE = Path(__file__).resolve().parent
ENGINE_DIR = Path(os.environ.get('ENGINE_DIR', BASE))
jobs = {}
jobs_lock = threading.Lock()
media_executor = ThreadPoolExecutor(max_workers=max(1, min(4, int(os.environ.get('MEDIA_WORKERS', '4')))), thread_name_prefix='media')
pool = None
live_pipeline = None


# 清理结束超过六小时且没有下载者的任务，下载期间通过 readers 计数保护结果。
def cleanup_jobs():
    with jobs_lock:
        expired = [job_id for job_id, job in jobs.items()
                   if job['status'] in ('done', 'error', 'cancelled') and not job.get('readers', 0)
                   and time.time() - job['updated'] > 6 * 3600]
        for job_id in expired:
            job = jobs.pop(job_id)
            # 仅删除本进程创建的任务目录，不使用用户提供的路径。
            shutil.rmtree(job['directory'], ignore_errors=True)


# 服务接收请求前初始化并预热引擎；关闭时回收会话、工作线程及临时文件。
@asynccontextmanager
async def lifespan(app):
    global pool, live_pipeline
    pool = await asyncio.to_thread(EnginePool, ENGINE_DIR,
                                   int(os.environ.get('ENGINE_CACHE_SIZE', '3')))
    await asyncio.to_thread(pool.warm_all, runs=5, keep_all='ENGINE_CACHE_SIZE' not in os.environ)
    live_pipeline = RealtimePipeline(pool)

    async def housekeeping():
        while True:
            await asyncio.sleep(10)
            await asyncio.to_thread(cleanup_jobs)
            await webrtc_reap()

    task = asyncio.create_task(housekeeping())
    yield
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await webrtc_shutdown()
    await asyncio.to_thread(media_executor.shutdown, wait=True)
    for job in jobs.values():
        shutil.rmtree(job['directory'], ignore_errors=True)


app = FastAPI(lifespan=lifespan)
webrtc_shutdown, webrtc_reap = install_webrtc(app, lambda: pool)


def update_job(job_id, **values):
    with jobs_lock:
        jobs[job_id].update(values, updated=time.time())


# 后台处理上传文件；通过检查点协作取消任务，并统一更新成功、失败或取消状态。
def run_media_job(job_id, kind, source, result, options):
    update_job(job_id, status='processing', stage='decoding')
    started = time.perf_counter()
    def check_cancel():
        with jobs_lock:
            cancelled = jobs[job_id].get('cancel_requested', False)
        if cancelled:
            raise RuntimeError('任务已停止')
    def report(**values):
        check_cancel()
        update_job(job_id, **values)
    options = dict(options, check_cancel=check_cancel)
    try:
        check_cancel()
        processor = process_image if kind == 'image' else process_video
        metadata = processor(source, result, pool, options,
                             report)
        check_cancel()
        update_job(job_id, status='done', stage='done', metadata=metadata,
                   elapsed=round(time.perf_counter() - started, 2))
        print(f'[media/{kind}] {job_id}: {json.dumps(metadata, ensure_ascii=False)}', flush=True)
    except Exception as exc:
        traceback.print_exc()
        with jobs_lock:
            cancelled = jobs[job_id].get('cancel_requested', False)
        update_job(job_id, status='cancelled' if cancelled else 'error',
                   stage='cancelled' if cancelled else 'error', error=str(exc))
    finally:
        try:
            Path(source).unlink(missing_ok=True)
        except OSError:
            pass


# 验证参数并预留任务名额，将上传内容分块落盘后提交后台执行器。
async def submit_media(kind, file, conf, engine, profile, output):
    if pool is None:
        raise HTTPException(503, '模型目录尚未初始化')
    if not np.isfinite(conf) or not 0 <= conf <= 1:
        raise HTTPException(400, '置信度必须在 0 到 1 之间')
    if profile not in ('speed', 'balanced', 'detail') or output not in ('original', 'nearest'):
        raise HTTPException(400, '无效的处理选项')
    if engine != 'auto' and engine not in pool.engines:
        raise HTTPException(400, '引擎不存在，请刷新页面')
    job_id = uuid.uuid4().hex
    with jobs_lock:
        if sum(j['status'] not in ('done', 'error', 'cancelled') for j in jobs.values()) >= 4:
            raise HTTPException(429, '上传处理队列已满，请稍后再试')
        directory = Path(tempfile.mkdtemp(prefix='yolo-media-'))
        source = directory / 'source'
        result = directory / ('result.jpg' if kind == 'image' else 'result.mp4')
        jobs[job_id] = dict(status='uploading', stage='uploading', kind=kind,
                            directory=str(directory), result=str(result), updated=time.time(), readers=0)
    options = dict(conf=conf, engine=engine, profile=profile, output=output)
    try:
        # UploadFile 由 Starlette 暂存；分块复制，避免将整个视频读入内存。
        def save_upload():
            with open(source, 'wb') as destination:
                shutil.copyfileobj(file.file, destination, length=1024 * 1024)
        save_task = asyncio.create_task(asyncio.to_thread(save_upload))
        try:
            await asyncio.shield(save_task)
        except asyncio.CancelledError:
            await save_task  # 等待写入完成后再清理，避免删除仍在使用的文件。
            raise
        if not source.stat().st_size:
            raise ValueError('上传文件为空')
        update_job(job_id, status='queued', stage='queued')
        media_executor.submit(run_media_job, job_id, kind, str(source), str(result), options)
        return dict(job_id=job_id)
    except BaseException:
        with jobs_lock:
            jobs.pop(job_id, None)
        shutil.rmtree(directory, ignore_errors=True)
        raise
    finally:
        await file.close()


@app.get('/')
async def index():
    return HTMLResponse(HTML)


@app.get('/api/engines')
async def engines():
    return dict(engines=pool.catalog(), startup_warmup=pool.warmup_report)


@app.post('/api/image', status_code=202)
async def image_upload(file: UploadFile = File(...), conf: float = Form(.4),
                       engine: str = Form('auto'), profile: str = Form('detail')):
    return await submit_media('image', file, conf, engine, profile, 'original')


@app.post('/api/video', status_code=202)
async def video_upload(file: UploadFile = File(...), conf: float = Form(.4),
                       engine: str = Form('auto'), profile: str = Form('detail'),
                       output: str = Form('original')):
    return await submit_media('video', file, conf, engine, profile, output)


@app.get('/api/jobs/{job_id}')
async def job_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, '任务不存在或已过期')
        return {k: v for k, v in job.items() if k not in ('directory', 'result', 'readers')}


@app.post('/api/jobs/{job_id}/stop')
async def stop_media_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, '任务不存在或已过期')
        if job['status'] not in ('done', 'error', 'cancelled'):
            job['cancel_requested'] = True
            job['updated'] = time.time()
        return dict(status=job['status'], stop_requested=job.get('cancel_requested', False))


def release_result(job_id):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]['readers'] -= 1
            jobs[job_id]['updated'] = time.time()


# 增加结果读取计数，响应结束后由后台回调释放，防止下载期间被过期清理。
@app.get('/api/jobs/{job_id}/result')
async def job_result(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, '任务不存在或已过期')
        if job['status'] != 'done':
            raise HTTPException(409, '任务尚未完成')
        job['readers'] += 1
        job['updated'] = time.time()
        result, kind = job['result'], job['kind']
    return FileResponse(result, media_type='image/jpeg' if kind == 'image' else 'video/mp4',
                        background=BackgroundTask(release_result, job_id))


def process_live_frame(raw, config):
    return live_pipeline.process(raw, config)


# 接收与推理解耦；每帧携带独立配置快照，连接结束时取消任务并释放排队帧。
@app.websocket('/ws')
async def realtime(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json(dict(type='init'))
    # 服务端按入队顺序处理已接收帧；浏览器采集与显示端仍可能跳过画面。
    # 当推理速度跟不上输入时，
    # 有界队列通过等待入队向接收端传递背压。
    pending_frames = asyncio.Queue(maxsize=4)

    async def receive_frames():
        config = None
        while True:
            message = await websocket.receive()
            if message['type'] == 'websocket.disconnect':
                return
            if message.get('text') is not None:
                value = json.loads(message['text'])
                if not isinstance(value, dict) or value.get('type') != 'frame':
                    raise ValueError('无效的帧配置')
                if not isinstance(value.get('id'), int) or not isinstance(value.get('revision'), int):
                    raise ValueError('帧编号或画幅版本无效，请刷新网页')
                config = value
            elif message.get('bytes') is not None:
                current, config = config, None
                if current is None:
                    raise ValueError('缺少帧配置')
                raw = message['bytes']
                if len(raw) > 20 * 1024 * 1024:
                    raise ValueError('实时 JPEG 过大，请降低采集分辨率')
                await pending_frames.put((raw, current, time.perf_counter()))

    async def infer_fifo():
        while True:
            raw, config, arrived = await pending_frames.get()
            started = time.perf_counter()
            try:
                result = await asyncio.to_thread(process_live_frame, raw, config)
            except Exception as exc:
                result = dict(type='result', id=config['id'], error=str(exc))
            completed = time.perf_counter()
            result.update(revision=config['revision'], captured_at=config.get('captured_at'),
                          sent_at=config.get('sent_at'), draw_ms=config.get('draw_ms'),
                          encode_ms=config.get('encode_ms'), jpeg_quality=config.get('jpeg_quality'),
                          buffered_bytes=config.get('buffered_bytes'),
                          jpeg_bytes=len(raw), pending_frames=pending_frames.qsize(),
                          queue_capacity=pending_frames.maxsize,
                          server_process_ms=round((completed - started) * 1000, 2),
                          server_total_ms=round((completed - arrived) * 1000, 2),
                          queue_ms=round((started - arrived) * 1000, 2))
            # 仅由此任务发送推理结果，保证 WebSocket 写入顺序。
            jpeg = result.pop('_jpeg', None)
            if jpeg is None:
                await websocket.send_json(result)
            else:
                # 使用一条完整消息：4 字节大端元数据长度、UTF-8 元数据，
                # 最后是 JPEG 数据，确保图像与本帧检测结果对应。
                header = json.dumps(result, ensure_ascii=False).encode('utf-8')
                await websocket.send_bytes(len(header).to_bytes(4, 'big') + header + jpeg)

    tasks = [asyncio.create_task(receive_frames()), asyncio.create_task(infer_fifo())]
    try:
        finished, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in finished:
            task.result()
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception as exc:
        print(f'[realtime] 会话结束: {exc}', flush=True)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        while not pending_frames.empty():
            pending_frames.get_nowait()
        try:
            await websocket.close()
        except (RuntimeError, WebSocketDisconnect):
            pass


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', '7860')),
                ws_max_size=24 * 1024 * 1024, ws_max_queue=2, log_level='info')

