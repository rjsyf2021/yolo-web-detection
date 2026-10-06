#!/usr/bin/env python3
"""YOLO26 640×640 推理基准：对比不同型号与独立的源画面尺寸。

网络输入固定为 640×640；与应用一致使用 Ultralytics 默认灰边 LetterBox，
坐标还原后可在源分辨率画框。默认比较 4:3 / 16:9 的 720p、1080p 画幅。
统计预测和可选画框耗时，不包含摄像头、视频编解码或网络传输。
"""
import argparse
import gc
import json
import math
import re
import statistics
import time
from pathlib import Path


BASE = Path(__file__).resolve().parent
PATTERN = re.compile(r'yolo26([nsmxl])_(\d+)x(\d+)\.engine')
ALIGN = 32
DEFAULT_IMAGE = BASE / 'bus.jpg'
SOURCE_SIZES = ('1280x720', '1920x1080', '960x720', '1440x1080')


# 按 ALIGN 对齐文件名中的标称尺寸；实际绑定形状仍由 TensorRT 加载时验证。
def aligned(value):
    return (value + ALIGN - 1) // ALIGN * ALIGN


# 只接受 YOLO26 n/s/m/l/x 的 640×640 引擎，与应用的发现规则一致。
def discover(directory, variants=None):
    found = []
    for path in sorted(Path(directory).glob('*.engine')):
        match = PATTERN.fullmatch(path.name)
        if not match:
            continue
        variant, width, height = match.groups()
        width, height = int(width), int(height)
        if (width, height) != (640, 640):
            continue
        if variants and variant not in variants:
            continue
        found.append(dict(key=path.stem, path=path, variant=variant,
                          width=width, height=height,
                          input_w=aligned(width), input_h=aligned(height),
                          pixels=width * height))
    # 排序：先型号规模，再分辨率，便于同族对照。
    found.sort(key=lambda item: ('nsmlx'.index(item['variant']), item['key']))
    return found


def parse_source_size(value):
    if value == 'original':
        return None
    try:
        width, height = map(int, value.lower().split('x'))
        if min(width, height) < 1 or max(width, height) > 8192:
            raise ValueError()
        return width, height
    except ValueError:
        raise argparse.ArgumentTypeError('源尺寸须为 宽x高（1–8192）或 original') from None


def load_source(path):
    import cv2
    import numpy as np
    if path.is_file():
        source = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if source is None:
            raise ValueError(f'无法解码测试图片：{path}')
        return source, 'local'
    if path != DEFAULT_IMAGE:
        raise FileNotFoundError(f'测试图片不存在：{path}')
    print('[提示] 未提供测试图片，使用合成画面；结果不代表真实素材的检测与画框开销', flush=True)
    source = np.zeros((1080, 1920, 3), dtype=np.uint8)
    source[:, :, 1] = np.linspace(0, 255, 1920, dtype=np.uint8)
    return source, 'synthetic'


def source_frame(source, size):
    """构造指定源画幅，耗时不计入测速；之后模型仍会进行 640×640 预处理。

    此处的黑边属于测试素材布局，与模型输入的灰边 LetterBox 是两个阶段。
    改变源画幅用于对照预处理/画框成本，不能用来评估模型准确率。
    """
    import cv2
    import numpy as np
    if size is None:
        return source
    width, height = size
    scale = min(width / source.shape[1], height / source.shape[0])
    w, h = max(1, round(source.shape[1] * scale)), max(1, round(source.shape[0] * scale))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    top, left = (height - h) // 2, (width - w) // 2
    canvas[top:top+h, left:left+w] = cv2.resize(source, (w, h))
    return canvas


# 预热后自适应测量：至少达到最少轮数与最短时长，用上限轮数兜底。
def measure(model, image, imgsz, conf, device, warmup, rounds, seconds, cap, render):
    for _ in range(warmup):
        result = model.predict(image, imgsz=imgsz, rect=False, conf=conf, device=device,
                               verbose=False, stream=False)[0].cpu()
        if render == 'server':
            result.plot()
    times, speeds = [], []
    started = time.perf_counter()
    while True:
        begin = time.perf_counter()
        result = model.predict(image, imgsz=imgsz, rect=False, conf=conf,
                               device=device, verbose=False, stream=False)[0]
        result = result.cpu()
        draw_started = time.perf_counter()
        if render == 'server':
            drawn = result.plot()
            if drawn.shape != image.shape:
                raise RuntimeError('画框结果未保持源分辨率')
        draw_ms = (time.perf_counter() - draw_started) * 1000 if render == 'server' else 0.
        times.append((time.perf_counter() - begin) * 1000)
        speeds.append(dict(result.speed, drawing=draw_ms))
        if len(times) >= cap:
            break
        if len(times) >= rounds and time.perf_counter() - started >= seconds:
            break
    return times, speeds


# 汇总预测/可选画框耗时；FPS 是耗时倒数，不代表摄像头到浏览器的端到端帧率。
def summarize(times, speeds):
    ordered = sorted(times)
    mean = statistics.fmean(times)
    median = statistics.median(times)
    index = min(len(ordered) - 1, max(0, math.ceil(.95 * len(ordered)) - 1))

    def stage(name):
        values = [float(item.get(name, 0.)) for item in speeds]
        return round(statistics.fmean(values), 3) if values else 0.

    return dict(
        rounds=len(times),
        mean_ms=round(mean, 3),
        median_ms=round(median, 3),
        p95_ms=round(ordered[index], 3),
        fps_mean=round(1000 / mean, 2),
        fps_median=round(1000 / median, 2),
        preprocess_ms=stage('preprocess'),
        inference_ms=stage('inference'),
        postprocess_ms=stage('postprocess'),
        drawing_ms=stage('drawing'),
    )


# 打印 FPS 排行表；分辨率与输入尺寸同时给出，便于判断是否发生补边。
def print_table(results):
    width = 104
    print(f"\n{'=' * width}")
    print('预测/画框 FPS 排行（不含解码、编码与传输）')
    print(f"{'=' * width}")
    print(f"{'引擎':<22}{'型号':<4}{'源画面':<10}{'输入':<10}"
          f"{'平均ms':>8}{'预处理ms':>9}{'FPS':>8}{'中位FPS':>8}"
          f"{'推理ms':>8}{'后处理ms':>9}{'显存MB':>8}")
    print('-' * width)
    for item in sorted(results, key=lambda row: row['fps_mean'], reverse=True):
        source = f"{item['source_w']}x{item['source_h']}"
        net = f"{item['input_w']}x{item['input_h']}"
        print(f"{item['key']:<22}{item['variant']:<4}{source:<10}{net:<10}"
              f"{item['mean_ms']:>8.2f}{item['preprocess_ms']:>9.2f}"
              f"{item['fps_mean']:>8.1f}{item['fps_median']:>8.1f}"
              f"{item['inference_ms']:>8.2f}{item['postprocess_ms']:>9.2f}"
              f"{item['vram_peak_mb']:>8.1f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir', type=Path, default=BASE,
                        help='引擎目录（默认脚本所在目录）')
    parser.add_argument('--variants', nargs='+', choices=('n', 's', 'm', 'l', 'x'),
                        help='只测指定型号；省略时测目录内全部型号')
    parser.add_argument('--image', type=Path, default=DEFAULT_IMAGE,
                        help='自备测试图片；默认 bus.jpg 不存在时用合成图，不联网下载')
    parser.add_argument('--source-sizes', nargs='+', type=parse_source_size,
                        default=[parse_source_size(v) for v in SOURCE_SIZES],
                        help='源画面尺寸，默认四种采集画幅；original 表示测试图原始尺寸')
    parser.add_argument('--render', choices=('none', 'server'), default='server',
                        help='server 含原分辨率画框，none 仅检测和坐标后处理')
    parser.add_argument('--conf', type=float, default=.4)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=10, help='每个引擎的预热次数')
    parser.add_argument('--rounds', type=int, default=50, help='最少测量轮数')
    parser.add_argument('--seconds', type=float, default=3.0, help='每个引擎的最短测量秒数')
    parser.add_argument('--max-rounds', type=int, default=300, help='测量轮数上限')
    parser.add_argument('--json', type=Path, default=BASE / 'fps-benchmark.json')
    parser.add_argument('--list', action='store_true', help='只列出发现的引擎，不加载 GPU')
    args = parser.parse_args()

    if args.warmup < 0 or args.rounds < 1 or args.max_rounds < args.rounds:
        parser.error('预热需非负，最少轮数至少为 1 且不超过上限轮数')
    if not math.isfinite(args.seconds) or args.seconds < 0:
        parser.error('最短测量时长必须为非负有限值')
    if not math.isfinite(args.conf) or not 0 <= args.conf <= 1:
        parser.error('置信度必须在 0 到 1 之间')

    targets = discover(args.dir, set(args.variants) if args.variants else None)
    if not targets:
        raise SystemExit(f'在 {args.dir} 未找到 yolo26<型号>_640x640.engine')
    print(f'发现 {len(targets)} 个引擎：'
          + ', '.join(item['key'] for item in targets), flush=True)
    if args.list:
        for item in targets:
            print(f"  {item['key']:<24}{item['variant']}  "
                  f"{item['width']}x{item['height']}  "
                  f"输入 {item['input_w']}x{item['input_h']}  "
                  '源画面独立设置')
        return

    import torch
    from ultralytics import YOLO
    if not torch.cuda.is_available():
        raise SystemExit('CUDA 不可用，请在 GPU 服务器的原 YOLO 环境中运行')

    source, source_kind = load_source(args.image)
    report = dict(render=args.render, source_kind=source_kind,
                  scope='预测与坐标还原，server 模式含原图画框；不含编解码与传输',
                  memory_scope='PyTorch 分配器峰值，不含全部 TensorRT/CUDA 显存',
                  gpu=torch.cuda.get_device_name(args.device), image=str(args.image),
                  conf=args.conf, warmup=args.warmup, min_rounds=args.rounds,
                  min_seconds=args.seconds, max_rounds=args.max_rounds,
                  results=[])
    print(f"\n设备：{report['gpu']} · 预热 {args.warmup} · "
          f"最少 {args.rounds} 轮 / {args.seconds}s · 上限 {args.max_rounds} 轮\n", flush=True)

    gpu_error = None
    for index, item in enumerate(targets, 1):
        model = None
        for size in args.source_sizes:
            width, height = size if size is not None else (source.shape[1], source.shape[0])
            identity = dict(key=item['key'], variant=item['variant'], source_w=width,
                            source_h=height, input_w=640, input_h=640, render=args.render)
            if gpu_error is not None:
                report['results'].append(dict(identity, status='skipped',
                                              error='GPU 清理/同步失败，跳过后续测试：' + gpu_error))
                print(f"    [跳过] {item['key']} · 源 {width}×{height}: {gpu_error}", flush=True)
                continue
            image = None
            failed_row = None
            try:
                if model is None:
                    model = YOLO(str(item['path']), task='detect')
                image = source_frame(source, size)
                print(f"[{index}/{len(targets)}] {item['key']} · 源 {width}×{height} · 推理 640×640", flush=True)
                torch.cuda.reset_peak_memory_stats(args.device)
                times, speeds = measure(model, image, (640, 640), args.conf, args.device,
                                        args.warmup, args.rounds, args.seconds, args.max_rounds, args.render)
                stats = summarize(times, speeds)
                stats.update(identity, status='ok',
                             vram_peak_mb=round(torch.cuda.max_memory_allocated(args.device) / 1048576, 1))
                report['results'].append(stats)
                drawing = f" · 画框 {stats['drawing_ms']:.2f} ms" if args.render == 'server' else ''
                print(f"    {stats['rounds']} 轮 · {stats['mean_ms']:.2f} ms · {stats['fps_mean']:.1f} FPS · "
                      f"预处理 {stats['preprocess_ms']:.2f} / 推理 {stats['inference_ms']:.2f} / "
                      f"后处理 {stats['postprocess_ms']:.2f} ms{drawing}", flush=True)
            except Exception as exc:
                failed_row = dict(identity, status='error', error=str(exc))
                report['results'].append(failed_row)
                print(f"    [失败] {item['key']} · 源 {width}×{height}: {exc}", flush=True)
            # 离开 except 后释放回溯和失败预测器；下一尺寸重建模型，避免复用半初始化状态。
            image = None
            if failed_row is not None:
                model = None
                gc.collect()
                try:
                    # 检查异步 CUDA 错误；若上下文不可继续使用，不再尝试其他尺寸或型号。
                    torch.cuda.synchronize(args.device)
                    torch.cuda.empty_cache()
                except Exception as exc:
                    gpu_error = str(exc)
                    failed_row['gpu_error'] = gpu_error
        model = None
        gc.collect()
        if gpu_error is None:
            try:
                torch.cuda.empty_cache()
            except Exception as exc:
                gpu_error = str(exc)
                report.setdefault('cleanup_errors', []).append(dict(key=item['key'], error=gpu_error))
                print(f"    [清理失败] {item['key']}: {gpu_error}", flush=True)

    ok = [item for item in report['results'] if 'error' not in item]
    if ok:
        print_table(ok)
    failed = [item for item in report['results'] if 'error' in item]
    if failed:
        print('\n失败/跳过档位：' + '、'.join(
            f"{item['key']} {item['source_w']}×{item['source_h']} ({item['status']})" for item in failed))
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n已保存：{args.json.resolve()}")
    print('说明：FPS 包含预处理与后处理；server 模式还含原分辨率画框。'
          '不代表视频流端到端 FPS；显存统计仅覆盖 PyTorch 分配器。')
    if failed or report.get('cleanup_errors'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
