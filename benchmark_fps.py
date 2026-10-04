#!/usr/bin/env python3
"""自适应 FPS 基准：自动发现本地 YOLO26 引擎，按型号与分辨率测量单帧吞吐。

发现规则与应用一致：目录下的 yolo26<型号>_<宽>x<高>.engine（型号 n/s/m/l/x）。
每个引擎按其标称分辨率准备测试图，再以与应用相同的方式调用 predict，
统计端到端 FPS、纯推理耗时、各阶段耗时与峰值显存。

"自适应"体现在两处：档位由本地实际存在的引擎决定（型号与分辨率都不写死），
测量轮数按最短测量时长自动调整，快引擎多测、慢引擎少测。
"""
import argparse
import gc
import json
import math
import re
import statistics
import time
import urllib.request
from pathlib import Path


BASE = Path(__file__).resolve().parent
PATTERN = re.compile(r'yolo26([nsmxl])_(\d+)x(\d+)\.engine')
ALIGN = 32
DEFAULT_IMAGE = BASE / 'bus.jpg'
IMAGE_URL = 'https://ultralytics.com/images/bus.jpg'


# 按 ALIGN 向上对齐，得到引擎实际网络输入尺寸。
def aligned(value):
    return (value + ALIGN - 1) // ALIGN * ALIGN


# 扫描目录中的引擎文件；型号与分辨率都从文件名读出，不写死任何档位。
def discover(directory, variants=None):
    found = []
    for path in sorted(Path(directory).glob('*.engine')):
        match = PATTERN.fullmatch(path.name)
        if not match:
            continue
        variant, width, height = match.groups()
        width, height = int(width), int(height)
        if min(width, height) <= 0:
            continue
        if variants and variant not in variants:
            continue
        found.append(dict(key=path.stem, path=path, variant=variant,
                          width=width, height=height,
                          input_w=aligned(width), input_h=aligned(height),
                          pixels=width * height))
    # 排序：先型号规模，再分辨率，便于同族对照。
    found.sort(key=lambda item: (item['variant'], item['pixels'], item['key']))
    return found


# 读取测试图并等比例容纳到引擎标称尺寸；本地缺失则下载，再失败则用合成图。
def load_image(path, width, height):
    import cv2
    import numpy as np
    source = cv2.imread(str(path)) if path.is_file() else None
    if source is None:
        try:
            with urllib.request.urlopen(IMAGE_URL, timeout=30) as response:
                source = cv2.imdecode(np.frombuffer(response.read(), np.uint8),
                                      cv2.IMREAD_COLOR)
            if source is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), source)
                print(f'[提示] 已缓存测试图：{path}', flush=True)
        except Exception as exc:
            print(f'[提示] 无法获取测试图（{exc}），改用合成图测速', flush=True)
    if source is None:
        # 合成图只用于计时，不代表真实场景的检测数量与后处理开销。
        source = np.zeros((720, 1280, 3), dtype=np.uint8)
        source[:, :, 1] = np.linspace(0, 255, 1280, dtype=np.uint8)
    scale = min(width / source.shape[1], height / source.shape[0])
    size = (max(1, round(source.shape[1] * scale)), max(1, round(source.shape[0] * scale)))
    resized = cv2.resize(source, size, interpolation=cv2.INTER_AREA)
    # 补边到标称分辨率，复现应用里等比例容纳、比例不符时留黑边的布局。
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    top, left = (height - size[1]) // 2, (width - size[0]) // 2
    canvas[top:top + size[1], left:left + size[0]] = resized
    return np.ascontiguousarray(canvas), (resized.shape[1], resized.shape[0])


# 预热后自适应测量：至少达到最少轮数与最短时长，用上限轮数兜底。
def measure(model, image, imgsz, conf, device, warmup, rounds, seconds, cap):
    for _ in range(warmup):
        model.predict(image, imgsz=imgsz, rect=False, conf=conf, device=device,
                      verbose=False, stream=False)
    times, speeds = [], []
    started = time.perf_counter()
    while True:
        begin = time.perf_counter()
        result = model.predict(image, imgsz=imgsz, rect=False, conf=conf,
                               device=device, verbose=False, stream=False)[0]
        times.append((time.perf_counter() - begin) * 1000)
        # 读取阶段耗时放在计时区间之外，避免污染整体耗时。
        speeds.append(result.speed)
        if len(times) >= cap:
            break
        if len(times) >= rounds and time.perf_counter() - started >= seconds:
            break
    return times, speeds


# 汇总耗时样本：端到端 FPS 用均值与中位各给一份，阶段耗时取多轮平均。
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
    )


# 打印 FPS 排行表；分辨率与输入尺寸同时给出，便于判断是否发生补边。
def print_table(results):
    print(f"\n{'=' * 96}")
    print('FPS 排行（按端到端 FPS 降序）')
    print(f"{'=' * 96}")
    header = (f"{'引擎':<24}{'型号':<5}{'分辨率':<12}{'输入':<12}"
              f"{'平均ms':>9}{'FPS':>9}{'中位FPS':>9}{'推理ms':>9}{'后处理ms':>10}{'显存MB':>9}")
    print(header)
    print('-' * 96)
    for item in sorted(results, key=lambda row: row['fps_mean'], reverse=True):
        print(f"{item['key']:<24}{item['variant']:<5}"
              f"{item['width']}x{item['height']:<8}"
              f"{item['input_w']}x{item['input_h']:<8}"
              f"{item['mean_ms']:>7.2f}  {item['fps_mean']:>7.1f}  {item['fps_median']:>7.1f}  "
              f"{item['inference_ms']:>7.2f}  {item['postprocess_ms']:>8.2f}  "
              f"{item['vram_peak_mb']:>7.1f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir', type=Path, default=BASE,
                        help='引擎目录（默认脚本所在目录）')
    parser.add_argument('--variants', nargs='+', choices=('n', 's', 'm', 'l', 'x'),
                        help='只测指定型号；省略时测目录内全部型号')
    parser.add_argument('--image', type=Path, default=DEFAULT_IMAGE,
                        help='测试图片；本地缺失时自动下载，失败则用合成图')
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
        raise SystemExit(f'在 {args.dir} 未找到 yolo26<型号>_<宽>x<高>.engine')
    print(f'发现 {len(targets)} 个引擎：'
          + ', '.join(item['key'] for item in targets), flush=True)
    if args.list:
        for item in targets:
            print(f"  {item['key']:<24}{item['variant']}  "
                  f"{item['width']}x{item['height']}  "
                  f"输入 {item['input_w']}x{item['input_h']}  "
                  f"{item['pixels'] / 1e6:.2f} MP")
        return

    import torch
    from ultralytics import YOLO
    if not torch.cuda.is_available():
        raise SystemExit('CUDA 不可用，请在 GPU 服务器的原 YOLO 环境中运行')

    report = dict(gpu=torch.cuda.get_device_name(args.device), image=str(args.image),
                  conf=args.conf, warmup=args.warmup, min_rounds=args.rounds,
                  min_seconds=args.seconds, max_rounds=args.max_rounds,
                  results=[])
    print(f"\n设备：{report['gpu']} · 预热 {args.warmup} · "
          f"最少 {args.rounds} 轮 / {args.seconds}s · 上限 {args.max_rounds} 轮\n", flush=True)

    for index, item in enumerate(targets, 1):
        label = f"{item['key']}（{item['variant']} · {item['width']}x{item['height']}）"
        print(f"[{index}/{len(targets)}] {label}", flush=True)
        try:
            model = YOLO(str(item['path']), task='detect')
            image, fitted = load_image(args.image, item['width'], item['height'])
            if (item['width'], item['height']) != fitted:
                print(f'    测试图 {fitted[0]}x{fitted[1]} 已补边到标称尺寸', flush=True)
            torch.cuda.reset_peak_memory_stats(args.device)
            times, speeds = measure(model, image, (item['input_h'], item['input_w']),
                                    args.conf, args.device, args.warmup,
                                    args.rounds, args.seconds, args.max_rounds)
            stats = summarize(times, speeds)
            stats.update(key=item['key'], variant=item['variant'], width=item['width'],
                         height=item['height'], input_w=item['input_w'],
                         input_h=item['input_h'],
                         vram_peak_mb=round(torch.cuda.max_memory_allocated(args.device) / 1048576, 1))
            report['results'].append(stats)
            print(f"    {stats['rounds']} 轮 · 平均 {stats['mean_ms']:.2f} ms · "
                  f"FPS {stats['fps_mean']:.1f}（中位 {stats['fps_median']:.1f}）· "
                  f"推理 {stats['inference_ms']:.2f} ms · 显存 {stats['vram_peak_mb']:.1f} MB",
                  flush=True)
            del model, image, times, speeds, stats
        except Exception as exc:
            # 单档失败不影响其余档位，失败原因记录进报告。
            report['results'].append(dict(key=item['key'], variant=item['variant'],
                                          width=item['width'], height=item['height'],
                                          error=str(exc)))
            print(f"    [失败] {exc}", flush=True)
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    ok = [item for item in report['results'] if 'error' not in item]
    if ok:
        print_table(ok)
    failed = [item for item in report['results'] if 'error' in item]
    if failed:
        print('\n失败档位：' + '、'.join(item['key'] for item in failed))
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n已保存：{args.json.resolve()}")
    print('说明：端到端 FPS 含预处理与后处理；纯推理 FPS 可用 1000 / 推理ms 估算。'
          '阶段耗时取多轮平均。')


if __name__ == '__main__':
    main()
