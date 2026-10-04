"""批量导出 YOLO26 n/s/m/l/x 检测模型的 16:9 和 4:3 半精度 TensorRT 引擎。

使用 Ultralytics FP16 导出流程；当前 TensorRT 11 环境通过 ModelOpt
在 CUDA GPU 0 上执行参考推理并选择混合精度。档位不超过 1080p，
参考推理一次性采集中间输出，较大模型需要更多显存与系统内存。
ONNX 导出、参考推理、引擎构建分别使用独立子进程，单档失败后继续其余档位。
"""
import argparse
import ast
from contextlib import contextmanager
import math
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from unittest.mock import patch


BASE = Path(__file__).resolve().parent
MODEL_VARIANTS = ("n", "s", "m", "l", "x")
# 文件名保留标称尺寸，引擎实际输入需要向上对齐到 ALIGN 的倍数。
ALIGN = 32
# 顺序为（高，宽）。标称高度最高 1080；更高分辨率不再预生成。
widescreen = [
    (720, 1280),
    (1080, 1920),
    (480, 854),     # 16:9 480p（手机端常用尺寸）
    (960, 1706),    # 16:9 960p（按高度 960 换算，宽取最近偶数）
]
standard_43 = [
    (480, 640),
    (720, 960),
    (960, 1280),
    (1080, 1440),
]
all_sizes = widescreen + standard_43


def input_size(width, height):
    """按 ALIGN 的倍数向上对齐，返回引擎实际输入 (宽, 高)。"""
    return ((width + ALIGN - 1) // ALIGN * ALIGN,
            (height + ALIGN - 1) // ALIGN * ALIGN)


def check_environment():
    """执行小型 CUDA 推理，提前发现动态库缺失或后端回退，避免逐档重复失败。"""
    import numpy as np
    import torch
    import onnx
    import onnxruntime as ort
    import tensorrt as trt
    from modelopt.onnx import autocast

    print(f"[环境] Python {sys.version.split()[0]} · PyTorch {torch.__version__} · "
          f"TensorRT {trt.__version__} · ONNX Runtime {ort.__version__}", flush=True)
    torch_cuda = torch.version.cuda or ""
    ort_cuda = getattr(ort, "cuda_version", "") or ""
    print(f"[环境] PyTorch CUDA {torch_cuda or '未知'} · ORT CUDA {ort_cuda or '未知'}", flush=True)
    if torch_cuda.split('.')[0] != '13' or ort_cuda.split('.')[0] != '13':
        raise RuntimeError("本脚本要求 PyTorch 与 ONNX Runtime 均使用 CUDA 13 构建；"
                           "请检查当前虚拟环境，不能仅依据系统 nvcc 版本判断")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU 不可用，请检查驱动、设备访问权限及当前虚拟环境")
    if int(trt.__version__.split('.')[0]) < 11:
        raise RuntimeError("当前 GPU 参考推理流程要求 TensorRT 11 或更高版本")
    if not callable(getattr(autocast, 'convert_to_mixed_precision', None)):
        raise RuntimeError("ModelOpt 缺少混合精度转换接口，请检查 nvidia-modelopt 版本")
    require_cuda_provider(ort)
    ort.preload_dlls()
    header = Path(sysconfig.get_path('include')) / 'Python.h'
    if not header.is_file():
        print(f"[提示] 缺少 {header}，Triton 扩展编译可能失败；需安装匹配当前 Python 的开发头文件。", flush=True)
    # 使用有计算量的算子而非 Identity，验证 CUDA 会话实际能执行。
    helper = onnx.helper
    graph = helper.make_graph(
        [helper.make_node('MatMul', ['x', 'w'], ['y'])], 'cuda-preflight',
        [helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [1, 32])],
        [helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [1, 32])],
        [onnx.numpy_helper.from_array(np.eye(32, dtype=np.float32), name='w')],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)], ir_version=10)
    session = ort.InferenceSession(model.SerializeToString(),
                                   providers=[('CUDAExecutionProvider', {'device_id': 0})])
    check_cuda_session(session)
    x = np.arange(32, dtype=np.float32).reshape(1, 32)
    np.testing.assert_allclose(session.run(None, {'x': x})[0], x, rtol=1e-5, atol=1e-5)
    del session
    free, total = torch.cuda.mem_get_info(0)
    print(f"[环境通过] {torch.cuda.get_device_name(0)} · CUDA 推理正常 · "
          f"空闲显存 {free / 1024**3:.1f}/{total / 1024**3:.1f} GiB。"
          "参考推理在 GPU 上进行，中间输出仍占用系统内存；模型越大，资源需求越高。", flush=True)


def require_cuda_provider(ort):
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNX Runtime 未提供 CUDA 后端，请检查 CUDA 13 版 onnxruntime-gpu")


def check_cuda_session(session):
    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError("ONNX Runtime CUDA 初始化失败，已禁止纯 CPU 回退。"
                           "请查看上方缺失的 .so 动态库或驱动错误；先运行 --check-env 排查")
    session.disable_fallback()


@contextmanager
def gpu_reference_inference():
    """仅在当前导出子进程中指定参考推理后端，不修改已安装的依赖文件。"""
    import onnxruntime as ort
    from modelopt.onnx import autocast

    require_cuda_provider(ort)
    original_convert = autocast.convert_to_mixed_precision
    original_session = ort.InferenceSession

    def cuda_session(*args, **kwargs):
        session = original_session(*args, **kwargs)
        # 声明支持 CUDA 不代表初始化成功：缺少 CUDA/cuDNN 库时 ORT 可能退回 CPU。
        # 禁止运行异常后重新创建纯 CPU 会话；形状等辅助算子仍可由 CPU 执行。
        check_cuda_session(session)
        print("[参考推理] 已确认 CUDAExecutionProvider，使用 GPU 0", flush=True)
        return session

    def convert_on_cuda(*args, **kwargs):
        # Ultralytics 暂未透传此参数，覆盖 ModelOpt 默认的 providers=["cpu"]。
        # 沿用完整中间输出采集；模型较大时仍需足够的显存与系统内存。
        kwargs["providers"] = ["cuda:0"]
        with patch.object(ort, "InferenceSession", cuda_session):
            return original_convert(*args, **kwargs)

    # 包装范围仅包含当前导出；成功或异常退出都会恢复原函数。
    with patch.object(autocast, "convert_to_mixed_precision", convert_on_cuda):
        yield


def run_stage(stage, directory, height, width, workspace=None, variant="s"):
    """各阶段独立进程：前一阶段退出后，才允许后一阶段分配 GPU 内存。"""
    command = [sys.executable, str(Path(__file__).resolve()), "--stage", stage,
               "--worker", str(height), str(width), "--worker-dir", str(directory),
               "--models", variant]
    if workspace is not None:
        command += ["--workspace", str(workspace)]
    completed = subprocess.run(command)
    if completed.returncode:
        raise RuntimeError(f"{stage} 阶段失败（{width}×{height}），退出码 {completed.returncode}；"
                           "目标引擎未替换")


def export_stage(stage, directory, height, width, workspace, variant="s"):
    directory = Path(directory)
    input_w, input_h = input_size(width, height)
    model_name = f"yolo26{variant}"
    onnx_file = directory / f"{model_name}.onnx"
    if stage == "onnx":
        from ultralytics import YOLO
        source = directory / f"{model_name}.pt"
        shutil.copy2(BASE / source.name, source)
        model = YOLO(str(source))
        if model.task != "detect":
            raise ValueError(f"{source.name} 不是检测模型，本应用仅支持 detect 任务")
        exported = model.export(format="onnx", imgsz=(input_h, input_w), quantize=32,
                                device=0, batch=1, dynamic=False)
        if not exported or not Path(exported).is_file():
            raise RuntimeError("未生成 FP32 ONNX")
        if Path(exported).resolve() != onnx_file.resolve():
            Path(exported).replace(onnx_file)
    elif stage == "reference":
        import onnxruntime as ort
        from ultralytics.utils.export.engine import modelopt_quantize_onnx
        ort.preload_dlls()
        # 沿用官方参考图片与混合精度分类，只改变参考推理设备，不降低校准分辨率。
        with gpu_reference_inference():
            modelopt_quantize_onnx(str(onnx_file), quantize=16,
                                  shape=(1, 3, input_h, input_w), dynamic=False, prefix="[参考推理] ")
    elif stage == "engine":
        import onnx
        from ultralytics.utils.export.engine import onnx2engine
        metadata = {item.key: item.value for item in
                    onnx.load(str(onnx_file), load_external_data=False).metadata_props}
        for key in ("imgsz", "names", "kpt_shape", "kpt_names", "args", "end2end"):
            if key in metadata:
                try:
                    metadata[key] = ast.literal_eval(metadata[key])
                except (ValueError, SyntaxError):
                    pass
        for key in ("stride", "batch", "channels"):
            if key in metadata:
                metadata[key] = int(metadata[key])
        # 混合精度已写入 ONNX；quantize=None 避免再次触发参考推理。
        print("[构建] 使用已完成参考分析的 FP16/FP32 混合精度图", flush=True)
        onnx2engine(str(directory / f"{model_name}.fp16.onnx"),
                    output_file=directory / f"{model_name}.engine",
                    quantize=None, workspace=workspace, dynamic=False,
                    shape=(1, 3, input_h, input_w), metadata=metadata or None, prefix="[TensorRT] ")


def export_one(height, width, overwrite=False, workspace=None, temp_parent=None, variant="s"):
    """三个串行子进程分别导出、参考分析、构建；全部成功才替换目标。"""
    target = BASE / f"yolo26{variant}_{width}x{height}.engine"
    if target.exists() and not overwrite:
        print(f"[跳过] 已存在：{target.name}；需要重新导出时使用 --overwrite", flush=True)
        return target
    weights = BASE / f"yolo26{variant}.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"缺少模型权重：{weights}；请自行准备，脚本不会自动下载")
    with tempfile.TemporaryDirectory(prefix=".yolo-export-", dir=temp_parent or BASE) as directory:
        print(f"[导出] yolo26{variant} · {width}×{height}", flush=True)
        for stage in ("onnx", "reference", "engine"):
            run_stage(stage, directory, height, width, workspace, variant)
        engine = Path(directory) / f"yolo26{variant}.engine"
        if not engine.is_file() or engine.stat().st_size == 0:
            raise RuntimeError("未生成有效引擎")
        engine.replace(target)
    print(f"[完成] {target.name}（{target.stat().st_size / 1024**2:.1f} MiB）", flush=True)
    return target


def run_worker(height, width, overwrite=False, workspace=None, variant="s"):
    """在独立子进程中导出单个档位，隔离内存并允许单档失败后继续其余档位。"""
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(height), str(width),
               "--models", variant]
    if overwrite:
        command.append("--overwrite")
    if workspace is not None:
        command += ["--workspace", str(workspace)]
    # 父进程只拥有并清理本次创建的目录；子进程被杀死后也能清理，不扫描其他任务的目录。
    with tempfile.TemporaryDirectory(prefix=".yolo-export-worker-", dir=BASE) as directory:
        command += ["--worker-dir", directory]
        return subprocess.run(command).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODEL_VARIANTS,
                        help="指定 YOLO26 型号（如 m l x）；省略时发现脚本目录中已有的对应 .pt 权重")
    parser.add_argument("--overwrite", action="store_true", help="重新导出并覆盖已有目标引擎")
    parser.add_argument("--list", action="store_true", help="只列出导出档位，不加载模型或使用 GPU")
    parser.add_argument("--check-env", action="store_true", help="检查 CUDA 13 环境并运行小型 GPU 推理，不导出引擎")
    parser.add_argument("--workspace", type=float, default=None,
                        help="TensorRT 工作区上限（GiB），默认由 TensorRT 自动决定")
    parser.add_argument("--worker", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"),
                        help=argparse.SUPPRESS)  # 供内部子进程调用，不在帮助中显示
    parser.add_argument("--worker-dir", help=argparse.SUPPRESS)
    parser.add_argument("--stage", choices=("onnx", "reference", "engine"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stage and (not args.worker or not args.worker_dir):
        parser.error("内部阶段需要 --worker 和 --worker-dir")
    if args.workspace is not None and (not math.isfinite(args.workspace) or args.workspace <= 0):
        parser.error("--workspace 必须是大于 0 的有限数值，单位为 GiB")
    if args.worker:
        if not args.models or len(args.models) != 1:
            parser.error("内部子进程需要用 --models 指定一个型号")
        variant = args.models[0]
        height, width = args.worker
        if min(height, width) <= 0:
            parser.error("导出宽高必须大于 0")
        if args.stage:
            export_stage(args.stage, args.worker_dir, height, width, args.workspace, variant)
        else:
            export_one(height, width, overwrite=args.overwrite, workspace=args.workspace,
                       temp_parent=args.worker_dir, variant=variant)
        return
    variants = list(dict.fromkeys(args.models)) if args.models else [
        variant for variant in MODEL_VARIANTS if (BASE / f"yolo26{variant}.pt").is_file()
    ]
    if args.list:
        # 无本地权重时也能查看全部支持档位，不加载 GPU 依赖或下载文件。
        for variant in variants or MODEL_VARIANTS:
            weights_state = "权重就绪" if (BASE / f"yolo26{variant}.pt").is_file() else "需自备权重"
            for h, w in all_sizes:
                input_w, input_h = input_size(w, h)
                name = f"yolo26{variant}_{w}x{h}.engine"
                state = "已存在" if (BASE / name).exists() else "待导出"
                print(f"{name}：标称 {w}×{h}，输入 {input_w}×{input_h} · {state} · {weights_state}")
        return
    if not args.check_env:
        if not variants:
            parser.error("未找到本地检测权重；请将自备的 yolo26n/s/m/l/x.pt 放在脚本目录，"
                         "也可用 --models 指定要导出的型号。脚本不会自动下载")
        # 先检查所有需要构建的型号，避免跑完部分档位后才发现缺少权重。
        missing = [f"yolo26{variant}.pt" for variant in variants
                   if not (BASE / f"yolo26{variant}.pt").is_file()
                   and any(args.overwrite or not (BASE / f"yolo26{variant}_{w}x{h}.engine").exists()
                           for h, w in all_sizes)]
        if missing:
            parser.error("缺少自备权重：" + "、".join(missing))

    try:
        if args.check_env:
            check_environment()
        else:
            # 预检查也在独立进程，避免父进程在整个导出期间占用 CUDA 上下文。
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--check-env"], check=True)
    except Exception as exc:
        raise SystemExit(f"[环境检查失败] {exc}") from exc
    if args.check_env:
        return
    failed = []
    skipped = 0
    for variant in variants:
        for h, w in all_sizes:
            name = f"yolo26{variant}_{w}x{h}.engine"
            if (BASE / name).exists() and not args.overwrite:
                print(f"[跳过] 已存在：{name}；需要重新导出时使用 --overwrite", flush=True)
                skipped += 1
                continue
            code = run_worker(h, w, args.overwrite, args.workspace, variant)
            if code == 0:
                continue
            hint = "（可能被 OOM 或外部信号终止，请检查系统内存和显存）" if code in (137, -9) else ""
            failed.append(f"yolo26{variant} {w}×{h}")
            print(f"[失败] {name}：子进程退出码 {code}{hint}", flush=True)
    if failed:
        raise SystemExit("以下档位导出失败：" + "、".join(failed))
    print(f"完成：{len(variants)} 种型号，共 {len(variants) * len(all_sizes)} 档，跳过已存在 {skipped} 档。"
          "重启 app_ws-multi.py 后可在图片 / 视频页面选择。")


if __name__ == "__main__":
    main()
