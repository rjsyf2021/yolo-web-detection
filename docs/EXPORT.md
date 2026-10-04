# 导出 TensorRT 引擎

模型由使用者自行准备，仓库不附带权重、ONNX 模型或 TensorRT 引擎。导出脚本不自动下载权重；已有兼容 `.engine` 文件时，可直接运行应用，无需执行本页的导出步骤。

export_all_engines.py 支持 YOLO26 n/s/m/l/x 检测模型，自动发现脚本目录中的 `yolo26n.pt`、`yolo26s.pt`、`yolo26m.pt`、`yolo26l.pt`、`yolo26x.pt`。使用 CUDA GPU 0、固定输入和 batch=1，为每个型号导出八种矩形引擎到同一目录。标称高度最高为 1080；实际输入会按步幅 32 向上对齐。权重必须与文件名型号一致，仅重命名不会改变模型规模；分割、姿态和分类权重不在支持范围内。

## 使用方法

在已配置 GPU 依赖的虚拟环境中执行，权重须自行准备：

```bash
# 列出档位，不加载模型或使用 GPU
python export_all_engines.py --list
# 检查 CUDA 13 后端并执行小型 GPU 推理，不导出引擎
python export_all_engines.py --check-env
# 为发现的全部本地型号批量导出，默认跳过已有目标
python export_all_engines.py
# 应用默认扫描脚本目录中的引擎
python app_ws-multi.py
```

只导出指定型号，例如已准备 yolo26m.pt、yolo26l.pt、yolo26x.pt：

```bash
python export_all_engines.py --models m l x --list
python export_all_engines.py --models m l x
# 仅重新导出 m 的全部尺寸，不覆盖其他型号
python export_all_engines.py --models m --overwrite
```

没有指定 `--models` 时，只导出本地已存在权重的型号；例如只有 yolo26s.pt，行为仍为导出 s 的八种尺寸。没有权重时，`--list` 展示全部支持档位；实际导出则提示自备权重并退出。指定型号的权重缺失且仍有待构建档位时，会在 GPU 环境检查之前报错。`--check-env` 可独立运行，不要求权重。

可用 `--workspace 4` 将 TensorRT 构建工作区上限设为 4 GiB；这不是整个导出进程的显存或系统内存上限。省略时由 TensorRT 自动决定。

公开参数为 `--models`、`--list`、`--check-env`、`--overwrite`、`--workspace` 和帮助参数。当前不提供自选权重路径、尺寸、输出目录或 GPU 编号的命令行参数。应用可用 `ENGINE_DIR` 指向另一个已有引擎目录，该变量不改变导出工具的输出位置。

## 尺寸与文件

下表以 s 为例，其余型号使用相同尺寸，并将文件名前缀替换为相应型号，例如 `yolo26x_1920x1080.engine`。多个型号可共存，重启应用后显示在图片、视频及实时检测的引擎列表中。

| 文件名 | 实际输入宽×高 |
| --- | --- |
| yolo26s_1280x720.engine | 1280×736 |
| yolo26s_1920x1080.engine | 1920×1088 |
| yolo26s_854x480.engine | 864×480 |
| yolo26s_1706x960.engine | 1728×960 |
| yolo26s_640x480.engine | 640×480 |
| yolo26s_960x720.engine | 960×736 |
| yolo26s_1280x960.engine | 1280×960 |
| yolo26s_1440x1080.engine | 1440×1088 |

文件名保留标称画面尺寸，与应用的步幅对齐规则一致。不能通过改名改变引擎实际输入形状。输出为 `.engine`，不额外生成 JSON 参数文件。

每档依次在独立子进程执行 FP32 ONNX 导出、GPU 参考推理与混合精度转换、TensorRT 构建。构建使用已经转换的 FP16/FP32 混合精度 ONNX，不重复参考推理。当前流程每档重新导出 ONNX，不读取用户已有的 FP16 ONNX。

全部阶段成功且生成非空引擎后，才替换目标文件；单档失败保留原引擎并继续其余档位，最后以非零状态报告失败。临时中间文件正常结束时清理，不作为复用缓存。

## 环境与排错

本脚本要求 PyTorch 和 ONNX Runtime 均为 CUDA 13 构建、TensorRT 至少为 11，并具有 ModelOpt 混合精度接口。仅安装 CUDA 13 工具包不代表 Python 包已使用 CUDA 13。`requirements.txt` 只列应用层依赖，不安装或锁定导出所需 GPU 软件栈。

当前开发环境安装版本如下，仅供对照，不是完整的依赖锁文件，也不代表其他机器已通过全部尺寸导出验证：

| 组件 | 版本 |
| --- | --- |
| PyTorch | 2.15.0.dev20260928+cu130 |
| torchvision | 0.30.0.dev20261002+cu130 |
| Ultralytics | 8.4.171 |
| TensorRT | 11.3.0.99 |
| ONNX | 1.21.0 |
| ONNX Runtime GPU | 1.30.0 |
| NVIDIA ModelOpt | 0.47.0 |

参考推理使用 GPU 计算，仍会占用系统 RAM 保存模型及中间输出；当前最高 1080p 档位一次性采集中间输出。m/l/x 的资源需求可能明显高于 s，分辨率未超过 1080p 不代表内存一定足够。显存充足不能排除系统内存不足。退出码 137 或 -9 可能来自 OOM，也可能是外部强制终止，需要结合系统日志判断。

CUDA 后端缺失或初始化失败时，先处理 `--check-env` 的错误；脚本禁止静默退回纯 CPU 参考推理。若提示缺少 `Python.h`，需为实际运行的 Python 安装匹配的开发头文件，否则依赖触发扩展编译时可能失败。

请记录实际 GPU、驱动及软件版本。引擎跨设备或 TensorRT 版本不保证兼容，宜在部署环境导出。环境小型推理通过也不能代替真实模型的完整导出与准确率对比。
