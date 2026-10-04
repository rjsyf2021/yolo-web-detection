# 导出 TensorRT 引擎

根据项目实际使用的 export_all_engines.py 整理，使用 FP16、固定输入、batch=1；没有运行本地导出或验证生成的引擎。

在服务器原 GPU 虚拟环境中执行，提供来源及许可明确的本地权重：

```bash
python export_all_engines.py --weights /path/to/yolo26s.pt --output-dir models
ENGINE_DIR=./models python app_ws-multi.py
```

只导出部分尺寸：

```bash
python export_all_engines.py --weights /path/to/yolo26s.pt --sizes 1280x720 1920x1080
```

默认跳过已有目标，显式传入 --overwrite 才覆盖。--device 可选择 GPU 编号。脚本不自动下载权重，临时导出中间文件会在结束后清理。

| 文件名中的宽高 | 实际输入宽高（步幅32对齐） |
| --- | --- |
| 640×480 | 640×480 |
| 960×720 | 960×736 |
| 1280×960 | 1280×960 |
| 1440×1080 | 1440×1088 |
| 1280×720 | 1280×736 |
| 1920×1080 | 1920×1088 |

这与应用的输入尺寸计算一致。文件名保留画面名义尺寸，不能将实际720高的引擎仅改名后假定为736高。输出旁边的 JSON 记录导出参数，不包含权重的来源认证信息。

请记录实际 GPU、驱动、CUDA、TensorRT、PyTorch 和 Ultralytics 版本。TensorRT 引擎不能保证跨设备、跨版本可用，优先在目标环境导出。脚本使用的 yolo26 权重需要安装支持它的 Ultralytics 版本。

用户提供的另一个 export_engines.py 使用单个整数作为 imgsz，会导出方形输入，未纳入本发布包。原文件夹中的测试素材、实际网络配置和模型均未打包。
