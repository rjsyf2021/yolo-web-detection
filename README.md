# YOLO Web Detection

在浏览器中完成图片、视频批量检测与实时摄像头目标检测。支持手机、平板和电脑，使用 FastAPI、Ultralytics YOLO 与 TensorRT，在服务器集中执行推理。

这是一个持续迭代的个人项目。代码由开发者结合 AI 辅助完成，性能取决于输入、模型、浏览器和服务器配置。

## 功能

- 图片与完整视频上传，服务器推理、画框，结果预览和下载。
- 批量队列、1–4 项同时处理、停止任务及清除队列。
- JPEG / WebSocket 与 WebRTC / MediaMTX 两条实时传输路径。
- 客户端或服务器画框；横竖屏适配；4:3、16:9 模型与分辨率选择。
- WebRTC 10–60 FPS 目标帧率、0.5–50 Mbps 码率档位，分辨率优先策略。
- WebRTC FIFO 4 帧或单张待处理帧的低延迟模式。
- 启动模型预热、硬件解码与编码回退、各阶段耗时统计。

## 架构

~~~text
图片/视频文件 → FastAPI → 解码 → TensorRT → CPU画框 → 编码 → 下载
浏览器JPEG   → WebSocket → 解码 → TensorRT → 检测框或标注JPEG
浏览器WebRTC → MediaMTX → RTSP解码 → TensorRT → 检测框
                                             └→ 画框/编码 → MediaMTX → 浏览器
~~~

单个视频采用解码、推理、画框、编码流水线；不同视频各有画框与编码线程。共享模型的推理受锁保护，仍然串行。增加文件并发不保证单视频 FPS 或总吞吐提高。

## 部署

### 1. GPU 环境

使用已有的 Linux NVIDIA GPU 环境，先安装匹配驱动、CUDA、PyTorch、torchvision、Ultralytics 和 TensorRT。项目不自动升级这些组件，也不包含模型或 GPU 运行库。

在原来的虚拟环境中补充应用依赖：

~~~bash
source /path/to/yolo_env/bin/activate
python -m pip install -r requirements.txt
~~~

已有能运行的环境无需重复安装。requirements.txt 是依赖清单，不是完整、已验证的版本锁文件。已有服务器使用 PyAV 19.0.1；没有引入 aiortc。

### 2. 准备模型并启动

已有权重可使用随附的 export_all_engines.py 导出六种引擎，具体命令见 [模型导出说明](docs/EXPORT.md)。

将自己导出的 TensorRT 引擎放在 models/，文件名必须符合当前扫描规则，例如 yolo26s_1280x720.engine、yolo26s_1920x1080.engine。当前扫描器只接受 yolo26[nsmxl]_宽x高.engine，不是任意 YOLO 版本的通用加载器。

文件名描述名义宽高，代码按 32 步幅计算输入尺寸；必须与实际引擎匹配，不能靠改名改变模型输入。引擎需要与部署环境兼容。

~~~bash
ENGINE_DIR=./models python app_ws-multi.py
~~~

默认监听 0.0.0.0:7860。启动预热结束后浏览器访问服务器页面。远程手机摄像头需要浏览器信任的 HTTPS；localhost 可用于本机访问。HTTPS 反向代理需要支持 WebSocket。

### 3. WebRTC（可选）

JPEG 路径无需 MediaMTX。WebRTC 需要单独安装 [MediaMTX](https://mediamtx.org/docs/kickoff/install)，并使用与其版本兼容的配置。

~~~bash
cp mediamtx.example.yml mediamtx.yml
# 编辑 webrtcAdditionalHosts：把占位符换成客户端可达的服务器地址
./mediamtx mediamtx.yml
~~~

示例将 RTSP 8554 和信令 HTTP 8889 绑定回环地址，UDP 8189 用于 WebRTC 媒体传输；网页请求由应用代理信令。跨网访问还涉及防火墙、NAT 与 ICE，示例不是完整公网部署方案。已有正常工作的 MediaMTX 配置无需替换。

## 配置

| 环境变量 | 默认值 | 用途 |
| --- | --- | --- |
| ENGINE_DIR | 应用所在目录 | 引擎目录 |
| PORT | 7860 | HTTP 端口 |
| MEDIA_WORKERS | 4（限制为1–4） | 服务端文件任务工作线程数 |
| ENGINE_CACHE_SIZE | 未设置时启动尝试保留全部模型 | 显式设置可限制模型缓存；显存不足时可能驱逐模型 |
| MEDIAMTX_HTTP | http://127.0.0.1:8889 | WHIP/WHEP 信令地址 |
| MEDIAMTX_RTSP | rtsp://127.0.0.1:8554 | RTSP 地址 |

这些是进程环境变量；应用不会自动读取 .env。前端默认同时处理3项，资源紧张时先调为1项。

## 性能与已知限制

- 历史测试：Tesla T10、1080p 样例、yolo26s_1920x1080.engine、预热30帧后测300帧，串行基准约39.51 FPS，平均25.308 ms/帧。这是独立基准结果，不代表当前并行版的完整回归结果。
- 部分实时配置曾反馈约60 FPS；不保证所有设备和分辨率达到60 FPS。
- 客户端画框显示本地新画面，检测框可能滞后；服务器画框对应同帧，但增加编码回传延迟。
- NVDEC 后仍存在 CPU BGR 转换/搬运，CPU 画框；不是全 GPU 零拷贝管线。
- FIFO 可向上游累积延迟；低延迟模式会丢弃待处理旧帧。
- 码率、帧率及分辨率优先均受浏览器实际支持限制。
- 无账号鉴权或用户隔离，任务状态保存在单进程内存中。请用于受信任网络，不要直接裸露到公网，也不要开启多个 Uvicorn worker。
- 停止是协作式取消，不能保证立即打断正在执行的推理或音轨合成。任务结果不是永久存储。
- 当前发布整理只做源码检查，未在本机运行应用或测试，未完成最新版本服务器回归。

## 故障排查

启动找不到模型：检查 ENGINE_DIR、命名和输入尺寸。摄像头只有30 FPS：查看实际轨道与视频源统计，设备上限60并不表示该模式支持60。解码回退：提供终端中 [webrtc:...] 的异常类型和首帧日志，不应仅凭 invalid data 就认定硬件不支持。

## 发布与许可

见 [发布步骤](docs/PUBLISH.md) 和 [第三方依赖说明](THIRD_PARTY_NOTICES.md)。当前尚未替作者选择项目许可证，不应将本项目描述为 MIT 授权。公开发布前需确定与依赖兼容的许可。
