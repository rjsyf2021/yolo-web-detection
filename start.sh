#!/bin/bash
# 启动 MediaMTX 与 YOLO 检测服务（图片/视频 + 实时）。本机与部署环境通用。
# 依赖：同目录的 start_mediamtx.sh、mediamtx.yml，以及已装好依赖的 Python 虚拟环境。
set -euo pipefail

# 切到脚本所在目录，不依赖调用方的当前目录。
cd -- "$(dirname -- "$0")"

# 后台启动 MediaMTX（其自身按版本/架构缓存二进制，不修改 Python 环境）。
"$PWD/start_mediamtx.sh" &
sleep 2

# 最多轮询 30 次 RTSP 监听端口；超时后仍会继续启动 Python 服务。
for _ in $(seq 1 30); do
    ss -tln 2>/dev/null | grep -q ':8554' && break
    sleep 0.1
done

# 激活虚拟环境；用 YOLO_VENV 指定其它路径，默认 ~/yolo_env。
# shellcheck disable=SC1091
source "${YOLO_VENV:-$HOME/yolo_env}/bin/activate"
exec python app_ws-multi.py
