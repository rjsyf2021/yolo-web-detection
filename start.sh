#!/usr/bin/env bash
# 前台管理 MediaMTX 和应用；任一进程退出时回收本次启动的全部子进程。
set -euo pipefail
cd -- "$(dirname -- "$0")"
if ((BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1))); then
    echo 'start.sh 需要 Bash 5.1 或更新版本。' >&2
    exit 1
fi

venv_path="${YOLO_VENV:-$HOME/yolo_env}"
startup_timeout="${MEDIAMTX_START_TIMEOUT:-60}"
[[ -f mediamtx.yml ]] || { echo '缺少 mediamtx.yml，请先复制并编辑 mediamtx.example.yml' >&2; exit 1; }
[[ -x start_mediamtx.sh ]] || { echo 'start_mediamtx.sh 不存在或不可执行' >&2; exit 1; }
[[ -x "$venv_path/bin/python" && -f "$venv_path/bin/activate" ]] || {
    echo "无效虚拟环境：$venv_path；请设置 YOLO_VENV" >&2; exit 1;
}
[[ "$startup_timeout" =~ ^[1-9][0-9]*$ ]] || { echo 'MEDIAMTX_START_TIMEOUT 必须为正整数秒数' >&2; exit 1; }
for tool in ss setsid; do
    command -v "$tool" >/dev/null || { echo "缺少启动依赖：$tool" >&2; exit 1; }
done
# 激活失败时尚未创建后台进程。
# shellcheck disable=SC1091
source "$venv_path/bin/activate"

rtsp_ready() {
    local listeners
    listeners=$(ss -H -ltn 'sport = :8554') || return 1
    [[ -n "$listeners" ]]
}
if rtsp_ready; then
    echo '8554 端口已被占用。已有 MediaMTX 时请直接启动 python app_ws-multi.py。' >&2
    exit 1
fi

children=()
cleanup() {
    local pid pending attempt
    trap - EXIT INT TERM
    # setsid 为每个服务建立独立进程组；仅回收本脚本创建的组，包含下载等子进程。
    for pid in "${children[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
    for ((attempt=0; attempt<50; attempt++)); do
        pending=0
        for pid in "${children[@]}"; do
            if kill -0 -- "-$pid" 2>/dev/null; then pending=1; fi
        done
        ((pending)) || break
        sleep 0.1
    done
    for pid in "${children[@]}"; do
        kill -KILL -- "-$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

setsid "$PWD/start_mediamtx.sh" &
media_pid=$!
children+=("$media_pid")
deadline=$((SECONDS + startup_timeout))
until rtsp_ready; do
    if ! kill -0 "$media_pid" 2>/dev/null; then
        echo 'MediaMTX 启动失败，请查看上方日志。' >&2
        exit 1
    fi
    if ((SECONDS >= deadline)); then
        echo "MediaMTX 在 ${startup_timeout} 秒内未监听 8554，已停止启动。" >&2
        exit 1
    fi
    sleep 0.1
done
if ! kill -0 "$media_pid" 2>/dev/null; then
    echo 'MediaMTX 已退出，未启动应用。' >&2
    exit 1
fi

setsid "$venv_path/bin/python" "$PWD/app_ws-multi.py" &
app_pid=$!
children+=("$app_pid")
status=0
wait -n -p finished "$media_pid" "$app_pid" || status=$?
if [[ "${finished:-}" == "$media_pid" ]]; then
    echo 'MediaMTX 已退出，正在停止应用。' >&2
    ((status != 0)) || status=1
fi
exit "$status"
