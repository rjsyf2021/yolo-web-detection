#!/usr/bin/env bash
# 仅在 Linux GPU 服务器上运行；本脚本不安装 Python 包。
set -euo pipefail
cd -- "$(dirname -- "$0")"
version=v1.21.1
case "$(uname -m)" in
  x86_64) arch=amd64 ;;
  aarch64|arm64) arch=arm64 ;;
  *) printf '%s\n' 'Unsupported architecture; download MediaMTX manually.' >&2; exit 1 ;;
esac
runtime="$PWD/.mediamtx-${version}-${arch}"
mkdir -p -- "$runtime"
# 按版本和架构缓存可执行文件，仅在缺失时下载；退出时清理临时压缩包。
if [ ! -x "$runtime/mediamtx" ]; then
  archive=$(mktemp)
  trap 'rm -f -- "$archive"' EXIT
  curl --fail --location --retry 3 --output "$archive" \
    "https://github.com/bluenviron/mediamtx/releases/download/${version}/mediamtx_${version}_linux_${arch}.tar.gz"
  tar -xzf "$archive" -C "$runtime" mediamtx
  chmod +x "$runtime/mediamtx"
  rm -f -- "$archive"
  trap - EXIT
fi
exec "$runtime/mediamtx" "$PWD/mediamtx.yml"
