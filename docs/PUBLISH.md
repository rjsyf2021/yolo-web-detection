# 上传到 GitHub

建议仓库名：yolo-web-detection

建议 Description：Browser-based YOLO detection with TensorRT, batch image/video processing, WebSocket and WebRTC streaming.

建议 Topics：yolo, tensorrt, fastapi, webrtc, mediamtx, object-detection, computer-vision

1. 先确定项目许可，参考 THIRD_PARTY_NOTICES.md。
2. 在 GitHub 创建仓库，把本目录内容作为仓库根目录上传，不要上传外层压缩包。
3. 保留 .gitignore；不要上传真实 mediamtx.yml、虚拟环境、模型、私人照片视频或访问凭据。
4. README 可补充自己的演示截图和视频，注意获得画面中人物的授权并隐藏网络信息。
5. 首个版本建议标记为实验版；在服务器验证图片、带音轨视频、两种实时路径、停止任务、横竖屏和并发后再记录测试环境。

若使用 Git 命令，在项目目录执行：

~~~bash
git init
git add .
git diff --cached --stat
git commit -m "Initial release: YOLO web detection"
git branch -M main
git remote add origin <你的仓库HTTPS地址>
git push -u origin main
~~~

本包尚未创建远程仓库或推送。GPU 组件版本应从实际服务器环境记录；不要把历史测试版本当作可复现的完整环境锁。
