#!/bin/bash
# Hugo本地预览脚本

echo "🚀 启动Hugo本地预览服务器..."
echo "📝 访问 http://localhost:1313 查看博客"
echo "⏹️  按 Ctrl+C 停止服务器"
echo ""

hugo server -D --bind 0.0.0.0 --baseURL http://localhost:1313/
