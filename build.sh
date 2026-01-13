#!/bin/bash
# Hugo构建脚本 - 生成静态HTML

echo "🔨 开始构建Hugo静态站点..."
echo ""

# 清理旧的构建文件
if [ -d "public" ]; then
    echo "🧹 清理旧的构建文件..."
    rm -rf public
fi

# 构建站点
hugo --minify

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 构建完成！"
    echo "📁 静态文件已生成到 public/ 目录"
    echo "🌐 你可以在浏览器中打开 public/index.html 查看"
    echo ""
    echo "💡 提示: 使用以下命令在本地预览:"
    echo "   cd public && python3 -m http.server 8000"
else
    echo ""
    echo "❌ 构建失败，请检查错误信息"
    exit 1
fi
