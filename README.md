# PDF-Loom

这是从原系统中独立抽取出的 PDF 翻译服务。它不连接原项目数据库、不依赖
OnlyOffice，也不导入原项目的 `server` 包。原项目代码保持不变。

## 处理路径

服务会检查 PDF 的每一页，不只看首页：

- 扫描件、带旧 OCR 隐藏层的扫描件、扫描/文字混合件：
  `PaddleOCR PP-StructureV3 → 清洁 OCR 源文层 → PDFMathTranslate → 逐页串行 LLM → OCR ledger → layout.json → 原页背景保留渲染 → 表格矢量重绘 → 严格验证`
- 纯文字件：
  `PDFMathTranslate（内含 LLM 翻译与版面回填）`

扫描路径中的表格不会写入 OCR 中间文字层，避免 PDFMathTranslate 与表格专用
逻辑重复处理。服务读取 PP-StructureV3 的 HTML/Markdown 表格结构，保留
`rowspan` / `colspan`，锁定数字、日期、百分比、单位、化学式和标识符，只翻译
说明文字，然后清除整张源表并重绘为可搜索的矢量表格。默认从 9 pt、1.25 倍
行距排版；原页空间不足时按完整数据行分页并重复表头，不通过缩小文字掩盖溢出。
仍无法安全容纳则明确失败，不输出串列或溢出的交付件。签字/日期类表格默认保持
原图，避免破坏手写内容。

正文和标题按 PP-StructureV3 的段落区域重新聚合，不按每条 OCR 短行分别回填；
这样可以避免中文短行翻成英文后出现断词、窄框裁切。若 PDFMathTranslate 把相邻
段落合并成一个文字框，重绘阶段会在 OCR 坐标仍可无歧义对应时拆回完整译文段落，
否则任务明确失败。页眉、页脚、页码和图形区域继续按语义单独保护。

扫描路径会从 `translation_ledger.json` 自动生成 PDF Translator schema v1
兼容的 `layout.json`。原始页面以 300 DPI 背景保留，只覆盖正文和标题的 OCR
坐标框并写入可搜索译文；表格框在此阶段保持原样，随后交给现有矢量表格模块完整
替换。验证器检查每个原始页的尺寸映射、逐元素可搜索译文、文字边界，以及翻译框和
表格重绘区域之外的背景相似度。表格续页是显式允许的扩展页，正文和普通区域仍固定
在其原始页坐标内。

本仓库面向原系统的医药/CMC 场景，默认启用 `ENFORCE_CMC_TERMINOLOGY=true`。
短表格标签会结合表头、同行单元格和全文片段翻译；高风险术语还会做强制验收，
例如质控项目 `Assay → 含量测定`、批次处置语境中的
`release/released → 放行/批准放行`。模型首次违反术语约束时会带着更强上下文重试，
仍不合格则任务失败，不把“检测”“发布”等误译作为成功结果交付。只有另行配置了
通用领域提示词和 QA 的部署才应关闭这一开关。

对中文扫描件还会在进入翻译前修正高置信度的 `0OS` / `00S` / `O0S → OOS`
识别混淆；常见 CMC 表头和章节名使用精确目标词约束，例如
`产品批号 → Batch No.`、`TOC平行样/ppb → TOC parallel samples/ppb`、
`检验用具 → Test Utensils`。

## 目录

```text
ocr_pdf_agent/
├── src/ocr_pdf_agent/     # 分类、OCR、PDFMathTranslate、LLM、回填和 API
├── tests/                 # 扫描件/文字件及表格重绘自动测试
├── scripts/               # 样本生成与双路径 smoke test
├── storage/               # 默认任务存储（Git 忽略内容）
├── Dockerfile
└── compose.example.yml    # 独立部署示例，不接入父项目 deploy.sh
```

## 配置

```bash
cp .env.example .env
chmod 600 .env
```

至少配置：

- `API_KEY`、`BASE_URL`、`MODEL_NAME`：OpenAI-compatible 翻译模型。
- `PADDLEOCR_SERVICE_TOKEN`：远端 PaddleOCR 的独立服务令牌。
- `PADDLEOCR_API_URL`：默认沿用 `http://192.168.1.88:18093`。

对外或跨服务调用时应设置 `SERVICE_API_TOKEN`；设置后，所有 `/v1/*` 请求必须
携带 `X-OCR-PDF-Agent-Token`。`/health` 保持无鉴权，且只返回配置是否就绪，
不返回任何密钥。

服务也兼容迁移期变量 `ATTACHMENT_OCR_SERVICE_TOKEN`，但不会打印或返回令牌。

`BASE_URL` 和 `MODEL_NAME` 没有任何代码默认值或备用模型；两项必须在运行环境中
明确设置。缺失、为空或 URL 格式错误时，服务会直接拒绝启动，绝不会自动切换到
DeepSeek 公网接口或其他模型。

## 本地运行

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
ocr-pdf-agent serve --port 8010
```

父仓库所在计算节点没有宿主机 Python 时，可构建独立镜像；`compose.example.yml`
只是部署模板，不属于父项目 `joincare-translate-compute`，不要用父项目的
`deploy.sh` 管理它。

## HTTP API

上传任务：

```bash
curl -F 'file=@document.pdf;type=application/pdf' \
  -F 'source_language=auto' \
  -F 'target_language=zh-CN' \
  -H 'X-OCR-PDF-Agent-Token: <service-token>' \
  http://127.0.0.1:8010/v1/jobs
```

查询与下载：

```bash
curl http://127.0.0.1:8010/v1/jobs/<job_id>
curl -O http://127.0.0.1:8010/v1/jobs/<job_id>/artifacts/translated
curl -O http://127.0.0.1:8010/v1/jobs/<job_id>/artifacts/bilingual
```

可下载的 artifact 名称为：`translated`、`bilingual`、`manifest`、`ocr`、
`ocr-input`、`ledger`、`layout`、`layout-render-report`、
`layout-verification`、`source`。扫描任务的 `ledger` 按完整正文段落、标题和
表格单元格记录坐标、原文、译文及受保护字面量；每个任务目录还保留分类证据、
OCR JSON、中间 PDF、最终 PDF、SHA-256 和分阶段耗时，便于排错与审计。

默认启用 `STRICT_OUTPUT_QA=true`：目标语脚本缺失、输出无文字或译文与原文
几乎相同、或源文命中的强制 CMC 术语未出现在译文中时，任务会失败而不会把疑似
未翻译/误译文件作为成功结果交付。

## CLI

```bash
ocr-pdf-agent classify input.pdf
ocr-pdf-agent translate input.pdf --source-language auto \
  --target-language zh-CN --output-dir ./output
```

## 测试

离线测试不调用外部服务，但会真实生成扫描型 PDF 和文字型 PDF、执行路由、
生成 OCR 中间层并验证表格矢量重绘：

```bash
PYTHONPATH=src python -m unittest discover -v
PYTHONPATH=src:. python scripts/run_smoke.py --offline \
  --output-dir test-results/offline-smoke
```

配置真实 OCR/LLM 后执行双路径端到端测试：

```bash
PYTHONPATH=src:. python scripts/run_smoke.py \
  --output-dir test-results/live-smoke
```

扫描件 smoke test 除了检查路由、OCR、正文回填和表格重绘，还会从最终 PDF 反向
提取文字，强制确认含有“含量测定”“放行”和原始数值 `99.5`，并确认不含
“检测”“发布”以及残留英文 `Assay` / `Reviewed for release`。

## 边界与许可证

- 该服务只接受 PDF，不负责 Word/Excel/OnlyOffice。
- OCR 仍是远端依赖；本项目不会创建、启动、删除或重建 PaddleOCR。
- PDFMathTranslate/pdf2zh 1.9.11 使用 AGPL-3.0，上线或对外分发前应完成相应
  许可证合规评估。
>>>>>>> b775206 (PDF Translate tools)
