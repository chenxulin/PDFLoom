# PDFLoom

> 面向医药 / CMC 文档的**版式保真 PDF 翻译服务**：自动识别原生、扫描、旧 OCR 层和混合 PDF，输出可搜索的单语 / 双语 PDF，并保留完整审计链路。

## 核心亮点

| 功能 | 如何实现 | 解决的问题 |
| --- | --- | --- |
| **全文级智能路由** | 使用 PyMuPDF 检查每一页的文字层和图片覆盖率；扫描页占主导时进入 OCR 链路，混合 PDF 按同一文档级规则处理 | 与 Joincare 当前扫描件路由保持一致，避免首页决定整份文件 |
| **干净的 OCR 中间层** | 调用 PaddleOCR PP-StructureV3 逐页识别；清除旧 OCR 层、遮除源文字和页眉页脚，再写入唯一可见的 OCR 源文层 | 防止原扫描字形、旧隐藏层和译文重叠 |
| **上下文串行翻译** | 按“页码 + 阅读顺序”逐段翻译，为模型提供前后段落、表头、当前行和文档片段；数据密集段落按标点拆分，失败时继续细分重试 | 减少跨段语义不一致、短标签误译和大段数据遗漏 |
| **数据完整性保护** | 翻译前将日期、数值、百分比、单位、化学式、批号和缩写替换为受保护占位符；翻译后逐项校验并恢复，缺失或被改写时任务失败 | 防止 `99.5%`、`A-001`、`mg/mL` 等关键事实被大模型修改 |
| **复杂表格矢量重建** | 解析 OCR 返回的 HTML / Markdown 表格，保留 `rowspan`、`colspan`；结合表头、列头、当前行和全文上下文翻译单元格，再清除整张源表并绘制可搜索矢量表格 | 避免译文覆盖原表、表格错列和数据串行 |
| **长表格自动分页** | 固定使用 9 pt、1.25 倍行距排版；空间不足时按完整数据行分页并重复表头，绝不通过无限缩小字体掩盖溢出；签字 / 日期表保留原图 | 保证长表可读、行列完整，并保护手写签名 |
| **医药术语与扫描件策略** | 保留独立的 CMC 术语约束和 OCR 纠错，如 `Assay → 含量测定`、`release → 放行`、`0OS → OOS`；扫描件完成排版与表格重绘后不再运行额外自动 QA | 与 Joincare 当前扫描件交付策略一致，同时保留 Agent 自己的术语与模型配置 |
| **Codex Skill 工程化** | 保留 `pdf-translator` Skill 供人工逐页审阅或定制交付使用；HTTP 扫描件链路则固定走 Joincare 同款 OCR、v1 翻译和后处理流程 | 将可选的人工版式工作流与服务端自动扫描翻译明确分离 |
| **异步任务与可审计产物** | FastAPI 接收任务，`asyncio.Semaphore` 控制作业；OCR 按页限流、方向复核、内容缓存并原子更新任务清单 | 支持独立服务部署、问题定位和结果追溯 |

## 技术栈

| 类别 | 技术 |
| --- | --- |
| 语言与并发 | Python 3.12、asyncio |
| API 服务 | FastAPI、Uvicorn、Pydantic Settings、HTTPX |
| PDF 处理 | PyMuPDF、PDFMathTranslate / pdf2zh 1.9.11 |
| OCR | PaddleOCR PP-StructureV3（远程服务） |
| 大模型 | DeepSeek、Openai、Kimi、GLM等 |
| Agent 工程化 | Codex Skill、独立 HTTP 服务、磁盘任务状态 |
| 工程化 | Docker、Docker Compose、磁盘任务状态、SHA-256 审计 |
| 质量保障 | pytest、pytest-asyncio、Ruff；扫描件自动 QA 与 Joincare 当前策略一致（关闭） |

## 快速开始

### Docker

```bash
cp .env.example .env
# 填写 API_KEY、BASE_URL、MODEL_NAME、PADDLEOCR_API_URL 和 PADDLEOCR_SERVICE_TOKEN
docker compose -f compose.example.yml up --build
```

服务默认监听 `http://127.0.0.1:28510`。

### 本地运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
ocr-pdf-agent serve --port 8010
```

## 使用方式

### Codex Skill

仓库已包含完整 Skill 源码、执行脚本和版式 / 领域规则：[`skills/pdf-translator/`](skills/pdf-translator/)。安装后即可通过自然语言触发完整工作流。

```bash
mkdir -p ~/.codex/skills
cp -R skills/pdf-translator ~/.codex/skills/
python3 -m pip install -r skills/pdf-translator/requirements.txt
```

示例提示词：`使用 $pdf-translator 将这份 PDF 翻译为英文，并保持原始版式。`

### HTTP API

```bash
# 创建翻译任务
curl -X POST http://127.0.0.1:28510/v1/jobs \
  -H 'X-OCR-PDF-Agent-Token: <service-token>' \
  -F 'file=@document.pdf;type=application/pdf' \
  -F 'source_language=auto' \
  -F 'target_language=zh-CN'

# 查询状态并下载结果
curl -H 'X-OCR-PDF-Agent-Token: <service-token>' \
  http://127.0.0.1:28510/v1/jobs/<job_id>
curl -OJ -H 'X-OCR-PDF-Agent-Token: <service-token>' \
  http://127.0.0.1:28510/v1/jobs/<job_id>/artifacts/translated
```

| Endpoint | 说明 |
| --- | --- |
| `GET /health` | 服务及依赖配置状态 |
| `POST /v1/jobs` | 上传 PDF，异步创建翻译任务 |
| `GET /v1/jobs/{job_id}` | 查询阶段、进度、耗时和结果 |
| `GET /v1/jobs/{job_id}/artifacts/{name}` | 下载 PDF 或审计产物 |

### CLI

```bash
ocr-pdf-agent classify input.pdf
ocr-pdf-agent translate input.pdf --source-language auto \
  --target-language zh-CN --output-dir ./output
```

## 输出产物

- `translated.pdf` / `bilingual.pdf`：可搜索的单语 / 双语译文。
- `manifest.json`：任务状态、阶段耗时、校验结果和文件哈希。
- `ocr_ppstructurev3.json`：结构化 OCR 原始结果。
- `translation_ledger.json`：带坐标和受保护数据的翻译台账。
- `ocr_pdfmathtranslate_input.pdf`：清理过旧文字层的 v1 OCR 中间文件。
## 项目边界

- 仅处理 PDF，不负责 Word、Excel 或 OnlyOffice 文档。
- OCR 与大模型为外部依赖；服务本身不依赖数据库。
- `pdf2zh 1.9.11` 使用 AGPL-3.0，对外分发或商用部署前需完成许可证合规评估。
