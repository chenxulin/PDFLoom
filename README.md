# PDFLoom

> 面向医药 / CMC 文档的**版式保真 PDF 翻译服务**：自动识别原生、扫描、旧 OCR 层和混合 PDF，输出可搜索的单语 / 双语 PDF，并保留完整审计链路。

## 核心亮点

| 功能 | 如何实现 | 解决的问题 |
| --- | --- | --- |
| **全文级智能路由** | 使用 PyMuPDF 检查每一页的文字量、文本块、图片数量和图片覆盖率，将文档识别为原生、扫描、可搜索扫描或混合 PDF；原生 PDF 直达 PDFMathTranslate，任一页需要 OCR 时进入扫描链路 | 避免只看首页导致扫描附件漏译，也避免原生 PDF 被无意义 OCR |
| **干净的 OCR 中间层** | 调用 PaddleOCR PP-StructureV3 获取文字块、语义区域和表格结构；清除旧 OCR 隐藏层，只回填正文与标题，主动跳过表格、公式、图片、印章和低置信噪点 | 防止旧文字层、图片文字和表格内容被重复翻译 |
| **上下文串行翻译** | 按“页码 + 阅读顺序”逐段翻译，为模型提供前后段落、表头、当前行和文档片段；数据密集段落按标点拆分，失败时继续细分重试 | 减少跨段语义不一致、短标签误译和大段数据遗漏 |
| **数据完整性保护** | 翻译前将日期、数值、百分比、单位、化学式、批号和缩写替换为受保护占位符；翻译后逐项校验并恢复，缺失或被改写时任务失败 | 防止 `99.5%`、`A-001`、`mg/mL` 等关键事实被大模型修改 |
| **复杂表格矢量重建** | 解析 OCR 返回的 HTML / Markdown 表格，保留 `rowspan`、`colspan`；结合表头、列头、当前行和全文上下文翻译单元格，再清除整张源表并绘制可搜索矢量表格 | 避免译文覆盖原表、表格错列和数据串行 |
| **长表格自动分页** | 固定使用 9 pt、1.25 倍行距排版；空间不足时按完整数据行分页并重复表头，绝不通过无限缩小字体掩盖溢出；签字 / 日期表保留原图 | 保证长表可读、行列完整，并保护手写签名 |
| **医药术语与严格 QA** | 内置 CMC 术语约束和 OCR 纠错，如 `Assay → 含量测定`、`release → 放行`、`0OS → OOS`；最终校验目标语言、译文完整性、术语、页面尺寸、文字边界和非翻译区域背景相似度 | 对疑似未翻译、误译、越界或破版结果执行 fail-closed，不把坏文件当成功结果交付 |
| **Codex Skill 工程化** | 将完整的版式保真方法封装为 `pdf-translator` Skill：强制逐页内容台账、领域术语优先级、坐标化布局、溢出即失败，以及机械验证后的逐页视觉复核 | 把一次性提示词和脚本升级为可复用、可审计、质量标准一致的 Agent 工作流 |
| **异步任务与可审计产物** | FastAPI 接收任务，`asyncio.Semaphore` 控制并发；原子更新任务清单，记录阶段进度、耗时、SHA-256、OCR JSON、翻译台账、布局文件和验证报告 | 支持服务化部署、问题定位和结果追溯 |

## 技术栈

| 类别 | 技术 |
| --- | --- |
| 语言与并发 | Python 3.12、asyncio |
| API 服务 | FastAPI、Uvicorn、Pydantic Settings、HTTPX |
| PDF 处理 | PyMuPDF、PDFMathTranslate / pdf2zh 1.9.11 |
| OCR | PaddleOCR PP-StructureV3（远程服务） |
| 大模型 | DeepSeek、Openai、Kimi、GLM等 |
| Agent 工程化 | Codex Skill、SKILL.md 工作流、schema v1 布局协议 |
| 工程化 | Docker、Docker Compose、磁盘任务状态、SHA-256 审计 |
| 质量保障 | pytest、pytest-asyncio、Ruff |

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
- `layout.json` / `layout_verification.json`：版式描述与严格验证报告。
## 项目边界

- 仅处理 PDF，不负责 Word、Excel 或 OnlyOffice 文档。
- OCR 与大模型为外部依赖；服务本身不依赖数据库。
- `pdf2zh 1.9.11` 使用 AGPL-3.0，对外分发或商用部署前需完成许可证合规评估。
