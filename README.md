# PDFLoom

这是一个面向文字型、扫描型及复杂版式 PDF的智能翻译Agent工具，构建了包含文档预检查、智能分流、OCR、受控翻译、版面重建、质量校验与审计交付的端到端处理链路，并通过独立 HTTP 服务对外提供异步任务能力。支持通过HTTP接口接入到现有项目中。

## 核心能力

| 能力 | 说明 |
| --- | --- |
| 文档智能分流 | 在翻译前预检 PDF，识别文字型与扫描型文档，并选择对应处理链路 |
| pdf-translator Skill | 将翻译、版面规范、术语约束和交付规则封装为可复用 Skill |
| 版式优先重建 | 统一正文和标题样式，支持表格重建及矢量文字输出 |
| 原始视觉证据保留 | 图片、图表、印章和签名等内容保留原始视觉呈现 |
| 受控术语翻译 | 使用预定义术语库约束专有名称及行业术语的译法 |
| 事实字段保护 | 保护数字、日期、单位、化学式、产品批号等关键内容 |
## 技术栈

| 能力 | 实现 |
| --- | --- |
| 服务与配置 | Python 3.12、FastAPI、Pydantic Settings、HTTPX、asyncio |
| PDF 与 OCR | PyMuPDF、PDFMathTranslate / pdf2zh、PaddleOCR PP-StructureV3 |
| 翻译 | OpenAI 兼容模型接口（如 DeepSeek、Kimi、GLM 等）与独立术语规则 |
| 工程化 | Docker Compose、异步任务、磁盘状态、SHA-256 审计、pytest |

## 快速开始

### Docker

```bash
cp .env.example .env
# 至少配置：API_KEY、BASE_URL、MODEL_NAME、PADDLEOCR_API_URL、PADDLEOCR_SERVICE_TOKEN
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

### HTTP API

```bash
# 创建异步翻译任务
curl -X POST http://127.0.0.1:28510/v1/jobs \
  -H 'X-OCR-PDF-Agent-Token: <service-token>' \
  -F 'file=@document.pdf;type=application/pdf' \
  -F 'source_language=auto' \
  -F 'target_language=zh-CN'

# 查询任务并下载单语译文
curl -H 'X-OCR-PDF-Agent-Token: <service-token>' \
  http://127.0.0.1:28510/v1/jobs/<job_id>
curl -OJ -H 'X-OCR-PDF-Agent-Token: <service-token>' \
  http://127.0.0.1:28510/v1/jobs/<job_id>/artifacts/translated
```

| Endpoint | 用途 |
| --- | --- |
| `GET /health` | 查看服务与 OCR / 模型配置状态 |
| `POST /v1/jobs` | 上传 PDF 并创建异步任务 |
| `GET /v1/jobs/{job_id}` | 查询进度、耗时、路由和产物 |
| `GET /v1/jobs/{job_id}/artifacts/{name}` | 下载 `translated`、`bilingual`、`manifest`、`ocr`、`ocr-input`、`ledger` 或 `source` |

### CLI

```bash
ocr-pdf-agent classify input.pdf
ocr-pdf-agent translate input.pdf --source-language auto \
  --target-language zh-CN --output-dir ./output
```

### pdf-translator Skill

仓库同时提供可安装的 SKILL，适合从自然语言工作流触发版式保真翻译：

```bash
mkdir -p ~/.codex/skills
cp -R skills/pdf-translator ~/.codex/skills/
python3 -m pip install -r skills/pdf-translator/requirements.txt
```

示例提示词：`使用 $pdf-translator 将这份 PDF 翻译为英文，并保持原始版式。`

