# PDFLoom

> 面向医药 / CMC 文档的**版式保真 PDF 翻译服务**：自动识别原生、扫描、旧 OCR 层和混合 PDF，输出可搜索的单语 / 双语 PDF，并保留完整审计链路。

## 核心亮点

- pdf-translator SKILL:

- 

## 技术栈

| 类别 | 技术 |
| --- | --- |
| 语言与并发 | Python 3.12、asyncio |
| API 服务 | FastAPI、Uvicorn、Pydantic Settings、HTTPX |
| PDF 处理 | PyMuPDF、PDFMathTranslate、pdf2zh |
| OCR | PaddleOCR PP-StructureV3 |
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

### pdf-translator Skill

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

