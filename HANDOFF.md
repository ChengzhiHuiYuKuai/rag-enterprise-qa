# RAG 企业知识库智能问答系统 - 项目交接文档

## 项目概述

基于 LangChain + LangGraph 的 RAG 问答系统，支持企业上传内部文档（PDF/Word/Markdown/Excel），通过混合检索（向量 + BM25）实现精准问答，支持 SSE token 级流式输出。

**技术栈**：LangChain/LangGraph + ChromaDB + BGE-M3 + DeepSeek + FastAPI + Streamlit + Docker

---

## 当前完成进度

### ✅ 已完成

| 模块 | 状态 | 说明 |
|------|------|------|
| 项目骨架 | ✅ | 完整目录结构，依赖配置 |
| 配置管理 | ✅ | `.env` 环境变量，pydantic-settings |
| 文档处理 | ✅ | PDF/Word/Markdown/Excel 加载与切分 |
| 向量存储 | ✅ | ChromaDB 嵌入式模式，持久化 |
| Embedding | ✅ | text2vec-base-chinese，CPU 推理，低内存占用 |
| 混合检索 | ✅ | 向量检索 + BM25，加权融合（自动从 ChromaDB 加载索引） |
| LangGraph 状态图 | ✅ | 精简版：检索 → 生成 |
| FastAPI 接口 | ✅ | 对话/文档管理/健康检查 |
| Streamlit 前端 | ✅ | 聊天界面 + 文档管理侧边栏 |
| Docker 配置 | ✅ | Dockerfile + docker-compose.yml |
| 流式响应 | ✅ | SSE token 级流式输出，使用 `astream_events` |
| LLM 切换 | ✅ | 已切换为 DeepSeek-V4-Flash（低延迟） |
| 代码质量修复 | ✅ | 混合检索启用、LLM 单例、SSE 异常处理等 |

### 🔄 待优化

| 项目 | 状态 | 说明 |
|------|------|------|
| 查询改写 | ⏸️ 已禁用 | 为提速暂时去掉了，后续可加回 |
| 相关性判断 | ⏸️ 已禁用 | 同上 |
| 文档元数据持久化 | ✅ 已完成 | `_doc_registry` 持久化到 `data/doc_registry.json` |
| 对话历史 token 截断 | ✅ 已完成 | 按 `max_history_tokens`（默认2000）截断，防止 token 消耗二次方增长 |

---

## 关键配置

### .env 文件

```env
# LLM（DeepSeek）
MIMO_API_KEY=your-deepseek-api-key
MIMO_BASE_URL=https://api.deepseek.com/v1
MIMO_MODEL=deepseek-v4-flash

# Embedding（轻量模型，适合 CPU / 低内存服务器）
EMBEDDING_MODEL=shibing624/text2vec-base-chinese
EMBEDDING_DEVICE=cpu
HF_ENDPOINT=https://hf-mirror.com

# 检索（混合检索已启用）
TOP_K=5
BM25_WEIGHT=0.3
VECTOR_WEIGHT=0.7

# 对话历史（token 预算截断）
MAX_HISTORY_TOKENS=2000
```

### 环境信息

- Python 3.11
- PyTorch 2.11.0+cpu（服务器 CPU 推理，本地开发可用 cu128 GPU 版本）
- Embedding 模型：`text2vec-base-chinese`（~400MB，CPU 推理，适合低内存服务器）
- ChromaDB 数据存储在 `data/chroma/`
- LLM：DeepSeek-V4-Flash（低延迟，兼容 OpenAI 协议）
- 文档元数据持久化到 `data/doc_registry.json`

---

## 启动命令

```bash
# 激活虚拟环境
.venv\Scripts\activate

# 启动后端（自动 reload）
python -m app.main

# 启动前端（另一个终端）
streamlit run frontend/app.py
```

- 后端 API：http://localhost:8000
- API 文档：http://localhost:8000/docs
- 前端界面：http://localhost:8501

### Docker 部署（推荐）

```bash
# 构建并启动
docker compose up -d --build

# 查看日志
docker compose logs -f

# 重启
docker compose restart
```

**注意事项：**
- 服务器无 GPU 时，使用 CPU 版 PyTorch（`requirements.txt` 中 `torch>=2.1.0+cpu`）
- 配置国内镜像加速：Dockerfile 中已配置 apt（阿里云）+ pip（阿里云）+ HuggingFace（hf-mirror）
- 数据持久化：`data/` 目录挂载到宿主机，重启不丢失

---

## 下一步待做

### 可选优化

- 加回查询改写（多轮对话指代消解）
- 加回相关性判断（兜底策略）
- 加缓存机制（相同问题秒回）
- 对话历史按 token 截断

---

## 项目结构

```
rag-enterprise-qa/
├── app/
│   ├── api/
│   │   ├── chat.py              # 对话接口（SSE token 级流式输出）
│   │   ├── documents.py         # 文档 CRUD（上传后自动刷新 BM25 索引）
│   │   └── health.py            # 健康检查
│   ├── core/
│   │   ├── graph.py             # LangGraph 状态图（精简版：retrieve → generate）
│   │   ├── nodes.py             # 节点实现（LLM 单例，混合检索）
│   │   ├── state.py             # 状态定义
│   │   └── retrieval.py         # 混合检索（向量 + BM25，自动加载索引）
│   ├── document/
│   │   ├── loader.py            # 多格式加载器
│   │   └── splitter.py          # 切分策略
│   ├── vectorstore/
│   │   ├── chroma.py            # ChromaDB 封装
│   │   └── embeddings.py        # BGE-M3 + HuggingFace 镜像
│   ├── models/                  # Pydantic 数据模型
│   ├── config.py                # 配置管理
│   └── main.py                  # FastAPI 入口
├── frontend/app.py              # Streamlit 聊天界面（SSE 流式消费）
├── data/
│   ├── 中考科学知识点.md         # 测试文档
│   ├── uploads/                 # 上传文件存储
│   └── chroma/                  # ChromaDB 持久化
├── tests/                       # 测试用例
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env                         # 环境变量（已配置）
├── .env.example                 # 环境变量模板
└── README.md
```

---

## 已知问题

1. **PyTorch CUDA 兼容**：RTX 5060 需要 PyTorch 2.11+cu128，低版本不兼容
2. **HuggingFace 访问**：需要设置 `HF_ENDPOINT=https://hf-mirror.com` 镜像
3. **前端控制台报错**：`ERR_CONNECTION_TIMED_OUT` 是页面无关资源加载，不影响功能
4. **文档元数据重启丢失**：✅ 已修复 — `_doc_registry` 已持久化到 `data/doc_registry.json`，重启后文档列表不丢失
