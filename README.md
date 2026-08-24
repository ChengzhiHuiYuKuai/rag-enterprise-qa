# 📚 RAG 企业知识库智能问答系统

基于 **LangChain + LangGraph** 构建的 RAG 问答系统，支持企业上传内部文档（PDF/Word/Markdown/Excel），通过混合检索实现精准问答，支持 SSE token 级流式输出。

## ✨ 核心特性

- 🔗 **LangGraph 状态图编排** — 混合检索 → 生成回答，精简高效
- 📄 **多格式文档支持** — PDF、Word、Markdown、Excel 自动解析与切分
- 🔍 **混合检索策略** — 向量检索（语义相似）+ BM25（关键词匹配），加权融合
- 💬 **多轮对话记忆** — Token 预算截断，防止历史累积导致 token 消耗失控
- 🌊 **SSE 流式响应** — `astream_events` 实现 token 级真正的流式输出
- 🖥️ **可视化前端** — Streamlit 聊天界面，文档管理 + 智能问答

## 🏗️ 技术架构

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Streamlit   │────▶│   FastAPI    │────▶│  LangGraph  │
│   Frontend   │◀────│   Backend    │◀────│  StateGraph │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                                    ┌───────────┴───────────┐
                                    ▼                       ▼
                            ┌──────────────┐        ┌──────────────┐
                            │   Hybrid     │        │  LLM         │
                            │   Retriever  │        │  (DeepSeek)  │
                            └──────┬───────┘        └──────────────┘
                                   │
                           ┌───────┴───────┐
                           ▼               ▼
                   ┌────────────┐  ┌────────────┐
                   │  ChromaDB  │  │   BM25     │
                   │  (Vector)  │  │  (Keyword) │
                   └────────────┘  └────────────┘
```

### LangGraph 状态图流程

```
用户提问 → 混合检索（向量 + BM25）→ 生成回答（SSE 流式输出）
```

## 🛠️ 技术栈

| 组件 | 技术选型 | 说明 |
|------|----------|------|
| LLM | DeepSeek-V4-Flash | 低延迟，OpenAI 兼容协议 |
| Embedding | text2vec-base-chinese | 中文 Embedding，CPU 推理，低内存占用 |
| 向量存储 | ChromaDB | 嵌入式向量数据库，零依赖 |
| 关键词检索 | BM25 (rank-bm25) | 经典关键词匹配算法 |
| 编排框架 | LangGraph | 有状态图编排，`astream_events` 流式 |
| 后端 API | FastAPI | RESTful 接口，SSE 流式响应 |
| 前端 | Streamlit | 快速搭建聊天界面 |
| 部署 | Docker Compose | 一键容器化部署 |

## 📁 项目结构

```
rag-enterprise-qa/
├── app/
│   ├── api/                     # API 路由
│   │   ├── chat.py              # 对话接口 (SSE 流式)
│   │   ├── documents.py         # 文档上传/管理
│   │   └── health.py            # 健康检查
│   ├── core/                    # 核心逻辑
│   │   ├── graph.py             # LangGraph 状态图定义 ⭐
│   │   ├── nodes.py             # 图节点实现
│   │   ├── state.py             # 状态定义
│   │   └── retrieval.py         # 混合检索策略
│   ├── document/                # 文档处理
│   │   ├── loader.py            # 多格式加载器
│   │   └── splitter.py          # 切分策略
│   ├── vectorstore/             # 向量存储
│   │   ├── chroma.py            # ChromaDB 封装
│   │   └── embeddings.py        # Embedding 模型管理
│   ├── models/                  # 数据模型
│   │   ├── chat.py              # 对话请求/响应
│   │   └── document.py          # 文档元数据
│   ├── config.py                # 配置管理
│   └── main.py                  # FastAPI 入口
├── frontend/
│   └── app.py                   # Streamlit 聊天界面
├── data/
│   ├── uploads/                 # 上传的文件
│   ├── chroma/                  # ChromaDB 持久化数据
│   └── doc_registry.json        # 文档元数据持久化
├── docker-compose.yml           # Docker 编排
├── Dockerfile
├── .dockerignore                # Docker 构建排除
├── requirements.txt
├── .env.example                 # 环境变量模板
└── README.md
```

## 🚀 快速开始

### 1. 环境准备

```bash
# Python 3.10+
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key
```

关键配置：
```env
MIMO_API_KEY=your_deepseek_api_key
MIMO_BASE_URL=https://api.deepseek.com/v1
MIMO_MODEL=deepseek-v4-flash
EMBEDDING_MODEL=shibing624/text2vec-base-chinese
EMBEDDING_DEVICE=cpu
```

### 3. 启动服务

```bash
# 启动后端 API
python -m app.main

# 另一个终端，启动前端
streamlit run frontend/app.py
```

访问：
- API 文档: http://localhost:8000/docs
- 聊天界面: http://localhost:8501

### 4. Docker 部署

```bash
# 构建并启动（首次构建约 5-10 分钟）
docker compose up -d --build

# 查看日志
docker compose logs -f

# 重启
docker compose restart
```

> **服务器部署注意**：无 GPU 时使用 CPU 版 PyTorch，Dockerfile 已配置国内镜像加速（apt/pip/HuggingFace）。

## 📡 API 接口

### 对话

```bash
# 普通对话
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "公司的年假政策是什么？", "session_id": "test"}'

# 流式对话 (SSE)
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "公司的年假政策是什么？", "stream": true}'
```

### 文档管理

```bash
# 上传文档
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@company_handbook.pdf"

# 查看文档列表
curl http://localhost:8000/api/documents

# 删除文档
curl -X DELETE http://localhost:8000/api/documents/{doc_id}
```

## 🔍 核心设计亮点

### 1. LangGraph 状态图

使用 LangGraph 的 `StateGraph` 定义精简的 RAG 流程：

- **混合检索**: 向量检索（语义理解）+ BM25（精确匹配），自动从 ChromaDB 加载 BM25 索引
- **生成回答**: LLM 根据检索上下文生成回答

可通过修改 `graph.py` 轻松扩展查询改写、相关性判断等节点。

### 2. 混合检索

单靠向量检索会漏掉精确关键词匹配，单靠 BM25 又缺乏语义理解。混合策略结合两者优势：

```
final_score = vector_weight × vector_similarity + bm25_weight × bm25_score
```

BM25 索引在文档上传/删除时自动刷新，首次检索时自动从 ChromaDB 加载。

### 3. SSE 流式输出

使用 LangGraph 的 `astream_events` API 捕获 LLM 逐 token 的流式输出：

- 后端：`astream_events(version="v2")` 捕获 `on_chat_model_stream` 事件
- 前端：`requests.post(stream=True)` + `iter_lines()` 解析 SSE 事件流
- 效果：首 token 延迟与 LLM 首 token 一致，用户看到逐字生成效果

### 4. 文档切分策略

使用 `RecursiveCharacterTextSplitter`，针对中英文档优化分隔符优先级：
- 段落 (`\n\n`) → 换行 (`\n`) → 中文句号 (`。`) → 英文句号 (`. `) → 字符级

## 📊 性能指标

| 指标 | 数据 | 说明 |
|------|------|------|
| 文档处理 | ~7 秒/次 | Embedding 模型热启动，10 个块入库 |
| 问答响应 | ~10 秒 | 含混合检索 + LLM 推理（DeepSeek API） |
| 检索准确率 | Top-5 命中率 100% | 中文知识文档测试，混合检索优于纯向量 |
| 支持格式 | 4 种 | PDF / Word / Markdown / Excel |
| 分块策略 | 512 字符/块 | 64 字符重叠，中文分隔符优先 |

## 📄 License

MIT
