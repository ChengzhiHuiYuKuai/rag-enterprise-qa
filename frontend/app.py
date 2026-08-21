"""Streamlit 聊天界面

提供文档上传和智能问答的可视化界面。
"""

import json

import requests
import streamlit as st

# 配置
API_BASE = "http://localhost:8000/api"

st.set_page_config(
    page_title="RAG 企业知识库问答",
    page_icon="📚",
    layout="wide",
)

# ============================================================
# 侧边栏：文档管理
# ============================================================
with st.sidebar:
    st.title("📚 文档管理")

    # 文档上传
    uploaded_file = st.file_uploader(
        "上传文档",
        type=["pdf", "docx", "md", "xlsx"],
        help="支持 PDF、Word、Markdown、Excel 格式",
    )

    if uploaded_file and st.button("📤 上传并处理", use_container_width=True):
        with st.spinner("正在处理文档..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                resp = requests.post(f"{API_BASE}/documents/upload", files=files)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(
                        f"✅ 上传成功！\n"
                        f"文件: {data['filename']}\n"
                        f"切分: {data['chunk_count']} 个文档块"
                    )
                    st.rerun()
                else:
                    st.error(f"上传失败: {resp.text}")
            except Exception as e:
                st.error(f"上传异常: {e}")

    st.divider()

    # 文档列表
    st.subheader("📋 已上传文档")
    try:
        resp = requests.get(f"{API_BASE}/documents")
        if resp.status_code == 200:
            docs = resp.json().get("documents", [])
            if not docs:
                st.info("暂无文档，请先上传")
            for doc in docs:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"📄 {doc['filename']}")
                    st.caption(f"{doc['chunk_count']} 个块 | {doc['file_size'] // 1024} KB")
                with col2:
                    if st.button("🗑️", key=f"del_{doc['doc_id']}"):
                        requests.delete(f"{API_BASE}/documents/{doc['doc_id']}")
                        st.rerun()
    except Exception as e:
        st.warning(f"获取文档列表失败: {e}")

    st.divider()

    # 系统信息
    try:
        resp = requests.get(f"{API_BASE}/stats")
        if resp.status_code == 200:
            stats = resp.json()
            st.subheader("⚙️ 系统配置")
            st.caption(f"向量库块数: {stats['vector_store']['total_chunks']}")
            st.caption(f"检索 Top-K: {stats['config']['top_k']}")
            st.caption(f"BM25 权重: {stats['config']['bm25_weight']}")
            st.caption(f"向量权重: {stats['config']['vector_weight']}")
    except Exception:
        pass


# ============================================================
# 主区域：聊天界面
# ============================================================
st.title("💬 企业知识库问答")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "default"

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 参考来源"):
                for src in msg["sources"]:
                    st.caption(f"📄 {src.get('source', '未知')}")
                    st.text(src.get("content", "")[:150])

# 用户输入
if prompt := st.chat_input("请输入你的问题..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用 API 获取回答（SSE 流式）
    with st.chat_message("assistant"):
        status_container = st.status("正在思考...", expanded=True)
        answer_placeholder = st.empty()
        full_answer = ""
        sources = []

        try:
            resp = requests.post(
                f"{API_BASE}/chat",
                json={
                    "question": prompt,
                    "session_id": st.session_state.session_id,
                    "stream": True,
                },
                stream=True,
                timeout=120,
            )

            if resp.status_code != 200:
                status_container.update(label="❌ 请求失败", state="error")
                full_answer = f"请求失败: {resp.text}"
                answer_placeholder.error(full_answer)
            else:
                # 解析 SSE 事件流
                event_type = ""
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        try:
                            data = json.loads(line[5:].strip())
                        except json.JSONDecodeError:
                            continue

                        if event_type == "metadata":
                            count = data.get("retrieved_count", 0)
                            status_container.update(
                                label=f"📚 检索到 {count} 个文档片段，正在生成回答..."
                            )

                        elif event_type == "token":
                            full_answer += data.get("token", "")
                            # 显示打字光标效果
                            answer_placeholder.markdown(full_answer + "▌")

                        elif event_type == "sources":
                            sources = data.get("sources", [])

                        elif event_type == "done":
                            st.session_state.session_id = data.get(
                                "session_id", st.session_state.session_id
                            )
                            break

                        elif event_type == "error":
                            raise Exception(data.get("error", "未知错误"))

                # 最终显示（去掉光标）
                status_container.update(label="✅ 回答完成", state="complete")
                answer_placeholder.markdown(full_answer)

                # 显示来源
                if sources:
                    with st.expander("📎 参考来源"):
                        for src in sources:
                            st.caption(f"📄 {src.get('source', '未知')}")
                            st.text(src.get("content", "")[:150])

        except Exception as e:
            status_container.update(label="❌ 异常", state="error")
            full_answer = f"请求异常: {e}"
            answer_placeholder.error(full_answer)

        # 保存助手消息
        st.session_state.messages.append(
            {"role": "assistant", "content": full_answer, "sources": sources}
        )

# 清除对话按钮
if st.session_state.messages:
    if st.button("🗑️ 清除对话"):
        requests.delete(f"{API_BASE}/chat/session/{st.session_state.session_id}")
        st.session_state.messages = []
        st.rerun()
