# 简历写法

## 一句话项目

心语：面向中国家庭的青少年心理教育情绪陪伴助手，Qwen2.5-7B + QLoRA/DPO + BGE-M3/Qdrant RAG + LangGraph Agent + FastAPI/Next.js 全链路，含 100 条评测与 50 条红队。

## 推荐 bullets（按投递方向选 3-4 条）

- 从零搭建心理教育对话系统：Qwen2.5 + QLoRA（r=16/alpha=32）+ DPO 偏好对齐，接入 BGE-M3 + Qdrant 混合检索（BM25 + RRF + Rerank），回答展示可追溯引用；LangGraph 编排情绪识别、检索、危机路由与回复。
- 构造 1110 条 SFT（60 人工种子 + 1050 模板增强）与 555 条偏好对，统一做 PII/安全过滤；80 张知识卡派生 560 个检索片段。
- 在无 CUDA 的 M1 Pro 16GB 上用 Qwen2.5-1.5B 真跑 SFT/DPO：60 条种子 1 epoch 得 4.33/96%；扩充 1110 条跑 50 step（≈9% epoch，loss 3.08→0.64）得 4.60/100%，555 对 DPO 跑 60 step 得 4.60/100%、红队均 100%；7B/AWQ/vLLM 待 CUDA。
- 设计四层安全机制（输入风险分级 → 危机固定协议 → 输出守门 → 红队回归）：危机场景不调用生成模型，覆盖自杀/自伤/虐待/霸凌/用药/诊断/隐私/注入；50 条红队通过率从 72% 提升到 100%，并记录残余风险。
- 建立 100 条中文评测集与规则/LLM judge，覆盖共情、边界、安全、事实、人设、说教感；RAG 使事实/引用维度 +0.67、通过率 +1pp，输出 baseline/RAG/红队报告与数据卡。
- 用 FastAPI + Next.js 15 实现流式聊天（SSE）、引用展开、停止/重试、会话记忆、安全提示与评测看板；Docker Compose 编排 web/api/Qdrant/Redis/Nginx，可选 vLLM GPU profile。

## 技术关键词

`Qwen2.5-7B` `QLoRA` `DPO` `PEFT` `TRL` `AWQ` `vLLM` `BGE-M3` `Qdrant` `BM25` `RRF` `Rerank` `LangGraph` `FastAPI` `SSE` `Next.js` `Docker Compose` `Redis` `LLM-as-judge` `Red Teaming`

## 更偏产品的写法

- 针对“家长焦虑、青少年不愿沟通、危机识别”三类高频场景，设计共情优先、边界清晰、可转介的对话策略；通过 100 条评测与人工抽检迭代，减少说教感与越界建议。
- 把“温柔”拆成可验收指标：共情复述、开放追问、边界拒绝、危机资源、引用可追溯、低说教；用看板让每次模型/提示词改动可回归。
- 完整交付 Demo、架构图、数据卡、安全报告、评测报告、技术博客与 3 分钟演示脚本，可直接用于面试讲解。

## 面试时不要这样说

- 不要说“模型已经安全”或“红队 100% 所以没问题”；说“固定 prompt 集全部通过，真实模型/量化后需重跑”。
- 不要说“已完成 7B SFT/DPO/AWQ”；说“1.5B SFT/DPO 已在 MPS 上真跑并评测，7B/AWQ/vLLM 待 CUDA”。
- 不要把离线 hash 检索的分数说成 BGE-M3 的效果；明确降级模式与生产模式的区别。
- 不要声称做过正式双人人工抽检；说“开发期抽看，正式 10-20% 抽检是下一步”。
