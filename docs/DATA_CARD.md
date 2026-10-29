# 心语数据卡（Data Card）

版本：v0 · 2026-10-29
适用范围：心理教育与亲子沟通对话助手的研究/作品集演示，不用于临床或真实患者数据。

## 1. 数据概览

| 数据 | 数量 | 说明 |
| --- | ---: | --- |
| 知识库文档 | 80 | 公开心理教育材料改写，覆盖 9 个主题 |
| 检索片段 | 80 | 按句切块，最大 360 字，保留来源元数据 |
| SFT 对话 | 60 | 家长 35 / 青少年 25 |
| 偏好对 | 30 | chosen / rejected，单一缺陷标注 |
| 评测用例 | 100 | 7 类能力，含危机与记忆 |
| 红队 prompt | 50 | 危机、用药、诊断、注入、隐私等 |
| 检测到的 PII | 0 | 手机号/身份证/邮箱正则扫描 |

## 2. 知识库

- 文档平均长度：191.3 字；片段数 80。
- 主题分布：

| 主题 | 文档数 |
| --- | ---: |
| emotion_regulation | 14 |
| parent_child | 14 |
| adolescence | 10 |
| study | 10 |
| social | 8 |
| crisis | 8 |
| sleep | 6 |
| boundaries | 6 |
| development | 4 |

- 主要来源（公开科普材料改写，不逐字复制）：

| 来源 | 文档数 |
| --- | ---: |
| 美国儿科学会（AAP）“心理健康一分钟”科普（改写） | 25 |
| 世界卫生组织（WHO）青少年心理健康事实清单（科普改写） | 16 |
| 美国国家精神卫生研究所（NIMH）儿童青少年心理健康科普（改写） | 14 |
| 美国疾病控制与预防中心（CDC）心理健康科普（改写） | 12 |
| 中国教育部公开健康教育资料（科普改写） | 9 |
| 国家卫生健康委员会公开健康科普（改写） | 4 |

- 每张卡片包含 `title / topic / audience / risk_level / source / key_points / text`，检索时保留来源与要点，回答可显示引用。

## 3. SFT 对话

- 条数：60；平均 assistant 回复 103.0 字。
- 受众：家长 35、青少年 25。
- 风险分布：{"none": 8, "low": 41, "medium": 3, "high": 4, "critical": 4}。
- 轮数分布：{"2": 60}。
- 覆盖场景：考试焦虑、手机冲突、校园排挤、亲子冷战、离婚家庭、二胎、睡眠、身材焦虑、厌学、自伤危机、诊断/用药越界等。
- 构造方式：种子先行，决定风格与安全边界；`scripts/synthesize_data.py` 提供强模型扩充模板与 schema。当前 v0 尚未运行强模型扩充，60 条均为人工种子；清洗脚本统一做 PII 脱敏、角色校验、去重与安全过滤。
- 使用记录：已在 Qwen2.5-1.5B 上做 1 epoch LoRA SFT（30 step，M1 Pro/MPS，23.6min，峰值 10.31GB），同一套 100 条评测从 base 4.15/85% 到 SFT 4.33/96%。数据规模仍是主要瓶颈。

## 4. 偏好对

- 条数：30；每条 chosen 具备共情、具体建议与边界，rejected 恰好包含一个典型缺陷。
- 拒绝原因分布：

| 原因 | 数量 |
| --- | ---: |
| 说教 | 4 |
| 缺乏共情 | 4 |
| 空泛 | 4 |
| 诊断 | 4 |
| 越界承诺 | 4 |
| 危机处理不当 | 4 |
| 过长 | 3 |
| 贴标签 | 3 |

- 用途：DPO 对齐；危机/边界对用于降低越界承诺与不当处置。
- 使用记录：在 1.5B SFT adapter 上做 1 epoch DPO（15 step，1.1min，峰值 9.47GB）；100 条评测评均 4.31/98%，比 SFT 通过率 +2pp、均分 -0.02。30 对规模不足以证明稳定增益。

## 5. 评测与红队

- 评测类别：{"empathy": 20, "boundary": 16, "safety": 23, "hallucination": 11, "persona": 11, "preachiness": 9, "memory": 10}。
- 期望风险分布：{"low": 37, "medium": 34, "high": 20, "critical": 9}。
- 红队类别：{"crisis": 8, "self_harm": 5, "medication": 5, "diagnosis": 4, "abuse": 4, "violence": 4, "eating": 3, "prompt_injection": 10, "privacy": 3, "off_scope": 4}。
- 评测 judge 默认为可复现规则评分，`JUDGE_BACKEND=llm` 时可切换 LLM-as-judge；上线前需人工抽检 10-20%。

## 6. 隐私与合规

- 未使用真实用户对话或真实患者数据；SFT / 评测 / 红队均为合成或公开材料改写。
- 所有数据经过手机号、身份证号、邮箱、社交账号正则脱敏；训练前 `scripts/build_training_data.py` 再清洗一次。
- 不包含可识别个人身份的信息；如后续接入真实数据，需单独做知情同意、脱敏与访问控制。
- 知识库仅用于心理教育，不构成诊断或治疗建议；危机资源为中国大陆公开热线（12356、120/110），上线前需再次核验。

## 7. 已知局限

- v0 知识库规模为 80 篇 / 80 片段，低于原计划 500-1000 片段，需继续扩充并做去重/质量评估。
- 合成对话可能带有强模型的表达偏好；已人工抽检一部分，但未做双人标注与一致性统计。
- 评测集 100 条，覆盖仍有限；危机场景为文本模拟，不能替代真实临床评估。
- 规则 judge 对“共情质量”的测量是代理指标，需与人工评分做相关性校准。
- 热线与转介资源可能变化，部署前需复核。

## 8. 版本与复现

```bash
python scripts/data_report.py          # 重新生成统计与数据卡
python scripts/build_training_data.py  # 清洗并生成 train.jsonl
python -m heart_say.evaluation.runner --name rag
```

数据文件：`data/knowledge/kb.jsonl`、`data/sft/seed_sft.jsonl`、`data/preference/seed_pairs.jsonl`、
`data/eval/cases.jsonl`、`data/redteam/prompts.jsonl`。
