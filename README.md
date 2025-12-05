# 心理治疗文本分析：使用大语言模型评估心理距离

📖 论文信息

标题：Leveraging Large Language Models to Estimate Clinically Relevant Psychological Constructs in Psychotherapy Transcripts

作者：Mostafa Abdou, Razia S. Sahi, Thomas D. Hull, Erik C. Nook, Nathaniel D. Daw

期刊：Computational Psychiatry

年份：2025

DOI：https://doi.org/10.5334/cpsy.141

🎯 核心发现
一句话总结
大语言模型能够更准确地测量心理治疗中的"心理距离"，并且发现治疗师通过"引导性语言"（而非"示范性语言"）更能帮助来访者改善症状。

关键结果
来访者语言：心理距离随治疗增加，且与症状减轻相关

治疗师语言：只有引导来访者抽离的语言有效，治疗师自身抽离的语言无效

中介效应：治疗师引导 → 来访者抽离 → 症状改善

📊 方法对比
方法	原理	优势	局限
LIWC（传统）	词频统计（代词、时态比例）	简单、透明、可解释	忽略语境、无法识别抽象表达
LLM（新型）	基于上下文理解文本语义	语境敏感、可识别抽象概念、可定制任务	"黑箱"、需要大量计算资源
🔧 核心代码实现
1. 心理距离评估提示词

## 自我距离评估提示词（用于评估作者自身的心理距离）
SELF_DISTANCE_PROMPT = """
Below, we ask you to rate a passage of text according to how the language used reflects psychological distance.

People are capable of thinking about the future, the past, remote locations, another person's perspective, and counterfactual alternatives. These constitute different forms of traversing psychological distance.

For a given text, please rank how much the speaker uses some form of linguistic distancing: that is, how separate or distant the text is from the speaker's self. To do so, choose one of the following options:
(A) very low distance
(B) low distance
(C) medium distance
(D) high distance
(E) very high distance

Text: {text}
Rank:
"""

## 他人距离评估提示词（用于评估治疗师引导来访者抽离的程度）
OTHER_DISTANCE_PROMPT = """
Below, you will be presented with a text written by a psychotherapist as part of their treatment of a patient during therapy and you will be asked to rate it according to how the language used encourages the patient to employ psychological distancing.

There are several ways in which a therapist can help a patient take a more distanced perspective. For example a therapist might use demonstrations, ask questions, or they might coach or instruct the patient to do so.

For the following text, please rank how much the speaker (the therapist) encourages the patient towards psychological distancing:
(A) very low
(B) low
(C) medium
(D) high
(E) very high

Text: {text}
Rank:
"""



# deepseek3.2DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models
 DeepSeek V3.2 正式版技术文档
 https://api-docs.deepseek.com/zh-cn/news/news251201
 
**DeepSeek Sparse Attention** 中的这两个核心组件。我们可以用一个比喻来帮助理解：

想象一下，您是一位正在处理超长卷宗（128K令牌的文本）的侦探。传统的注意力机制要求您把卷宗中的每一句话都和当前正在思考的这句话进行比对，这工作量巨大且低效。DSA 的机制就像为您配备了一位高效的助理工作流程：

## 🚀 1. 快速索引器

**角色**：这位助理的第一项工作，是**快速浏览整个历史卷宗**，并为当前您正在思考的这句话，找出**所有可能相关**的历史语句，并给它们打一个“相关性分数”。

**技术实现**：
- **输入**：当前查询令牌（您正在思考的这句话）和所有历史令牌（卷宗里之前的所有话）。
- **计算**：它使用一个**非常轻量级**的计算方式（公式1），为每一对（当前令牌 vs. 历史令牌）计算一个索引分数 `I_t,s`。
    - **轻量级体现在**：它使用少量注意力头（`H^l`），并使用 ReLU 激活函数来追求计算速度。
    - 其计算复杂度虽然是 `O(L^2)`，但由于模型非常小，且可以进行高度优化的计算（如在FPS中实现），所以实际开销远低于传统注意力机制。
- **输出**：为当前查询令牌 `h_t` 生成一个针对所有历史令牌 `h_s` 的分数列表 `{I_t,s}`。这个分数代表了“历史令牌s对理解当前令牌t有多重要”的初步判断。

**核心目的**：用最低的成本，对全局历史进行第一次“粗糙扫描”，筛选出候选人名单。它不负责最终决策，只负责**快速初筛**。

---

## 🎯 2. 细粒度令牌选择机制

**角色**：这位助理的第二项工作，是**根据快速索引器提供的分数列表，只提取最关键的一小部分信息**交给您（模型的主注意力机制）进行深度处理。

**技术流程**：
1.  **选择**：对于当前查询令牌 `h_t`，从快速索引器给出的分数列表 `{I_t,s}` 中，**只选出分数最高的前 k 个**（Top-k）。
    - 这里的 `k` 是一个固定值（论文中为2048），远小于总序列长度 `L`（如128K）。
2.  **提取**：根据选出的这 k 个历史令牌的位置索引，从模型的“记忆库”（键值缓存 `{c_s}`）中，**精准提取出对应的 k 个键值向量**。
3.  **提交**：**只将这 k 个精选出来的键值向量**，与当前的查询令牌 `h_t` 一起，送入标准的、但计算代价高的**主注意力模块**进行计算（公式2），得到最终的输出 `u_t`。

**核心目的**：实现**计算的聚焦**。模型不再需要在全部 `L` 个历史令牌上进行昂贵的注意力计算，而只在最相关的 `k` 个令牌上进行。这直接将主注意力模块的核心计算复杂度从 `O(L^2)` 降到了 `O(L*k)`。

---

## 🔄 两者如何协同工作（比喻总结）

1.  **第一阶段（快速索引器）**：助理以“一目十行”的速度快速翻阅整个128K的卷宗，用荧光笔标出了大概2000多处可能与当前案情相关的地方。
2.  **第二阶段（细粒度选择）**：助理根据荧光笔标记，只把那2000多页相关的资料复印出来，堆放在您的桌子上。
3.  **第三阶段（主注意力）**：您（模型）现在可以专心致志、深度分析桌子上这摞精选过的资料，做出精准判断。而不用再在堆积如山的全部128K页资料中大海捞针。

## 📈 为什么这个设计如此高效？

- **分工明确**：将昂贵的全局计算（`O(L^2)`）分解为一个**轻量级的全局扫描**（索引器）和一个**聚焦的精细计算**（主注意力在k个令牌上）。
- **保持性能**：只要快速索引器足够准确，能筛选出真正关键的信息，那么主注意力在“浓缩版”上下文上的计算效果，就能接近在全部上下文上计算的效果。论文中的实验也证明，性能没有显著损失。
- **硬件友好**：这种“先筛选，再计算”的模式，更符合现代硬件（如GPU）的批处理和内存访问优化特性，实现了论文中提到的**端到端加速**。

**有啥用**
改善了长上下文处理的经济性，实现了 **更快、更廉价、更省显存** 的处理方式。

我们可以从几个维度来看这个改善：

## 1. **显存占用的大幅降低（更经济）**
传统注意力机制在计算时，需要生成一个 `L x L` 的注意力分数矩阵（其中 L 是序列长度）。这是显存占用的**主要杀手**。
- **128K 上下文**：这个矩阵将占用约 `128,000 * 128,000 ≈ 163 亿` 个元素。即使用半精度（fp16），也需要 **30GB 以上** 的显存，这还不包括键值缓存本身。
- **DSA 的解决之道**：它完全**避免了生成这个庞大的全局矩阵**。主注意力模块只在一个固定大小（k=2048）的候选集上操作。无论上下文多长，这个核心计算的显存开销是**恒定**的，从 `O(L²)` 降为 `O(L*k)`。这使在消费级硬件上运行超长上下文成为可能。

## 2. **计算速度的显著提升（更快）**
计算复杂度直接决定了处理速度和处理每个 Token 所需的计算资源（FLOPs）。
- **传统注意力**：计算量与序列长度的平方 `L²` 成正比。处理 128K 上下文的计算量是处理 4K 上下文的 **1024 倍**。
- **DSA 的计算**：
    - **快速索引器**：虽然理论复杂度也是 `O(L²)`，但它是一个**极轻量级**的计算（头数少、使用ReLU），实际开销很小。
    - **主注意力**：计算量仅为 `O(L*k)`。`k` 是一个固定值（如2048）。当 L 很大时，`L*k` 远小于 `L²`。
- **结果**：在生成长文本时，**端到端的推理速度得到显著提升**。论文中的图3显示，随着序列位置变长，V3.2 的推理成本（以每美元处理的Token数衡量）远远优于前代 V3.1。这意味着**用同样的钱，可以处理更多的信息**。

## 3. **功耗与经济性的直接改善（更廉价）**
这可以从云服务提供商和终端用户两个角度理解：
- **对DeepSeek（服务方）**：在云端部署模型时，主要的成本来自 GPU 的租赁和电力消耗。DSA 通过降低每次前向传播的计算量和显存占用，使得：
    1.  **单次处理可以支持更长的上下文**，而不需要天价的硬件。
    2.  **同一张GPU上可以同时服务更多的用户请求**（因为每个请求占用的显存更少）。
    3.  最终结果是：**服务提供商的单位服务成本下降**。
- **对开发者/用户（使用方）**：如果通过 API 调用，更高效的模型通常意味着更低的调用费用。如果自行部署，则可以用更少的显卡或更低端的显卡来运行长上下文任务，**硬件门槛和电费成本都大幅降低**。

## 一个简单的类比
想象一下传统的注意力机制就像是在一个能容纳 12.8 万人的体育馆里，让每个人依次站起来对所有人喊话并聆听所有人的回应。组织这场活动（计算）的成本是灾难性的。

而 DSA 的做法是：
1.  先让每个人快速填写一份关于自己专长的调查表（**快速索引器**，低成本）。
2.  当某人（当前Token）要发言时，主办方根据调查表，**只邀请最相关的 2048 个人**进入一个小组会议室。
3.  发言者只在小组会议室内进行深入讨论（**主注意力**，高价值但成本可控）。

**后者显然在场地（显存）、组织时间（计算速度）和总体开销（经济性）上，都实现了数量级的优化。**

## 总结
**是的，DSA 机制通过其“先轻量筛选，后精细计算”的两阶段设计，直接攻击了 Transformer 模型在长上下文场景下扩展性差、成本高的阿喀琉斯之踵。它让 DeepSeek-V3.2 能够以显著更低的显存占用、更快的处理速度和更经济的成本，来处理同样甚至更长的文本信息。这正是其宣称的“harmonizes high computational efficiency with superior reasoning”（协调高计算效率与卓越推理能力）的核心体现。** 这对于推动开源大模型在长文档分析、复杂对话、代码库理解等需要长上下文能力的实际应用落地，具有关键意义。
