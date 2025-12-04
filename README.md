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
python
# 自我距离评估提示词（用于评估作者自身的心理距离）
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

# 他人距离评估提示词（用于评估治疗师引导来访者抽离的程度）
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
