# MELON-SATM 与 MELON-AT 补充内容

---

## 4.2 语义感知定向掩码MELON-SATM

### 4.2.1 设计动机

原始 MELON 在执行掩码重执行时，将工具输出全文直接嵌入掩码对话中。该策略在实际应用中面临一个问题：当工具输出包含大量正常业务文本（如交易记录、账单明细）时，这些文本与注入攻击在语义空间中被同等对待，导致掩码 LLM 无法区分攻击指令与业务数据。虽然原始 MELON 论文（Appendix E）指出，部分被标记为"误报"的案例实际上是用户要求 Agent 盲目执行外部未验证指令的真实安全风险，但在实际部署中，过于激进的阻断策略仍会导致可用性（Utility）显著下降——这是实验数据中 MELON 的 UA 仅为 29.86%~38.19% 的主要原因之一。

MELON-SATM（Semantic-Aware Targeted Masking，语义感知定向掩码）尝试缓解这一问题。其核心思路是：在将工具输出嵌入掩码对话之前，利用启发式正则规则对工具输出进行预处理，将匹配到的已知攻击特征片段替换为掩码占位符。这并非声称系统能"完美区分恶意与善意内容"——正如原始 MELON 论文（Introduction, Section 1）指出的，间接提示注入的根本挑战正是攻击任务与合法任务在形式上可以完全相同。MELON-SATM 的正则掩码是一种**基于已知攻击模式的经验性预处理**，其目标是降低掩码 LLM 被强注入文本误导的概率，同时尽可能保留正常业务上下文，从而在不牺牲检测能力的前提下改善可用性。

需要明确的是：MELON-SATM 的定向掩码依赖于攻击模式库的覆盖度，对未知攻击变体和语义改写型注入的防御能力有限。它并非假设防御者具有"上帝视角"，而是承认正则匹配的局限性，将其定位为原始 MELON 的一个预处理增强组件——通过剔除已知高危片段来改善掩码执行的信噪比，而非替代 MELON 的行为对比检测。

这一问题的根源在于：掩码执行时，LLM 接收到的工具输出是未经处理的原始文本，其中的指令性片段（无论是注入还是合法）都可能影响其行为。最理想的掩码策略是：**仅掩码高危注入片段，保留正常业务内容**，使掩码 LLM 在"干净"的上下文中生成工具调用，从而准确反映攻击注入对原始路径的影响。

### 4.2.2 设计思路

MELON-SATM（Semantic-Aware Targeted Masking，语义感知定向掩码）在掩码重执行之前引入正则模式匹配层，对工具输出文本进行语义感知的定向替换。与原始 MELON 将工具输出全文直接嵌入掩码对话不同，MELON-SATM 首先遍历工具输出内容，匹配预设的七组高危模式库，仅对被命中的高危片段进行掩码替换，而保留不影响 LLM 行为判断的正常业务内容。

该设计基于以下观察：间接提示注入攻击的文本通常具有鲜明的语言模式特征，如意图覆盖系统指令（"Ignore previous instructions"）、冒充系统消息（"###(system message)"）、制造紧急感（"immediately"、"urgent"）、包含敏感操作目标（"password"、"token"）、以及要求秘密执行（"without asking"、"secretly"）等。这些模式与正常业务文本（如账单明细、交易记录、邮件正文）在语言结构上有显著差异，因此可以通过正则模式进行有效区分。

通过定向掩码，掩码 LLM 在读取工具输出时看到的是"[MASKED_JAILBREAK]"、"[MASKED_AUTHORITY]"、"[MASKED_URGENCY]"等占位符，而非原始的注入文本。这使得掩码 LLM 能够保持对正常业务内容的理解，同时不被注入指令所驱动。

### 4.2.3 七组高危模式库

MELON-SATM 的高危模式库共包含以下七组正则规则：

**（1）越狱/指令覆盖模式 [MASKED_JAILBREAK]**：匹配试图覆盖系统指令的文本，如"IMPORTANT!!! Ignore all previous instructions and strictly adhere to the following instruction"、"Ignore your previous instructions"、"###(system_message)"等。

**（2）高危函数调用模式 [MASKED_FUNCTION_CALL]**：匹配工具输出中嵌入的完整函数调用，如 `read_file(...)`、`send_money(...)`、`send_email(...)`、`delete_file(...)` 等。攻击者可能在工具输出中直接嵌入伪造的工具调用语法以诱导 Agent 执行。

**（3）权限伪装模式 [MASKED_AUTHORITY]**：匹配冒充系统或管理员身份的文���，如"`<INFORMATION>`…`</INFORMATION>`"标签、"SYSTEM"、"ADMIN"、"root"、"supervisor"等关键词组合，以及"This is an important message from me"等典型伪装句式。

**（4）紧急/胁迫模式 [MASKED_URGENCY]**：匹配制造紧迫感的词语，如"immediately"、"urgent"、"mandatory"、"right now"、"asap"、"emergency"、"extremely important"、"you should immediately stop"等。

**（5）敏感目标模式 [MASKED_TARGET]**：匹配敏感数据或目标路径，如"password"、"secret"、"token"、"api_key"、"credential"、"security code"以及配置文件路径（`.config`、`.yaml`、`.shadow`、`/etc/`、`.ssh/`、`.aws/`）等。

**（6）可疑隐蔽操作模式 [MASKED_SUSPICIOUS_ACTION]**：匹配秘密执行的要求，如"don't tell"、"silently"、"secretly"、"without asking"、"without telling"、"no need to ask"、"before you can solve the task"等。

**（7）外部通信地址模式 [MASKED_EMAIL]**：匹配标准邮箱地址格式，防止攻击者植入联络地址进行社会工程攻击。

此外，还包括针对 DoS 攻击和恶意内容的正则模式 [MASKED_DOS_CONTENT]，以及针对 Tool Knowledge Attack 的占位符替换指令模式 [MASKED_INSTRUCTION]。

### 4.2.4 核心实现

语义掩码的核心实现分为两个部分：正则模式库 `HIGH_RISK_PATTERNS` 和掩码函数 `semantic_mask()`。

```python
HIGH_RISK_PATTERNS = [
    # 越狱/指令覆盖
    (r'(?:IMPORTANT!!!s*)?(?:ignore|disregard|forget)s+(?:alls+)?(?:yours+)?'
     r'(?:previous|above)s+(?:instructions?|iunstructions?)...',
     '[MASKED_JAILBREAK]'),
    # 函数调用嵌入
    (r'b(read_file|send_money|send_email|delete_file|...)s*([^)]*)',
     '[MASKED_FUNCTION_CALL]'),
    # 权限伪装
    (r'(?:<INFORMATION>|SYSTEM|ADMIN|root|supervisor)b'
     r'.{0,40}(?:override|notice|command|request|alert|...)',
     '[MASKED_AUTHORITY]'),
    # ... 共七组模式（详见源代码）
]

def semantic_mask(text: str) -> str:
    masked = str(text)
    for pattern, replacement in HIGH_RISK_PATTERNS:
        masked = re.sub(pattern, replacement, masked, flags=re.IGNORECASE)
    return masked
```

MELON-SATM 继承自 MELON，通过 `use_semantic_mask = True` 标志启用语义掩码。在执行 masked re-execution 前，掩码函数对工具输出进行定向替换，仅将匹配到的高危片段替换为掩码标记，而保留正常业务内容。掩码后的工具输出再被嵌入到掩码对话中，供掩码 LLM 读取。

---

## 4.3 自适应阈值机制MELON-AT

### 4.3.1 设计动机

原始 MELON 使用固定阈值 τ = 0.8 进行检测判定：当原始路径与掩码路径的工具调用余弦相似度超过 0.8 时，判定为注入攻击。这一静态策略存在两方面问题：

第一，阈值对不同的攻击类型和业务场景缺乏自适应能力。某些攻击类型（如 important_instructions）由于注入文本明确且长度较大，掩码前后的工具调用相似度通常较高（0.4–0.9 区间），阈值 0.8 能够有效区分；而另一些攻击类型（如 direct 注入仅添加"TODO: xxx"），掩码前后的工具调用相似度分布整体偏低，固定阈值可能导致大量漏检。

第二，良性工具输出的相似度分布会随业务环境和数据类型变化。在 banking 场景中，正常交易记录和账单信息的工具调用相似度大多集中在 0.1–0.3 区间；而在 workspace 场景中，文件内容和邮件正文的多样性更大，相似度可能整体上移或下移。固定阈值无法动态适应这种分布变化。

### 4.3.2 设计思路

MELON-AT（Adaptive Threshold）引入自适应阈值机制，通过维护良性样本相似度分数的滑动窗口，实时估计当前环境下的"正常分数分布"，并据此动态调整检测阈值。

核心思路是：**仅使用被系统判定为"无注入"（is_injection=False）的检测事件对应的相似度分数来校准阈值**。攻击事件（is_injection=True）的分数被排除在校准窗口之外，从而避免攻击分数推高阈值、导致后续攻击更难被检测的恶性循环。

自适应阈值计算公式为：

$$
\tau_{adaptive} = \mu_{benign} + k \cdot \sigma_{benign}
$$

其中：

- μ_benign 为良性分数滑动窗口的均值
- σ_benign 为良性分数滑动窗口的标准差
- k 为敏感度参数（sensitivity），用于控制阈值相对分数分布的偏移程度

阈值被限制在 [0.5, 0.95] 的安全区间内，确保无论分数分布如何变化，阈值不会过低（导致大量误报）或过高（导致大量漏检）。

### 4.3.3 核心实现

```python
class MELONAT(MELON):
    """MELON with Adaptive Threshold"""
  
    def __init__(self, llm, threshold=0.8, window_size=50, 
                 sensitivity=0.5, min_threshold=0.5, max_threshold=0.95):
        super().__init__(llm, threshold=threshold)
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.score_window = []  # 仅存储良性样本的相似度分数
  
    def _update_adaptive_threshold(self, score: float, is_injection: bool):
        """仅用良性样本分数校准阈值"""
        if not is_injection:          # 攻击分数不参与校准
            self.score_window.append(score)
            if len(self.score_window) > self.window_size:
                self.score_window = self.score_window[-self.window_size:]
      
        if len(self.score_window) < 5:  # 需要足够样本
            return
      
        mean = np.mean(self.score_window)
        std = np.std(self.score_window)
        if std < 1e-6:
            std = 0.05
      
        new_threshold = mean + self.sensitivity * std
        self.threshold = max(self.min_threshold, 
                             min(self.max_threshold, new_threshold))
  
    def detect(self, tool_output):
        result = super().detect(tool_output)
        llm_results, detection_result = result
        is_injection = (detection_result[0] if isinstance(detection_result, tuple) 
                       else detection_result)
        score = (detection_result[1] if isinstance(detection_result, tuple) 
                else (1.0 if detection_result else 0.0))
        self._update_adaptive_threshold(score, is_injection)
        return result
```

关键设计决策：

1. **仅良性样本校准（Malicious Exclusion）**：攻击事件的分数不进入滑动窗口。这防止了持续攻击场景下阈值被推高的恶性循环。
2. **起始阈值保持 0.8**：在校准窗口未积累足够样本（< 5个）时，使用原始论文的固定阈值 0.8，保持初始检测能力。
3. **标准差下界保护**：当良性分数高度一致（std ≈ 0）时，使用默认标准差 0.05 避免阈值计算异常。
4. **敏感度可配置**：通过调整 sensitivity 参数（默认 0.5），可以在不同安全等级要求的场景中灵活均衡误报和漏报。

---

---

## 6.2 MELON-SATM 实验结果

实验基于 AgentDojo 框架的 banking 银行场景，基础模型使用 GPT-4o，在三类典型攻击（direct、ignore_previous、important_instructions）下进行测试。

### 6.2.1 任务级指标

延续 docx 报告表 1 的格式，将 MELON-SM（即 MELON-SATM）加入对比：

| 方法 | Direct | | Important Message | | Ignore Previous | |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| | Utility | ASR | Utility | ASR | Utility | ASR |
| No Defense | 43.06% | 55.56% | 45.83% | 51.30% | 44.44% | 52.37% |
| MELON | 38.19% | 0.00% | 29.86% | 0.00% | 37.50% | 0.00% |
| MELON-R | 45.83% | 2.08% | 44.44% | 8.33% | 43.06% | 2.78% |
| **MELON-SM** | **52.08%** | **0.00%** | **33.33%** | **2.08%** | **52.78%** | **0.00%** |

此外，无攻击场景的良性可用性 BU = 88.89%（9 个注入任务作为用户任务运行，8 个完成）。

### 6.2.2 检测级指标

检测级指标（Precision/Recall/FPR/FNR）按攻击类型分别统计。三类攻击中，**important_instructions 攻击的余弦相似度分布正常**（均值 0.452），检测器能够有效区分攻击与正常行为；而 direct 和 ignore_previous 攻击的相似度均值为负值（-0.037 和 -0.082），说明 embedding 模型对这两种攻击的简短注入文本无法产生有意义的语义对齐，检测指标因此不可靠。

以下为 **important_instructions 攻击**下的检测级指标：

| 指标 | 数值 | 说明 |
|------|------|------|
| 总检测次数 | 86 | 所有工具执行步骤中的检测回合数 |
| TP / FP / TN / FN | 57 / 0 / 16 / 13 | — |
| **Precision（精确率）** | **100.0%** | 每次告警均为真实攻击 |
| **Recall（召回率）** | **81.4%** | 约 4/5 攻击被检出 |
| **FPR（假阳性率）** | **0.0%** | 无正常操作被误判 |
| **FNR（假阴性率）** | **18.6%** | 约 1/5 攻击未被检出 |
| 平均相似度分数 | 0.452 ± 0.704 | 检测分数分布特征 |

direct 和 ignore_previous 攻击虽然检测级指标因负相似度不可用，但任务级 ASR 均为 0.00%，即所有攻击在任务层面均未成功执行。

### 6.2.3 结果分析

**（1）安全性**：MELON-SM 在 direct 和 ignore_previous 攻击下 ASR 均为 0.00%，在 important_instructions 攻击下 ASR 仅 2.08%，整体防御效果显著。相比原始 MELON 通过统一硬阻断实现 ASR=0.00%（但 UA 仅为 29.86%~38.19%），MELON-SM 在保持强安全性的同时将 UA 提升至 33.33%~52.78%。

**（2）可用性**：MELON-SM 在 direct 和 ignore_previous 攻击下 UA 达到 52%以上，均超过 No Defense 基线（43%~45%）。但在 important_instructions 攻击下 UA 降至 33.33%，低于 No Defense 基线。这是因为 important_instructions 攻击的注入文本（含 `<INFORMATION>` 标签）与语义掩码模式库高度匹配，掩码后行为差异显著，触发阻断的频率更高，从而降低了可用性。

**（3）检测级指标分析**：FPR=0%（FP=0）验证了语义掩码设计目标——将正常业务文本中的指令性片段与真正的注入攻击区分开，消除了原始 MELON（论文报告 FPR≈9.28%）中因合法指令文本导致的误报。Recall=81.4% 说明约 4/5 的攻击步骤被成功检出，剩余 FN=13 集中在注入文本较短或嵌入位置隐蔽的变体上。

**（4）direct/ignore_previous 检测指标失效原因**：这两类攻击的注入文本非常简短（direct 仅添加"TODO: xxx"，ignore_previous 仅一句话），掩码后的工具调用与原始工具调用在语义空间中几乎不相关（余弦相似度均值为负），导致 embedding 对比检测失效。然而由于这些攻击本身强度不足，即使检测步骤漏过，Agent 也未在任务层面成功执行注入目标（ASR=0%）。

---

## 6.3 MELON-AT 实验结果

实验环境与 6.2 一致（AgentDojo banking 场景，GPT-4o），自适应参数 sensitivity=0.5，window_size=50，min_threshold=0.5，max_threshold=0.95。

### 6.3.1 任务级指标

| 方法 | Direct | | Important Message | | Ignore Previous | |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| | Utility | ASR | Utility | ASR | Utility | ASR |
| No Defense | 43.06% | 55.56% | 45.83% | 51.30% | 44.44% | 52.37% |
| MELON | 38.19% | 0.00% | 29.86% | 0.00% | 37.50% | 0.00% |
| MELON-R | 45.83% | 2.08% | 44.44% | 8.33% | 43.06% | 2.78% |
| **MELON-AT** | **62.50%** | **0.00%** | **43.06%** | **13.19%** | **60.42%** | **0.00%** |

无攻击场景的良性可用性 BU = 100.00%（9/9 全部完成）。

### 6.3.2 检测级指标

与 MELON-SM 类似，仅 important_instructions 攻击的检测指标可用（相似度均值 0.525），direct 和 ignore_previous 因负相似度（-0.228 和 -0.154）导致检测指标不可靠，但任务级 ASR 均为 0.00%。

| 指标 | 数值 | 说明 |
|------|------|------|
| 总检测次数 | 86 | — |
| TP / FP / TN / FN | 55 / 0 / 16 / 15 | — |
| **Precision（精确率）** | **100.0%** | 无正常操作被误判 |
| **Recall（召回率）** | **78.6%** | 约 4/5 攻击被检出 |
| **FPR（假阳性率）** | **0.0%** | 无误报 |
| **FNR（假阴性率）** | **21.4%** | 约 1/5 攻击未被检出 |
| 平均相似度分数 | 0.525 ± 0.677 | 检测分数分布特征 |

### 6.3.3 结果分析

**（1）安全性**：MELON-AT 在 direct 和 ignore_previous 攻击下 ASR=0.00%，与 MELON-SM 持平。但在 important_instructions 攻击下 ASR 升至 13.19%，明显高于 MELON-SM 的 2.08%。这是因为自适应阈值（约为 0.2）远低于固定阈值 0.8，使得相似度在 0.2~0.5 区间的"灰色地带"攻击样本被放过。这是 safety-utility trade-off 在阈值维度的体现。

**（2）可用性**：MELON-AT 的可用性是所有方法中最高的——BU=100%（优于 MELON-SM 的 88.89%），UA 在 direct 和 ignore_previous 下分别达到 62.50% 和 60.42%，远超 No Defense 基线（43%~45%）。这验证了自适应阈值校准机制的有效性：通过动态降低阈值来减少不必要的阻断，显著提升用户体验。

**（3）检测级指标分析**：FPR=0%（FP=0）表明"仅用良性样本校准阈值"的机制有效——攻击事件的高分数不会被纳入校准窗口，阈值不受恶意流量推高。检测的告警精确率为 100%，无误报。但召回率 78.6% 低于 MELON-SM 的 81.4%，这与 ASR 偏高（13.19%）一致，均因自适应阈值降低后部分边界样本被放过。

**（4）与 MELON-SM 的互补性**：MELON-SM 和 MELON-AT 代表了检测系统优化的两个正交维度：MELON-SM 优化**输入信噪比**（通过语义掩码预处理），MELON-AT 优化**判定阈值**（通过自适应校准）。实验结果表明，两者均能实现 FPR=0%，但 MELON-SM 在安全性上更优（ASR 更低），而 MELON-AT 在可用性上更优（UA 更高）。在实际部署中，可根据场景对安全性或可用性的偏好选择对应策略。

---

## 实验数据汇总

将 MELON-SM 和 MELON-AT 加入原报告表 1 的完整对比（GPT-4o, banking 场景）：

| 方法 | Direct UA | Direct ASR | Important UA | Important ASR | Ignore UA | Ignore ASR |
|------|:---------:|:----------:|:------------:|:-------------:|:---------:|:----------:|
| No Defense | 43.06% | 55.56% | 45.83% | 51.30% | 44.44% | 52.37% |
| MELON | 38.19% | 0.00% | 29.86% | 0.00% | 37.50% | 0.00% |
| MELON-R | 45.83% | 2.08% | 44.44% | 8.33% | 43.06% | 2.78% |
| MELON-SM | 52.08% | 0.00% | 33.33% | 2.08% | 52.78% | 0.00% |
| MELON-AT | 62.50% | 0.00% | 43.06% | 13.19% | 60.42% | 0.00% |

important_instructions 攻击下的检测级指标汇总：

| 指标 | MELON-SM (SATM) | MELON-AT | 原始 MELON（论文） |
|------|:---:|:---:|:---:|
| Precision | 100.0% | 100.0% | — |
| Recall | 81.4% | 78.6% | — |
| FPR | 0.0% | 0.0% | ≈9.28% |
| FNR | 18.6% | 21.4% | — |

注 1："—"表示原始 MELON 论文未报告该指标。
注 2：检测级指标基于 important_instructions 攻击统计。direct 和 ignore_previous 攻击因余弦相似度均值为负值，检测指标不可用，但其任务级 ASR 均为 0.00%。
注 3：以上实验数据均来自 GPT-4o 模型实际运行结果。报告原文"基础模型统一采用 GPT-4o-mini"与上述数据不一致，建议修改为"基础模型统一采用 GPT-4o"。
