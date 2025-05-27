# 如何写好prompt

1.  **结构化设计 (Structured Design):**
    * **使用模板 (Templates):** LangGPT 提倡使用模板来组织 Prompt。这就像填写表单一样，而不是每次都从头开始写。结构化的 Prompt 更容易让大模型理解和遵循。
    * **模块化 (Modules):** 将 Prompt 分解为不同的模块（如角色定义、目标、约束、工作流程等）。这使得 Prompt 更清晰，易于管理和修改。
        * **固有模块 (Inherent Modules):** 包括一些通用模块，如：
            * `Profile` (角色/身份): 定义模型扮演的角色或身份。
            * `Goal` (目标):明确模型需要达成的任务目标。
            * `Constraint` (约束): 设定模型输出的边界和限制。
            * `Workflow` (工作流程): 指导模型完成任务的步骤（类似思维链）。
        * **扩展模块 (Extension Modules):** 用户可以根据特定需求自定义模块。
    * **利用 Markdown 等标记语言:** LangGPT 建议使用 Markdown 的层级结构（如标题、列表）来组织 Prompt 内容，这有助于模型更好地“看懂”和解析你的意图。JSON、YAML 等格式也可以使用。

2.  **清晰明确 (Clarity and Precision):**
    * **明确的角色设定 (Define Roles Clearly):** 赋予模型一个具体的角色（例如“你是一位资深科研论文审稿人”）。详细描述角色的技能、行为方式等，能让模型表现得更符合预期。
    * **具体的目标指令 (Specific Goal Instructions):** 任务目标要清晰、具体，避免模糊不清的描述。
    * **精确的约束条件 (Precise Constraints):** 如果对输出格式、长度、风格等有要求，需要明确指出。

3.  **可复用性与灵活性 (Reusability and Flexibility):**
    * **使用变量 (Use Variables):** LangGPT 引入了变量的概念（例如用 `<>` 包裹）。通过变量，可以轻松地引用、设置和更改 Prompt 中的内容，提高了 Prompt 的灵活性和可复用性，方便针对不同场景进行微调。
    * **易于迭代更新 (Facilitate Iterative Updates):** 结构化的设计使得修改和优化 Prompt 更加容易，而不需要从头重写。

4.  **引导思考与输出 (Guiding Thought and Output):**
    * **思维链 (Chain-of-Thought / Workflow):** 引导模型分步骤思考，可以显著提升复杂任务的表现。在 Prompt 中明确指出思考的路径或工作流程。
    * **提供示例 (Provide Examples / Few-shot Learning):** 给出清晰的输入输出示例，能帮助模型更好地理解任务要求和期望的输出格式。
    * **定义输出格式 (Define Output Format):** 明确要求模型输出的格式，例如使用特定的标签包裹答案、JSON 格式等。

5.  **迭代与优化 (Iteration and Optimization):**
    * **测试与调整 (Test and Tweak):** 编写完 Prompt 后，进行测试，观察模型的输出，并根据结果进行调整优化。LangGPT 的结构化方法使得这种调整更加方便。
    * **从错误中学习 (Learn from Errors):** 分析模型输出不佳的原因，并据此改进 Prompt。
