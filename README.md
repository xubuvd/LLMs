# ChatGLM-6B 全套微调代码，大规模数据上进行并行测试，发现许多bug，单机多卡负载不均衡，有些卡爆掉，修改代码bug，调出最佳效果后，再上传，请等待...

# 国内大语言模型ChatGPT专区，欢迎交流
# Open-source of LLMs 

 If you like the project, please show your support by leaving a star ⭐.

 No. |Projects | URL  | Comments
 --------| :-----------:  |:-----------:  | :-----------:|
 1|LLaMA |  https://github.com/facebookresearch/llama | 
 2|OpenChatKit|https://github.com/togethercomputer/OpenChatKit | 基于GPT-NeoX-20B的微调版本，200亿参数，48层，单机八卡，每卡六层网络，每一层的模型结构：![OpenChatKit](https://user-images.githubusercontent.com/59753505/228441689-16a55551-0b0c-4c59-9c1f-0206ec9f4069.jpg)
 3|Open-Assistant | https://github.com/LAION-AI/Open-Assistant |12B或者LLAMA-7B两个版本，Open Assistant 全流程训练细节（GPT3+RL）,https://zhuanlan.zhihu.com/p/609003237
 4|ChatGLM-6B | https://github.com/THUDM/ChatGLM-6B | ChatGLM-6B 是一个开源的、支持中英双语的对话语言模型，基于 General Language Model (GLM) 架构，具有 62 亿参数。可以作为很好的基础模型，在此之上做二次研发，在特定垂直领域。没有放出源代码，只有训练好的模型。
5|GLM-130B | https://github.com/THUDM/GLM-130B/ | 1300亿参数的中/英文大模型，没有放出源代码，只有训练好的模型
6| Alpaca 7B | https://crfm.stanford.edu/2023/03/13/alpaca.html |  A Strong Open-Source Instruction-Following Model，a model fine-tuned from the LLaMA 7B model on 52K instruction-following demonstrations. 
7| Claude | 用户可以通过邮箱等信息注册申请试用, 产品地址：https://www.anthropic.com/product, 申请地址：https://www.anthropic.com/earlyaccess, API说明: https://console.anthropic.com/docs/api |两个版本的 Claude：Claude 和 Claude Instant。 Claude 是最先进的高性能模型，而 Claude Instant 是更轻、更便宜、更快的选择。
8|LLama/ChatLLama|https://github.com/nebuly-ai/nebullvm/tree/main/apps/accelerate/chatllama | 中文支持不好，有全套的SFT，RLHF训练过程
9|chatglm-6B_finetuning | https://github.com/ssbuild/chatglm_finetuning | 1,chatGLM-6B的微调版本，正在补充RLHF代码，陆续放出来；28层网络，每一层的模型结构：![chatglm](https://user-images.githubusercontent.com/59753505/228441877-63aae805-b862-4c42-839e-c60ef9e2d135.jpg);<br>2，SFT阶段可以跑通，分别微调了1，2，3和4层，指令数据55K，单卡大约4个小时一个epoch；<br>3, 借鉴 Colossal-AI/Open-Assistant的强化学习代码（PPO，PPO-ptx算法），Colossal-AI可以迁移过来，被实践过。<br>4，Reward model，可选较多，直接基于GLM-6B模型微调一个Reward model。<br>难点就是训练数据；GPT3.5使用了33K的人工标注数据训练 Reword model。<br>每个问题，配置四个答案ABCD，人工从好到差排序比如B>A>D>C，排序后的数据微调Reward model。<br>单机8卡，A100， 80G，train_batch_size=4, max_seq_len设置成512，才可以跑50K级别的微调数据集，这份代码感觉有点疑问，需要优化的地方挺多的
10|ChatGLM-Tuning| https://github.com/mymusise/ChatGLM-Tuning| ChatGLM-6B的又一个微调版本
11|中文语言模型骆驼 (Luotuo)|https://github.com/LC1332/Chinese-alpaca-lora |基于 LLaMA、Stanford Alpaca、Alpaca LoRA、Japanese-Alpaca-LoRA 等完成，单卡就能完成训练部署
12|Alpaca-COT数据集 | https://github.com/PhoebusSi/Alpaca-CoT | 思维链（CoT）数据集，增强大语言模型的推理能力
13|Bloom | https://huggingface.co/bigscience/bloom | 训练和代码比较全
14|中文LLaMA&Alpaca大语言模型 | https://github.com/ymcui/Chinese-LLaMA-Alpaca | 在原版LLaMA的基础上扩充了中文词表并使用了中文数据进行二次预训练，进一步提升了中文基础语义理解能力。同时，在中文LLaMA的基础上，本项目使用了中文指令数据进行指令精调，显著提升了模型对指令的理解和执行能力。
15 |Colossal-AI/ColossalChat | https://github.com/hpcaitech/ColossalAI | 训练和代码比较全，包括 RLHF 训练代码；以 LLaMA 为基础预训练模型；开源了7B和13B两种模型；
16 | Cerebras-GPT七个版本 | 官网地址：https://www.cerebras.net/blog/cerebras-gpt-a-family-of-open-compute-efficient-large-language-models<br>GPT地址:https://www.cerebras.net/cerebras-gpt<br>Hugging Face地址:https://huggingface.co/cerebras | 七个参数版本：1.16亿、2.56亿、5.9亿、13亿、27亿、67亿和130亿参数, 基于GPT的生成人工智能大型语言模型
17 | BloombergGPT<br>(金融领域) | https://arxiv.org/abs/2303.17564 | BloombergGPT的训练数据库名为FINPILE，构建迄今为止最大的特定领域数据集, 由一系列英文金融信息组成，包括新闻、文件、新闻稿、网络爬取的金融文件以及提取到的社交媒体消息。训练专门用于金融领域的LLM，拥有500亿参数的语言模型。
18 | dolly-v1-6b | https://github.com/databrickslabs/dolly | 1, fine-tuned on a ~52K instruction (Self-Instruct从 ChatGPT自动获取)；<br>2，deepspeed ZeRo 3加速训练;<br>3.可借鉴的：deepspeed ZeRo 3加速训练部分；
19 | ChatDoctor | https://github.com/Kent0n-Li/ChatDoctor | 医疗领域对话模型，基于LLaMA-7B微调的大模型，经过四轮微调：<br>第一轮微调：羊驼的52K instruction-following 数据<br>;第二轮微调：患者和医生之间的5K对话数据集（ChatGPT GenMedGPT-5k和疾病数据库生成）；<br>第三轮微调：患者和医生之间的真实对话（HealthCareMagic-200k）；<br>第四轮微调：患者和医生之间的真实对话（icliniq-26k）.
20 | 开源中文对话大模型BELLE | https://github.com/LianjiaTech/BELLE | BELLE-7B（基于 BLOOMZ-7B1-mt 微调）<br>BELLE-13B的感觉还行
21 |  InstructGLM | https://github.com/yanqiangmiffy/InstructGLM | 基于ChatGLM-6B+LoRA在指令数据集上进行微调；截止4月4号下午，InstructGLM存在以下缺点：多卡不支持，原作者在回答issues时也确认了；社区不活跃，两周不更新代码，坑反馈的太少（才11条），deepspeed没有；这是我用三块卡跑，卡负载不均衡：![image](https://user-images.githubusercontent.com/59753505/229748872-df8f3909-f8e5-454c-8378-56766f8aa1a2.png)
22 | Cerebras-GPT  | https://huggingface.co/cerebras/Cerebras-GPT-13B | 参数量级130亿，大小比肩最近Meta开放的LLaMA-13B，数据集、模型权重和计算优化训练，全部开源。可商用！
23 | Baize<br>(加利福尼亚大学, 基于 LLaMA 的微调)|https://github.com/project-baize/baize-chatbot | 数据集生成: 让 ChatGPT 与自己进行对话，模拟用户和AI机器人的回复」。这个生成的语料数据集是在多轮对话的背景下训练和评估聊天模型的宝贵资源。此外，通过指定种子数据集，可以从特定领域进行采样，并微调聊天模型以专门针对特定领域，例如医疗保健或金融。<br>Parameter-efficient tuning， 输入序列的最大长度设置为512，LoRA中的秩k设置为8，使用8位整数格式 (int8) ，Adam 优化器」更新LoRA 参数，batch size为64，learning rate为2e-4、1e-4和 5e-5，可训练的LoRA参数在 NVIDIA A100-80GB GPU 上微调了1个 epoch。





# Reward打分模型，用于强化学习RLHF阶段
https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2<br>
OpenAssistant和LLaMA模型使用的打分模型


# Prompt数据集收集
1，人工标注一批；<br>
2, 从人工标注的选择200个作为种子，调用ChatGPT获取新的prompt数据，筛选一批；<br>
3, prompt总量在50K量级，可以满足RLHF阶段的微调了。<br>

# chatglm-6B_finetuning的源代码解析

模型，一层transformer_block，总共 28 层:<br>
ModuleList(<br>
  (0): GLMBlock(<br>
    (input_layernorm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)<br>
    (attention): SelfAttention(<br>
      (rotary_emb): RotaryEmbedding()<br>
      (query_key_value): Linear(in_features=4096, out_features=12288, bias=True)<br>
      (dense): Linear(in_features=4096, out_features=4096, bias=True)<br>
    )<br>
    (post_attention_layernorm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)<br>
    (mlp): GLU(<br>
      (dense_h_to_4h): Linear(in_features=4096, out_features=16384, bias=True)<br>
      (dense_4h_to_h): Linear(in_features=16384, out_features=4096, bias=True)<br>
    )<br>
  )<br>
)<br>

# 北京邮电大学 王小捷教授 ChatGPT 讲座分享

https://www.bilibili.com/video/BV1G24y187yx/?buvid=ZB476BB0B8710E3C4F548C7C2778AA1427C6&is_story_h5=false&mid=AdBmq4Rn7y73B2EmgVj16A%3D%3D&p=1&plat_id=114&share_from=ugc&share_medium=iphone&share_plat=ios&share_session_id=5BB03E0F-3FED-48AF-A5FE-7F3E52513D99&share_source=WEIXIN&share_tag=s_i&timestamp=1677718075&unique_k=lk400UP&up_id=354740423<br>


# ChatGPT相关资料（欢迎下载，顺便留个宝贵的小星星(Star)哦）
1. LLM涌现能力-张俊林.pdf<br>
2. 对话式大型语言模型-邱锡鹏.pdf <br>
3. 探索大语言模型的垂直化训练技术与应用-陈运文.pdf<br>
4. 哈尔滨工业大学：ChatGPT调研报告.pdf<br>
5. 探索大语言模型的垂直化训练技术与应用-陈运文.pdf<br>
6. 中文模型和部分预训练数据集： https://github.com/lonePatient/awesome-pretrained-chinese-nlp-models#<br>
7. 154页微软GPT-4研究报告 <br>

# 致谢

在https://github.com/ssbuild/chatglm_finetuning 的基础上增加强化学习模块、Reward模型和Prompt指令数据收集模块，在此感谢chatglm_finetuning 的开发人员。<br>
