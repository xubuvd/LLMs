# 国内大语言模型ChatGPT专区，欢迎交流
# Open-source of LLMs 

 If you like the project, please show your support by leaving a star ⭐.

 Projects | URL  | Comments
 --------| :-----------:  | :-----------:|
 LLaMA |  https://github.com/facebookresearch/llama | 
 OpenChatKit|https://github.com/togethercomputer/OpenChatKit | 200亿参数，48层，单机八卡
 Open-Assistant | https://github.com/LAION-AI/Open-Assistant |12B或者LLAMA-7B两个版本，Open Assistant 全流程训练细节（GPT3+RL）,https://zhuanlan.zhihu.com/p/609003237
 ChatGLM-6B | https://github.com/THUDM/ChatGLM-6B | ChatGLM-6B 是一个开源的、支持中英双语的对话语言模型，基于 General Language Model (GLM) 架构，具有 62 亿参数。可以作为很好的基础模型，在此之上做二次研发，在特定垂直领域。没有放出源代码，只有训练好的模型。
GLM-130B | https://github.com/THUDM/GLM-130B/ | 1300亿参数的中/英文大模型，没有放出源代码，只有训练好的模型
 Alpaca 7B | https://crfm.stanford.edu/2023/03/13/alpaca.html |  A Strong Open-Source Instruction-Following Model，a model fine-tuned from the LLaMA 7B model on 52K instruction-following demonstrations. 
 Claude | 用户可以通过邮箱等信息注册申请试用, 产品地址：https://www.anthropic.com/product, 申请地址：https://www.anthropic.com/earlyaccess, API说明: https://console.anthropic.com/docs/api |两个版本的 Claude：Claude 和 Claude Instant。 Claude 是最先进的高性能模型，而 Claude Instant 是更轻、更便宜、更快的选择。
 LLama/ChatLLama|https://github.com/nebuly-ai/nebullvm/tree/main/apps/accelerate/chatllama | 中文支持不好，有全套的SFT，RLHF训练过程
chatglm-6B_finetuning | https://github.com/ssbuild/chatglm_finetuning | chatGLM-6B的微调版本，不够全
ChatGLM-Tuning| https://github.com/mymusise/ChatGLM-Tuning| ChatGLM-6B的又一个微调版本
中文语言模型骆驼 (Luotuo)|https://github.com/LC1332/Chinese-alpaca-lora |基于 LLaMA、Stanford Alpaca、Alpaca LoRA、Japanese-Alpaca-LoRA 等完成，单卡就能完成训练部署
Alpaca-COT数据集 | https://github.com/PhoebusSi/Alpaca-CoT | 思维链（CoT）数据集，增强大语言模型的推理能力

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

