# 决定从0到1预训练大语言模型，放弃在别的预训练模型基础上直接微调的方式
预训练框架：选择 metaseq，github地址：https://github.com/facebookresearch/metaseq<br>
SFT和RLHF框架： 选择 DeepSpeed Chat框架，github地址：https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-chat；
模型结构： 选择 LLaMA-13B和LLaMA-65B两种模型结构；<br>
<br>
后续跟进相关进展，有问题欢迎交流 xubuvd@163.com<br>
<br>

# 今日头条，微头条“宽广的潮白河”，每日连载《大语言模型训练日记》，欢迎评论区交流问题，尽力回复每一条问题～

# 对待大语言模型的态度：
战略上可以藐视，但战术上如果藐视，那就是无知了。大语言模型训练依然是一个非常艰难的细活。尽管有那么多开源的东西，但是最致命的点、出效果的点，是否毫无保留的开源，需要打问号❓的。个人开源的代码基本上都不能规模化使用，坑多且深。选择大厂开源的代码，相对好一些，但仍有较多的坑...<br>

# ChatGLM-6B 全套微调代码，经过两周的一番折腾，决定放弃ChatGLM-6B的指令微调
经验教训：需要全套的SFT，RLHF代码，如果在某个人开源代码上增加这个代码，bug较多，训练不稳定等问题较多，尤其RLHF强化学习是一种精细的活，能稳定训练的坑太多了，花费的时间较多，还不见得有效果。<br>

# 国内大语言模型ChatGPT专区，欢迎交流邮箱：xubuvd@163.com

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
9|chatglm-6B_finetuning | https://github.com/ssbuild/chatglm_finetuning | 1,chatGLM-6B的微调版本，正在补充RLHF代码，陆续放出来；28层网络，每一层的模型结构：![chatglm](https://user-images.githubusercontent.com/59753505/228441877-63aae805-b862-4c42-839e-c60ef9e2d135.jpg);<br><br>2，两种微调方式：LoRA微调和SFT微调，28层网络，指令数据5K，单机8卡，A100，80G显存，batch size 8, epoch 1或2（有生成重复问题），大约20分钟内完成；<br>3, 借鉴 Colossal-AI/Open-Assistant的强化学习代码（PPO，PPO-ptx算法），Colossal-AI可以迁移过来，被实践过。<br>4，Reward model，可选较多，直接基于GLM-6B模型微调一个Reward model。<br>难点就是训练数据；GPT3.5使用了33K的人工标注数据训练 Reword model。<br>每个问题，配置四个答案ABCD，人工从好到差排序比如B>A>D>C，排序后的数据微调Reward model。<br>单机8卡，A100， 80G，train_batch_size=4, max_seq_len设置成512，才可以跑50K级别的微调数据集，这份代码感觉有点疑问，需要优化的地方挺多的
10|ChatGLM-Tuning| https://github.com/mymusise/ChatGLM-Tuning| ChatGLM-6B的又一个微调版本
11|中文语言模型骆驼 (Luotuo)|https://github.com/LC1332/Chinese-alpaca-lora |基于 LLaMA、Stanford Alpaca、Alpaca LoRA、Japanese-Alpaca-LoRA 等完成，单卡就能完成训练部署
12|Alpaca-COT数据集 | https://github.com/PhoebusSi/Alpaca-CoT | 思维链（CoT）数据集，增强大语言模型的推理能力
13|Bloom | https://huggingface.co/bigscience/bloom | 训练和代码比较全
14|中文LLaMA&Alpaca大语言模型 | https://github.com/ymcui/Chinese-LLaMA-Alpaca | 在原版LLaMA（7B和13B）的基础上扩充了中文词表并使用了中文数据进行二次预训练，进一步提升了中文基础语义理解能力。同时，在中文LLaMA的基础上，本项目使用了中文指令数据进行指令精调，显著提升了模型对指令的理解和执行能力。
15 |Colossal-AI/ColossalChat | https://github.com/hpcaitech/ColossalAI | 训练和代码比较全，包括 RLHF 训练代码；以 LLaMA 为基础预训练模型；开源了7B和13B两种模型；
16 | Cerebras-GPT七个版本 | 官网地址：https://www.cerebras.net/blog/cerebras-gpt-a-family-of-open-compute-efficient-large-language-models<br>GPT地址:https://www.cerebras.net/cerebras-gpt<br>Hugging Face地址:https://huggingface.co/cerebras | 七个参数版本：1.16亿、2.56亿、5.9亿、13亿、27亿、67亿和130亿参数, 基于GPT的生成人工智能大型语言模型
17 | BloombergGPT<br>(金融领域) | https://arxiv.org/abs/2303.17564 | BloombergGPT的训练数据库名为FINPILE，构建迄今为止最大的特定领域数据集, 由一系列英文金融信息组成，包括新闻、文件、新闻稿、网络爬取的金融文件以及提取到的社交媒体消息。训练专门用于金融领域的LLM，拥有500亿参数的语言模型。
18 | dolly-v1-6b | https://github.com/databrickslabs/dolly | 1, fine-tuned on a ~52K instruction (Self-Instruct从 ChatGPT自动获取)；<br>2，deepspeed ZeRo 3加速训练;<br>3.可借鉴的：deepspeed ZeRo 3加速训练部分；
19 | ChatDoctor | https://github.com/Kent0n-Li/ChatDoctor | 医疗领域对话模型，基于LLaMA-7B微调的大模型，经过四轮微调：<br>第一轮微调：羊驼的52K instruction-following 数据<br>;第二轮微调：患者和医生之间的5K对话数据集（ChatGPT GenMedGPT-5k和疾病数据库生成）；<br>第三轮微调：患者和医生之间的真实对话（HealthCareMagic-200k）；<br>第四轮微调：患者和医生之间的真实对话（icliniq-26k）.
20 | 开源中文对话大模型BELLE | https://github.com/LianjiaTech/BELLE | BELLE-7B（基于 BLOOMZ-7B1-mt 微调）<br>BELLE-13B的感觉还行
21 |  InstructGLM | https://github.com/yanqiangmiffy/InstructGLM | 基于ChatGLM-6B+LoRA在指令数据集上进行微调；截止4月4号下午，InstructGLM存在以下缺点：多卡不支持，原作者在回答issues时也确认了；社区不活跃，两周不更新代码，坑反馈的太少（才11条），deepspeed没有；这是我用三块卡跑，卡负载不均衡：![image](https://user-images.githubusercontent.com/59753505/229748872-df8f3909-f8e5-454c-8378-56766f8aa1a2.png)
22 | Cerebras-GPT  | https://huggingface.co/cerebras/Cerebras-GPT-13B | 参数量级130亿，大小比肩最近Meta开放的LLaMA-13B，数据集、模型权重和计算优化训练，全部开源。可商用！
23 | Baize<br>(加利福尼亚大学, 基于 LLaMA 的微调)|https://github.com/project-baize/baize-chatbot | 数据集生成: 让 ChatGPT 与自己进行对话，模拟用户和AI机器人的回复」。这个生成的语料数据集是在多轮对话的背景下训练和评估聊天模型的宝贵资源。此外，通过指定种子数据集，可以从特定领域进行采样，并微调聊天模型以专门针对特定领域，例如医疗保健或金融。<br>Parameter-efficient tuning， 输入序列的最大长度设置为512，LoRA中的秩k设置为8，使用8位整数格式 (int8) ，Adam 优化器」更新LoRA 参数，batch size为64，learning rate为2e-4、1e-4和 5e-5，可训练的LoRA参数在 NVIDIA A100-80GB GPU 上微调了1个 epoch。
24 | Open-Llama |https://github.com/s-JoL/Open-Llama | Open-Llama是一个开源项目，提供了一整套用于构建大型语言模型的训练流程，从数据集准备到分词、预训练、指令调优，以及强化学习技术 RLHF。采用FastChat项目相同方法测评Open-Llama的效果和GPT3.5的效果对比，经过测试在中文问题上可以达到GPT3.5 84%的水平。
25  | DeepSpeed-Chat | https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-chat | 微调框架：包括指令微调（SFT），Reward model 和强化学习对齐意图（RLHF）
26 | fairseq | https://github.com/facebookresearch/fairseq | FaceBook开源的大语言模型预训练框架
27 | metaseq | https://github.com/facebookresearch/metaseq | FaceBook开源的大语言模型预训练模型框架，基于fairseq的新版本

# 可下载的中英文指令数据集，仍需要清洗，下载见目录instruction_data/
1，身份识别指令数据，需要自己修改细节内容 developer_instruction.json<br>
2, 51504条中文指令数据，instinwild_ch.json<br>
3, 52191条英文指令数据， instinwild_en.json<br>
4, 10021+10444条羊驼指令数据， alpaca-zh-data-part-00.json和alpaca-zh-data-part-01.json<br>
5, 543314条中文指令数据，belle.json<br>
6，还有许多指令数据，因为上传单个文件不能超过25M，需要的请私信 xubuvd@163.com <br>

# 可下载的开源数据集
1，悟道 200G文本，下载链接：https://data.baai.ac.cn/details/WuDaoCorporaText, 数据格式<br>
    {<br>
        "id": 2,<br>
        "uniqueKey": "074ca2f564544686f0fb6da026e00cac",<br>
        "titleUkey": "231af201b8e7e359f8ab3c1a716dbe86",<br>
        "dataType": "孕育常识",<br>
        "title": "幼儿急疹一定会出疹子吗",<br>
        "content": "婴儿抵抗力低下,时常发生小病小痛,可操碎了做父母的心,相信每个初为人母的妈妈,都会为了孩子的健康成长而对襁褓中的新生儿关怀备至,作为一个合格的妈妈,需要了解更多关于更好的照顾孩子的知识,才能防患于未然。那么幼儿急疹一定会出疹子吗。幼儿急疹一定会出疹子吗 幼儿急疹,也叫烧疹或玫瑰疹,是由病毒感染而引起的突发性皮疹,一年四季都可以发生,尤以春、秋两季较为普遍
。常见于出生6个月至1岁左右的宝宝。幼儿急疹的潜伏期大约是10~15天。它虽然是传染性的疾病,却很安全,不会象麻疹、水痘那样广泛传染,家中成员同时患上的机会不大。 症状为宝宝首先是持续3~4天发高
烧,体温在39~40度之间,热退后周身迅速出现皮疹,并且皮疹很快消退,没有脱屑,没有色素沉着。这些婴儿在没有出现皮疹前也有发热,热度可以比较高,但是感冒症状并不明显,精神、食欲等都还可以,咽喉可能
有些红,颈部、枕部的淋巴结可以触到,但无触痛感,其他也没有什么症状和体症。当体温将退或已退时,全身出现玫瑰红色的皮疹时才恍然大悟,其实这时幼儿急疹已近尾声。幼儿急疹对婴儿健康并没什么影响,
出过一次后将终身免疫。幼儿急疹的护理 (1)宝宝要多休息,不剧烈玩耍,体育锻炼暂停。 (2)多喝水,适当的加入果汁,这样即提高了维生素的摄入又利于出汗和排尿,可以促进毒物排出。 (3)宝宝患病期间吃
些易消化食物,已经可以吃固体食物的宝宝,此时吃流质或半流质饮食。但是注意尽量要有营养。(不建议喝糖分较高的甜水,宝宝此时食欲不佳,会影响宝宝食欲) (4)刻意的适当补充维生素c和维生素b。 (5)宝
宝休息的地方要安静,空气注意流通并保持新鲜。 (6)被子不能盖得太厚太多,这样不利于散热。 (7)注意宝宝的皮肤要保持清洁卫生,经常给孩子擦去身上的汗渍,即防止着凉同时防止出疹的宝宝感染。 (8)体
温超过39度时,可用温水或37%的酒精为孩子擦身,防止高热惊厥。(小宝宝不建议酒精降温,如果家长不知道酒 精浓度也不建议给大宝宝使用,对皮肤有刺激性) (9)幼儿急疹是为数不多的出疹可以外出玩耍见风
的疾病,但是中医认为此时宝宝体质虚,如果宝宝汗多,则不建议出 门见风。 (10) 此时部分宝宝可能很赖妈妈,希望一直依偎在妈妈怀里,可能是疾病导致宝宝的心理需要。所以请妈妈们尽量满足 宝宝的心理
需要,也有利于亲子关系。"<br>
    },<br>
<br>
2, Pile, 1.3T的英文数据, 需要强力清洗，下载链接 https://pile.eleuther.ai/, 数据格式：<br>
{"text": "Q:\n\nFor some reason after inputting cin text, the cout comes out blank. Any ideas?\n\nSo I am trying to create a simple Text RPG. But, this one problem is holding me back.\n#include <iostream>\n\nusing namespace std;\n\nint main()\n{\n int input;\n long Sven;\n long Macy;\n\n  cout<<\"Choose your Character- 1.Sven or 2.Macy: \";\n cin>>input;\n cin.ignore();\n\n if ( input == Sven ){\n cout<<\"Welcome to CRPG, my good Sir!\";\n }\n\n if (input == Macy ){\n cout<<\"Girls cant fight, go back: \";\n }\n}\n\nSo this code here is what I have at the moment. When I run the program, it allows me to type the name of the character I want to choose. But, the output is always just a blank area of text. I am more or less new to C++ but, I have nice prior knowledge. Any help is great.\n\nA:\n\nWhat threw me off is when you said it allows me to type the name of the character I want to choose\nIn that case, go ahead with comparing the strings:\nEDIT: As Mohammed suggested, comparing strings can be done directly:\nstring input;\n\ncout<<\"Choose your Character- 1.Sven or 2.Macy: \";\ncin>>input;\ncin.ignore();\n\nif ( input == \"Sven\" ){\n cout<<\"Welcome to CRPG, my good Sir!\";\n}\n\nelse if ( input == \"Macy\"){\ncout<<\"Girls cant fight, go back: \";\n}\n\n", "meta": {"pile_set_name": "StackExchange"}}<br>
<br>


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
