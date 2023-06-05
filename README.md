# 从0到1预训练大语言模型
预训练框架：DeepSpeedChat (放弃metaseq，ColossalAI)<br>
SFT和RLHF框架： 选择 DeepSpeedChat框架<br>
模型结构： 基于LLaMA构造的80B大语言模型；<br>
<br>
后续跟进相关进展，有问题欢迎交流 xubuvd@163.com<br>
<br>
# 数据集构造，数据清洗方法
慢慢加...,陆续开源数据收集、构造，清洗方法，包括预训练数据和指令数据<br>

# iDeepSpeedChat 训练稳定后，会开源出来...
DeepSpeedChat 这个开源框架更像一个玩具，实际训起来会有很多问题，需要深入优化，才能应用于大规模、超大规模语言模型训练.<br>
现在开源的训练框架还没有能打的。<br>

# iDeepSpeedChat 优化 for 预训练和指令微调
No.      |Bug             |     原做法    | 修改           | 注评
 --------| :-----------:  |:-----------:  | :-----------:|:-----------:|
 1       | SFT Loss计算方式 | 所有tokens的预测损失（CE loss） |只计算模型respnse部分的预测损失 | 计算所有token的loss，效果不好，只计算模型response的loss，其它部分mask掉
 2       | 新增pre-train和<br>SFT两种损失Loss计算  |  只有SFT loss计算一种 |增加pre-train预训练 | 支持SFT和Pre-train混合训练，同一个batch内部有两类数据
 3       | <endoftext>不作为一个特殊字符 | <endoftext>作为一个文本序列 | 使用<eos>特殊字符代替，不需要新加一个<endoftext> | 参考论文“A General Language Assistant as a Laboratory for Alignment”，用作特殊字符效果好一些。
 4  | subprocess.CalledProcessError: Command '['which', 'c++']' returned non-zero exit status 1. | g++ wasn't installed. | #apt-get install build-essential | g++环境问题
 5  | wandb.errors.UsageError: api_key not configured (no-tty). | |  #wandb login 根据提示获取api key注册一下即可 | wandb使用问题，退出后再进入要：$ wandb login --relogin
 6 | Calling torch.distributed.barrier() <br>results in the program being killed |#df -lh | #rm -f /dev/shm/nccl-*|Docker容器共享内存太小存满导致，<br>容器里跑训练会遇到，<br>Slurm集群里，面对裸机没有此类问题。
 7 |huggingface/tokenizers: The current process just got forked, after parallelism has already been used. | | | warning，暂不处理
 8 | 数据集索引大小的bug| | | 2982929829一个不可能出现的数字，<br>index缓存文件名字名字重复，加入子进程的<br>global rank, loacl rank命名，已解决。
 9 |wandb: ERROR Run initialization has timed out after 60.0 sec. | |两个可能原因：<br>1，某些node的网络没有打开导致的；<br>2，节点的网络中断；<br>上述两个原因都遇到过。 | 排查两个原因
 10 | OSError: [Errno 122] Disk quota exceeded| 模型文件checkpoint写到管理节点本地，<br>仅保存了4个checkpoints，空间就🈵️了，<br>pytorch_model_10.bin,<br>pytorch_model_20.bin,<br>pytorch_model_30.bin,<br>pytorch_model_40.bin| 1. checkpoints先保存在/hpc_data/pangwei/ 【因为写权限问题，先保存该目录下】，速度变慢，10分钟加载模型文件；<br>2. 保留当前三个checkpoints；<br>3. 保存历史上最好的一个checkpoint，根据验证集上的perplexity指标。checkpoints分为三种，后缀分别为：norm_{steps}, bestppl_{steps}, final_{steps}。| 磁盘配额不够了，磁盘已满或超出了用户所能使用的配额上限
 11 |混合训，支持任意多个训练数据文件 | 支持一类数据集读取| 支持四类不同数据集，每一类可以任意多：<br>--train_pt_data_path []<br>--eval_pt_data_path []<br>--train_sft_data_path []<br>--eval_sft_data_path []<br>预训练数据集，后缀：训练集pt_train.jsonl, 验证集 pt_eval.jsonl;<br>指令微调数据集，后缀：训练集 sft_train.jsonl, 验证集 sft_eval.jsonl。 | 支持混合训的数据集管理，便于不同数据集的配比
 12 | resume问题| | 1）保存 checkpoint 元信息，包括<br>epoch, global step, optimizer,<br>checkpoints file name；<br>2）resume 继续训练，断点重新训练。| 加载当前最新的一个checkpoint；
 13 | (ReqNodeNotAvail, Un)<br>scancel一个任务<br>又重新启动会<br>遇到此类错误| | slurm系统scancel任务后挂掉| 重启slurm吧
 14 | 缓存空间溢满<br>OSError: [Errno 28] <br>No space left on device:<br>'/tmp/data_files'| | 从/tmp/目录调整到/data/XXX/目录| 
 15 | Save checkpoints，<br>按照固定steps计算perplexity，<br>保存最优模型| 每个epoch结束后<br>才计算perplexity| 增加一个参数 args.eval_save_steps，<br>默认100| 
 16 | Save checkpoint 并行化| checkpoint<br>路径全局唯一，<br>如果在多个节点（gnode）上启动任务，<br>输出路径重合，<br>互相干扰输出结果| Save 路径上新增一个节点名字，<br>可以同时多机启动多个一样的任务。<br>路径例子：/hpc_data/XXX/<br>actor-models/chinese_llama_plus-gnode07-13b-gnode07-20230524-0356。| chinese_llama_plus-gnode07-13b-gnode07-20230524-0356：<br>模型名字-模型大小-节点名字-时间串
 17 |加入wandb | |![Screen Shot 2023-06-05 at 5 33 49 PM](https://github.com/xubuvd/LLMs/assets/59753505/0ceac3f9-8fac-40c8-b374-47b9f166f276)| 训练参数，资源和指标可视化<br>f"{model_name}-{JOB_ID}"
 18 |加入TGS,TFlops | |![W B Chart 5_25_2023, 11_35_14 AM](https://github.com/xubuvd/LLMs/assets/59753505/1ab9fd41-ee38-4b40-8be8-4e0b53078310)![W B Chart 5_25_2023, 11_35_24 AM](https://github.com/xubuvd/LLMs/assets/59753505/c1171af6-24f5-4c6c-97b8-29d02833622c) | 卡吞吐率
 19 | 统计预训练和<br>指令微调数据集的<br>总tokens数量| | | 60W条SFT数据集<br>*Total tokens for pre-training: 0<br>Total tokens for sft: 51166867<br>*Total tokens: 51166867
 20 |自动记录<br>训练的超参数<br>Copy run_*.sh到<br>输出目录下 | |copy2(script, <br>f'{output_dir}/run.sh')| 模型的超参数与<br>模型checkpoint文件<br>保存在一起，<br>便于分析模型性能与<br>参数的关系
 21 | RuntimeError: Too many open files. <br>Communication with the workers is no longer possible. <br>Please increase the limit using <br>ulimit -n in the shell or <br>change the sharing strategy by <br>calling torch.multiprocessing.set_sharing_strategy('file_system')<br> at the beginning of your code | |torch.multiprocessing.<br>set_sharing_strategy<br>('file_system') |快速文件存储系统的设置存在问题
 22 |缺少多机多卡支持| | |deepspeed-chat没有写这部分，增加进行中
 23 |torch.cuda.OutOfMemoryError:<br>CUDA out of memory.| Chinese-LLaMA-13B的模型训练遇到OOM问题| | 在epoch 循环的内部，进行了 evaluation()，<br>evaluation设置了model.eval()模式, <br>但是退出evaluation再次进入 epoch 循环时，<br>没有设置model.train()模式。
 24 |A100-PCIe主板接口网速瓶颈| |<img width="1393" alt="Screen Shot 2023-05-25 at 4 33 20 PM" src="https://github.com/xubuvd/LLMs/assets/59753505/92fa0fcd-19cd-461b-94a1-0eefa951e0fb"> | 训练正向进行和反向进行，<br>GPU达到峰值，<br>多卡之间信息同步时，<br>GPU利用率跌倒谷底。<br>NV-link的接口GPU平稳。
 25 | Index cache<br>文件名字重合 | 下一个文件使用了<br>前一个文件的索引<br>d_path:f_identity_qa_cn_re.json, train_dataset_size:198991, eval_dataset:1009<br>d_path:f_multiturn_cn_69k.json, train_dataset_size:69318, eval_dataset:336<br>d_path:f_ver_qa_cn_28k.json, train_dataset_size:28140, eval_dataset:166 | 索引使用基座模型名字，<br>训练数据文件名字区别开 |  
 26 | 单条数据不补齐到<br>max_seq_len,<br>batch内部数据补齐到batch内最大长度 | batch内所有数据补齐到max_seq_len | | 提高计算效率，大多数长度相对很短
 27 | Moss-13B<br>loss异常 | ![img_v2_9723abf5-9038-4a56-97db-e9efa35fcf5g](https://github.com/xubuvd/LLMs/assets/59753505/c5a6dba6-4343-4e05-9112-2c4ad7b2694d)| <img width="685" alt="Screen Shot 2023-05-31 at 2 56 47 PM" src="https://github.com/xubuvd/LLMs/assets/59753505/73d98c57-f70b-49cd-967b-9ac736bab9cf"><img width="692" alt="Screen Shot 2023-05-31 at 2 59 33 PM" src="https://github.com/xubuvd/LLMs/assets/59753505/ff3aeaad-8a67-4bdc-af73-77dff559ceef"> | 重新设置lr, warmup... 
 27 | Moss-13B<br>loss异常 | ![img_v2_9723abf5-9038-4a56-97db-e9efa35fcf5g](https://github.com/xubuvd/LLMs/assets/59753505/c5a6dba6-4343-4e05-9112-2c4ad7b2694d)| 重新设置lr, warmup... |
 28 | GPU 卡住 |<img width="1263" alt="Screen Shot 2023-05-31 at 7 06 50 PM" src="https://github.com/xubuvd/LLMs/assets/59753505/71d4ec0f-bc17-49e7-9401-e62777e8e5da">| | 偶发性卡住，多机多卡常遇到的事<br>读数据阶段，0号GPU卡住不动.
 29 | OSError: [Errno 122] <br>Disk quota exceeded | root@master:~# quota -uvs user_name<br>Disk quotas for user user_name (uid 1006):<br>Filesystem   space   quota   limit   grace   files   quota   limit   grace<br>/dev/sda1   2862G*  2852G   2862G   6days   96582   2900k   3000k | | 快速存储系统写满<br>加系统监控，写到80%提前预警
 30 | 读数据，机器之间<br>速度差别较大 | 1. gnode03机器：<br>@master:~/$ tail -f training.log<br>3% ▎ 32285/1087101 [01:29<48:05, 365.57it/s]<br>2.gnode04机器：<br>@master:~/$ tail -f training.log<br>6% ▌ 63310/1087101 [02:53<45:11, 377.51it/s]<br>$3.gnode06机器:<br>@master:~/$ tail -f training.log<br>19% █▉ 211851/1087101 [03:36<15:01, 970.99it/s] | 读取一个108万条数据的文件，<br>有的机器节点耗时50分钟；<br>有的机器节点耗时18分钟。| 快速文件系统配置存在问题
 32 | - | - |- | 效果优化
 33 | - | - |- | 效果优化
 34 | - | - |- | 效果优化
 35 | - | - |- | 效果优化
 36 | 启动训练时不提供具体数据文件，<br>只提供数据集目录，<br>自动读取目录下的所有数据文件  | 提供每一个训练数据文件名字  |提供训练数据集目录  | 预训练数据是由很多小文件组成的，<br>不方便在启动脚本里加入许多文件名字
 


 
# iDeepSpeedChat 实训可视化图
 ![Screen Shot 2023-06-01 at 5 56 13 PM](https://github.com/xubuvd/LLMs/assets/59753505/f22c4024-fba3-4a74-b1a5-4d1ca7107bcf)<br>
 ![Screen Shot 2023-06-05 at 5 33 49 PM](https://github.com/xubuvd/LLMs/assets/59753505/d1290a1b-cfa2-4bcd-b9e5-62a13542dbc8)<br>
 ![Screen Shot 2023-06-05 at 5 34 09 PM](https://github.com/xubuvd/LLMs/assets/59753505/ab729983-d2ee-4258-974b-ba6abbe8969c)<br>
 ![Screen Shot 2023-06-05 at 5 34 32 PM](https://github.com/xubuvd/LLMs/assets/59753505/a8ed8fee-6f2f-4259-ac34-f105a2188b60)<br>

 
# OpenAI购买平台
https://eylink.cn/<br>

# 人类反馈的强化学习RLHF（Reinforcement Learning from Human Feedback）
RLHF是一种利用强化学习方法从人类反馈中学习的技术，使大语言模型与人类偏好保持对齐并遵循人类意图，有三个较好的开源pipeline实现，Beaver（河狸），开源地址：https://github.com/PKU-Alignment/safe-rlhf<br>
DeepSpeedChat和trlX。Beaver项目开源内容包括：(a)数据集与模型：PKU-SafeRLHF<br>
1.开源迄今为止最大的多轮 RLHF 数据集，规模达到 100 万条。<br>
2.开源经 Safe-RLHF 对齐训练得到的 7B 参数的语言模型——Beaver，并支持在线部署。<br>
3.开源了预训练的Reward Model和Cost Model的模型和参数。<br>
(b) 首个可复现的RLHF基准，PKU-Alignment/safe-rlhf支持以下功能：<br>
1. 支持LLM 模型的 SFT（Supervised Fine-Tuning）、RLHF训练、Safe RLHF训练。支持目前主流的预训练模型如 LLaMA、OPT 等模型的训练。<br>
2. 支持 Reward Model 和 Cost Model 训练。<br>
3. 提供安全约束满足的多尺度验证方式，支持 BIG-bench、GPT-4 Evaluation 等。<br>
4. 支持参数定制化的 RLHF 和数据集定制接口。<br>
![rlhf_githubs](https://github.com/xubuvd/LLMs/assets/59753505/c3f76de4-64b5-4854-af31-2a78c27cb28c)


# ChatGLM-6B 全套微调代码，经过两周对ChatGLM-6B的指令微调，两条经验如下：<br>
1. glm-6B是经过SFT和RLHF后的版本，再次微调不够友好<br>
2. 需要在一个干净的纯基座模型上进行微调（SFT），使用高质量的业务指令数据；RLHF，是一个难题，需要一个高质量的reward model，目前正确的rlhf pipeline比较稀少，训练出来好的效果也是一个挑战。<br>

# 国内大语言模型ChatGPT专区，欢迎交流邮箱：xubuvd@163.com

# Open-source of LLMs 

 If you like the project, please show your support by leaving a star ⭐.

 No. |Projects | URL  | Comments
 --------| :-----------:  |:-----------:  | :-----------:|
 1|LLaMA |  https://github.com/facebookresearch/llama | 模型结构transformer block: ![LLaMA](https://user-images.githubusercontent.com/59753505/233027941-70b3b478-1137-40f1-a288-ae8858c8f7ce.jpg)
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
25  | DeepSpeed-Chat | https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-chat<br>https://github.com/microsoft/DeepSpeedExamples | 微调框架：包括指令微调（SFT），Reward model 和强化学习对齐意图（RLHF）
26 | fairseq | https://github.com/facebookresearch/fairseq | FaceBook开源的大语言模型预训练框架
27 | metaseq | https://github.com/facebookresearch/metaseq | FaceBook开源的大语言模型预训练模型框架，基于fairseq的新版本
28 | MiniGPT-4 | https://github.com/Vision-CAIR/MiniGPT-4 | 多模态大模型，基于 BLIP-2 和 Vicuna（LLaMA-7B基座）, 阿卜杜拉国王科技大学
29 | moss<br>(复旦大学) | https://github.com/OpenLMLab/MOSS<br>https://huggingface.co/models?other=moss | moss-13B开源了，重要贡献是提供了一个纯基座
30 | 红睡衣(RedPajama)开源计划 | https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T<br>预处理仓库:https://github.com/togethercomputer/RedPajama-Data | 红睡衣开源计划总共包括三部分：<br>1. 高质量、大规模、高覆盖度的预训练数据集；<br>2. 在预训练数据集上训练出的基础模型；<br>3. 指令调优数据集和模型，比基本模型更安全、可靠。<br>Ontocord.AI，苏黎世联邦理工学院DS3Lab，斯坦福CRFM，斯坦福Hazy Research 和蒙特利尔学习算法研究所的开源计划，旨在生成可复现、完全开放、最先进的语言模型，即从零一直开源到ChatGPT！。
31 | Panda<br>中文开源大语言模型 |  https://github.com/dandelionsllm/pandallm |基于Llama-7B、-13B、-33B和-65B进行了中文领域的持续预训练，在中文基准测试中表现优异，远超同等类型的中文语言模型，Panda的模型和训练所用中文数据集将以开源形式发布，任何人都可以免费使用和参与开发。
32 | BELLE<br>（LLaMA，链家） | https://github.com/LianjiaTech/BELLE | BELLE-LLaMA-EXT-13B，在LLaMA-13B的基础上扩展中文词表，并在400万高质量的对话数据上进行训练。
33 | Linly-Chinese-LLaMA   | https://github.com/CVI-SZU/Linly | LLaMA-7B/13B基础上，中文二次预训练，上下文长度2048


# ColossalAI 的性能测试
1， ZeRO 2的性能，tflops 约为251<br>
![Screen Shot 2023-04-21 at 10 32 14 AM](https://user-images.githubusercontent.com/59753505/233526814-3331b468-37d0-44dc-8484-d78da549466a.png)<br>
2, ZeRO 2和3的性能对比<br>

Model    |   ZeRO        | GPU数量	       | Bs	          | Seq len	    | Gpu mem	    | Gpu Usage	  | Iter	       |  Tflops	 |   TGS<br>(tokens per gpu per second)
--------| :-----------:  |:-----------:  | :-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|  
LLaMA-7B | zero2	        |  2            | 50	          | 2048	       |  90%	       |    60%	     |   25s	      |      250	 |   4096
LLaMA-7B |zero3	         |  2            | 76	          | 2048	       |  97%	       |    80%	     |   30s	      |      300	 |   5188

3, 黄色曲线是ZeRO3，绿色曲线是ZeRO2<br>
![img_v2_ef608a22-cae9-41a1-b725-0946e695e92g](https://user-images.githubusercontent.com/59753505/233527058-cb9a3bc8-23f3-456f-8bd8-6a8a773ae2f6.png)



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

# 评价大模型复杂推理能力的Benchmark
https://github.com/FranxYao/chain-of-thought-hub<br>

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

# 开源预训练代码
TencentPretrain: Tencent Pre-training Framework<br>
https://github.com/Tencent/TencentPretrain<br>

# 北京邮电大学 王小捷教授 ChatGPT 讲座分享

https://www.bilibili.com/video/BV1G24y187yx/?buvid=ZB476BB0B8710E3C4F548C7C2778AA1427C6&is_story_h5=false&mid=AdBmq4Rn7y73B2EmgVj16A%3D%3D&p=1&plat_id=114&share_from=ugc&share_medium=iphone&share_plat=ios&share_session_id=5BB03E0F-3FED-48AF-A5FE-7F3E52513D99&share_source=WEIXIN&share_tag=s_i&timestamp=1677718075&unique_k=lk400UP&up_id=354740423<br>

# 训练大语言模型中，一个查看GPU 卡的好用小工具
$ pip install nvitop<br>
$ nvitop<br>
效果见下图，显存使用率，GPU利用率，CPU利用率和内存使用率，一览无余：<br>
![WechatIMG73](https://github.com/xubuvd/LLMs/assets/59753505/a53899c2-e193-4ace-8efc-ac2ec5bc3e94)


# ChatGPT相关资料（欢迎下载，顺便留个宝贵的小星星(Star)哦）
1. LLM涌现能力-张俊林.pdf<br>
2. 对话式大型语言模型-邱锡鹏.pdf <br>
3. 探索大语言模型的垂直化训练技术与应用-陈运文.pdf<br>
4. 哈尔滨工业大学：ChatGPT调研报告.pdf<br>
5. 探索大语言模型的垂直化训练技术与应用-陈运文.pdf<br>
6. 中文模型和部分预训练数据集： https://github.com/lonePatient/awesome-pretrained-chinese-nlp-models#<br>
7. 154页微软GPT-4研究报告 <br>
8. 陆奇《新范式 新时代 新机会》北京场.pdf<br>

# 致谢
