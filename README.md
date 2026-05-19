# 领域大语言模型实操技术全栈

#垂直赛道的机会
今日头条号“宽广的潮白河”，每日不断更新大模型垂直赛道的一些心得体会，与大家分享和讨论。
https://www.toutiao.com/w/1865612933719043/?log_from=a00f16600efe9_1779187284612

## 领域推理小模型
推理模型（Reasoning Model）是新的垂直模型。在每个垂直领域，需要一个基于基座模型训练的推理模型，然后也可以基于这个推理模型作为后续其它训练的基础模型。对于多数中小型企业，能在业务上落地的领域模型，考虑两点：部署成本和业务性能。3B或者7B的小模型是一个好的备选，当然要先测试一下业务性能，检查一下是否有潜力后再选择一个合适的做强化学习训练。比如，如下表是在某业务上实测的性能，Qwen2.5-3B的精准率和召回率都很接近14B，是有训练潜力的，可选为基础模型。<br>
Models | 测试集1（7K条）   | 测试集2（4K条）| 每条平均耗时（秒）
--------| :-----------:  | :-----------:| :-----------:
Baidu3.5-ERNIE-Bot | P:57.06,R:68.54,A:72.65 | P:38.56,R:75.0,A:89.55 | 19~23秒
Deep-Seek-R1-671B | P:58.47,R:47.13,A:71.46 | P:47.57,R:60.48,A:93.76 | 34~64秒
Qwen2.5-72B-Instruct | P:53.0,R:78.37,A:69.93 | P:34.4,R:80.46,A:87.55 | 12~16秒
Qwen2.5-32B-Instruct | P:61.8,R:58.14,A:74.36 | P:52.31,R:58.58,A:93.13 | 8~10秒
Qwen2.5-14B-Instruct | P:51.56,R:65.42,A:72.87 | P:28.5,R:63.82,A:85.23 | 9~12秒
Qwen2.5-7B-Instruct |  P:44.35,R:88.33,A:59.62 | P:19.78,R:87.06,A:73.53 | 10~13秒
DeepSeek-R1-Distill-Qwen-7B | P:38.89,R:73.64,A:55.98 | P:12.73,R:76.85,A:63.36 | 13~14秒
<b>Qwen2.5-3B-Instruct</b> | P:44.2,R:74.62,A:62.79 | P:21.73,R:71.49,A:81.44 | 10~13秒
Qwen2.5-1.5B-Instruct | P:33.03,R:100.0,A:33.18 | P:7.28,R:100.0,A:7.82 | 3~4秒

<br>指标说明：P：精准率，R：召回率，A：准确率<br>

## Scaling Law for SFT & DPO


## 领域LLMs的数据模型 for SFT & DPO
分享一个观点："Without a well thought-out data model, organizations will be constantly fighting short-term data engineering problems and connecting the business needs with data requirements will be extremely difficult. "如果没有一个经过深思熟虑的数据模型，组织将不断地与短期数据工程问题作斗争，将业务需求与数据需求联系起来将是极其困难的。" 来源网页需要付费账户，不再注明。<br>
数据生产方案，是从原始数据到可训练数据的一整套流程，包括原始数据筛选（data selection）、数据清洗、数据embedding、数据质量打分、数据难度打分、数据Tag等等，利用这些特征将数据集分层，之后筛选进入SFT训练的数据集，偏好数据的种子从中选择，经过几十个模型生产数据，构造偏好关系，得到用于偏好对齐的训练数据。<br>

## 领域LLMs的有效训练方案 for SFT & DPO
训练方案将SFT和DPO串联起来，一轮SFT后紧接着一轮DPO，往复迭代三轮，每一轮都增加新数据进来，构造更有难度的偏好数据。<br>

## 预训练、SFT和DPO偏好对齐代码框架
已经在36台A100，288块GPU卡上实际跑过预训练、增量预训练和SFT，涉及1.5T的tokens训练，等待开源...<br>

### SFT代码开源
地址：https://github.com/xubuvd/PhasedSFT<br>
代码在16台A100、A800和H100机器上，经受过了大规模的SFT训练，数据量最多达 200万条，已经训练过的模型有Llama1、Llama2、Llama3、Qwen1、Qwen1.5、Qwen2 和 Mistral，稳定可用的。<br>

### 偏好训练 Post-Training with Direct Preference Optimization
```
地址：codes_datasets/Postraining_dpo
用法参考：codes_datasets/Postraining_dpo/README.md
```
代码在16台A100、A800和H100机器上训练过，DPO偏好数据最多到50万条，稳定可用。

## 预训练数据清洗源代码和高并发框架
大规模预训练数据集数据集清洗，具有可读性好、可用性好，Python多线程并发，包括一套清洗工程规范，清洗策略，多线程框架，文本去重代码，黑名单词典等内容。<br>
2024.08.13，开源出来一部分基于启发式规则的多线程python清洗代码，代码复用性、可用性好：codes_datasets/DataCleaning<br>
清洗代码唯一的启动脚本：<br>
```
nohup bash run_data_cleaning.sh > r.log 2>&1 &
```
停止程序的脚本：<br>
```
bash stopall.sh
```
## SFT 数据集清洗和数据质量提升代码框架
🍂step1: 数据清洗，也可复用预训练数据清洗代码<br>
🍂step2: 数据质量提升<br>
1.指令质量打分<br>
2.指令难度打分<br>
3.聚类+语义去重复<br>
代码整理中，等待开源...<br>

## 中文大语言模型的安全对齐工作，用于安全审核评估
一千条安全类指令数据集:codes_datasets/SFTData/cn-Safety-Prompts-gpt12k_baichuan1K.jsonl<br>
一千条安全类中文指令数据集,关于诈骗、欺骗、个人安全、个人攻击性歧视性仇恨言论、黄赌毒等类型: codes_datasets/SFTData/cn-Chinese-harmlessness-1K.jsonl<br>

## 开源sft数据集构造
### 1. 大规模COT高中试题数据集，命名为“cn-sft-exams-highSchool-1M.jsonl”
开源超大规模高中试题指令数据集，100万条中文指令数据，涵盖语文、数学、物理、化学、地理、历史、政治和英文。<br>
已上传：cn-sft-exams-highSchool-1k.jsonl<br>
指令格式：{"id": "26069", "data": ["问题","答案"]}<br>

### 2. SFT数据质量优化
数据质量和多样性，这两条是模型能力的来源。<br>
优化前：<br>
![sftExp8 2_tsne_100000](https://github.com/user-attachments/assets/5f38e112-ed85-47fa-b919-e8431fc63af6)
<br>
优化后：<br>
![sftExp8 3_tsne_100000](https://github.com/user-attachments/assets/5a0fe60a-a8ea-48a5-ba04-7945ea7995a7)
上述数据分布图生成代码：codes_datasets/utils/tsne_cluster_show.py<br>

### 2. 行政职业能力测验题数据集，命名为“cn-sft-CS-APAT-30K.jsonl”
共 3 万条行测试题，逻辑推理题目，旨在提升模型的逻辑能力。<br>
已上传：cn-exam-high-school-5W.jsonl.zip<br>
指令格式：{"id": "26069", "data": ["问题","答案"]}<br>

## 落地大语言模型LLM，关键问题是在某个垂直领域，如何构造高质量指令数据集？有方法论吗？
有的！<br>

## 从0到1预训练大语言模型

## 论文引用请参考
```
@online{CnSftData,
  author = {XuBu},
  year = {2024},
  title = {Chinese Large Language Models},
  url = {https://github.com/xubuvd/LLMs},
  month = {APR},
  lastaccessed = {APR 19, 2024}
}
```

## DPO 偏好数据对齐
DPO非常有效，目前在3万偏好数据集上，测试13B、70B和7B模型，效果非常显著；<br>
DPO版本相对于SFT模型，胜率提升了 「10个」百分点以上，人的观感收益也很显著。具体的，在业务上的准确率（accuracy），相对于SFT模型，提升了 「14.6」个百分点，F1值，提升了 「13.5」个百分点。<br>
### DPO偏序数据集构造<br>
### DPO训练，关键指标图示(一部分)：<br>
![Screen Shot 2023-12-14 at 2 24 54 PM](https://github.com/xubuvd/LLMs/assets/59753505/f22b0b2d-02ba-4cf5-aae2-77085664779c)

## 预训练数据收集和清洗
数据收集和清洗，对大语言模型（LLM）最终效果的影响极度重要。<br>
数据清洗需要一套方法论，预训练数据的三项关键指标：质量高、多样性和数量大。<br>
### 什么是质量高？
- 一大段自然语言文本，语法上连贯流畅，没有插入无关的词汇、句子，语义上完整；<br>
- 标点符号符合行文规范；
### 什么是多样性？
  - 覆盖各个通用领域和学科的知识<br>
  - 构建一个全面的「行业领域知识体系」<br>
  - 整合上述各类信息源，确保广泛的知识覆盖<br>
### 什么是数量大？
- 总量<br>
  - Pretrain 数据总量大，以LLaMA tokens计算，约 1~2 T tokens 左右<br>
- 分量<br>
  - 从数据多样性上看，各种数据类型的数据都有，大小基本符合互联网上的数据自然分布<br>

## 交流群
如果你也对本项目感兴趣，欢迎加入群聊参与讨论交流。
![xubu](https://github.com/xubuvd/LLMs/assets/59753505/1841da9f-110e-4b91-be0a-dbe351b399a0)

## 基于DeepSpeedChat改造，用于RLHF训练的框架
No.      |Bug             |     原做法    | 修改           | 注评
 --------| :-----------:  |:-----------:  | :-----------:|:-----------:|
 1       | Loss计算方式 | 所有tokens的预测损失（CE loss） |只计算模型respnse部分的预测损失 | 计算所有token的loss，效果不好，只计算模型response的loss，其它部分mask掉
 2       | 新增pre-train和<br>SFT两种损失Loss计算  |  只有loss计算一种 |增加pre-train预训练 | 支持SFT和Pre-train混合训练，同一个batch内部有两类数据
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
 37 | load_dataset不能读大文件| pyarrow.lib.ArrowCapacityError: <br>array cannot contain more than 2147483646 bytes, <br>have 2572789185 | 内存映射，流式读取 | from datasets import load_dataset
 38 | 数据量超出某个临界点OOM，<br>1T内存都被爆满 |<img width="1156" alt="Screen Shot 2023-06-10 at 12 21 25 AM" src="https://github.com/xubuvd/LLMs/assets/59753505/2585dad0-d80d-4c83-953d-d9a84e8a4fda"> | 加载数据之前，内存消耗：<br>1-ToatlMem:1007, UsedMem:211, FreeMem:768<br>加载数据之后，内存消耗：<br>2-ToatlMem:1007, UsedMem:898, FreeMem:42<br>此时，空闲内存42G，开始进入 deepspeed.initialize（），使用的是ZeRO 2优化 | 原模型训练流程拆分成两部分：<br>1， 生成内存映射文件（MemapGen）：<br>加载原始数据集，tokenizer后padding到max_seq_len大小，写入内存映射文件；<br>内存映射文件包括三个，分别是input_ids文件，attention_mask文件和 labels文件，还有一个config json配置文件。<br>2，模型训练<br>加载config文件和三个内存映射文件，跑起来...<br>优化点： 组装batch时，随机读取整个映射文件，速度巨慢；MemapGen时随机shuffle好文件，加载后按顺序组装batch。
 39 | OSError: [Errno 122] Disk quota exceeded | $ quota -uvs user_name  |  空间满，所有训练都会停止   | 增加硬盘存储监控，空间使用85%时提前预警，主动发现隐患点

## iDeepSpeedChat 实训可视化图
 ![Screen Shot 2023-06-01 at 5 56 13 PM](https://github.com/xubuvd/LLMs/assets/59753505/f22c4024-fba3-4a74-b1a5-4d1ca7107bcf)<br>
 ![Screen Shot 2023-06-05 at 5 33 49 PM](https://github.com/xubuvd/LLMs/assets/59753505/d1290a1b-cfa2-4bcd-b9e5-62a13542dbc8)<br>
 ![Screen Shot 2023-06-05 at 5 34 09 PM](https://github.com/xubuvd/LLMs/assets/59753505/ab729983-d2ee-4258-974b-ba6abbe8969c)<br>
 ![Screen Shot 2023-06-05 at 5 34 32 PM](https://github.com/xubuvd/LLMs/assets/59753505/a8ed8fee-6f2f-4259-ac34-f105a2188b60)<br>


## Open-source of LLMs 

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
34 | CAMEL        | https://github.com/camel-ai/camel  | https://www.camel-ai.org/chat
35 | Ziya-LLaMA-13B-v1 | https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1 | 姜子牙通用大模型V1是基于LLaMa的130亿参数的大规模预训练模型
36 | 悟道·天鹰（Aquila） | https://github.com/FlagAI-Open/FlagAI/tree/master/examples/Aquila/Aquila-chat |

## ColossalAI 的性能测试
1， ZeRO 2的性能，tflops 约为251<br>
![Screen Shot 2023-04-21 at 10 32 14 AM](https://user-images.githubusercontent.com/59753505/233526814-3331b468-37d0-44dc-8484-d78da549466a.png)<br>
2, ZeRO 2和3的性能对比<br>

Model    |   ZeRO        | GPU数量	       | Bs	          | Seq len	    | Gpu mem	    | Gpu Usage	  | Iter	       |  Tflops	 |   TGS<br>(tokens per gpu per second)
--------| :-----------:  |:-----------:  | :-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|  
LLaMA-7B | zero2	        |  2            | 50	          | 2048	       |  90%	       |    60%	     |   25s	      |      250	 |   4096
LLaMA-7B |zero3	         |  2            | 76	          | 2048	       |  97%	       |    80%	     |   30s	      |      300	 |   5188

3, 黄色曲线是ZeRO3，绿色曲线是ZeRO2<br>
![img_v2_ef608a22-cae9-41a1-b725-0946e695e92g](https://user-images.githubusercontent.com/59753505/233527058-cb9a3bc8-23f3-456f-8bd8-6a8a773ae2f6.png)

## 北京邮电大学 王小捷教授 ChatGPT 讲座分享

https://www.bilibili.com/video/BV1G24y187yx/?buvid=ZB476BB0B8710E3C4F548C7C2778AA1427C6&is_story_h5=false&mid=AdBmq4Rn7y73B2EmgVj16A%3D%3D&p=1&plat_id=114&share_from=ugc&share_medium=iphone&share_plat=ios&share_session_id=5BB03E0F-3FED-48AF-A5FE-7F3E52513D99&share_source=WEIXIN&share_tag=s_i&timestamp=1677718075&unique_k=lk400UP&up_id=354740423<br>

## 致谢
 
