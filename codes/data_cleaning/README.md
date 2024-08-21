```
Author: xubuvd
Date: 13/08/2024
Email: xubuvd@163.com
```

# 🌱 数据清洗方案 - Data Cleaning Recipe
它包含四个主要阶段：<br>
1. **初始数据清洗**：对28个特定领域的数据集应用多种启发式过滤方法。<br>
2. **文档级去重**：使用 MiniHash 去除重复文档。<br>
3. **统计分析**：使用 Llama3.1-8b-Instruct 模型分析总词汇量。<br>
4. **人工评估**：对100个数据点进行抽样和手动审查。<br>
<br>
It consists of four main stages:<br>
1. **Initial Data Cleaning**: Apply various heuristic filtering methods to 28 domain-specific datasets.<br>
2. **Document-Level Deduplication**: Use MiniHash to remove duplicate documents.<br>
3. **Statistical Analysis**: Analyze the total number of tokens using the Llama3.1-8b-Instruct model.<br>
4. **Human Evaluation**: Conduct a manual review by sampling 100 data points.<br>

# 🍂 启动和暂停 - running and killing
```
nohup bash run_data_cleaning.sh > r.log 2>&1 &
bash stopall.sh
```

