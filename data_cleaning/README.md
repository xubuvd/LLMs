```
Author: xubuvd<br>
Date: 13/08/2024<br>
Email: xubuvd@163.com<br>
```

# 🌱 Data Cleaning Recipe
It consists of four main stages:<br>
1. **Initial Data Cleaning**: Apply various heuristic methods to 28 domain-specific datasets.<br>
2. **Document-Level Deduplication**: Use MiniHash to remove duplicate documents.<br>
3. **Statistical Analysis**: Analyze the total number of tokens using the Llama3.1-8b-Instruct model.<br>
4. **Human Evaluation**: Conduct a manual review by sampling 100 data points.<br>

# 🍂 running and killing
```
nohup bash run_data_cleaning.sh > r.log 2>&1 &
```

```
bash stopall.sh
```

