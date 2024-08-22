
# step1: Post-Training with Direct Preference Optimization
```
bash scripts/postrain_with_dpo.sh
```

## Input data formation
traing data stored in jsonl style, one line is as follows:
```
{"id":"1","source":"xllm_dataset","prompt":"","chosen":"","reject":""}
{"id":"2","source":"xllm_dataset","prompt":"","chosen":"","reject":""}
{"id":"3","source":"xllm_dataset","prompt":"","chosen":"","reject":""}
...
```

# step2: Make the trained model infering with vllm
```
bash dpo_infer.sh
```

# step3: Compare performance of two models of DPO vs. SFT with gpt4-0613
```
bash dpo_pairwise_score.sh
```

# step4: Calculate win-rate for two compared models
```
bash dpo_pairwise_winrate.sh 
```

