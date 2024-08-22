import argparse
import json
import os
import time
import openai
from tqdm import tqdm
import asyncio
from typing import Any
import logging
#import tiktoken

import unicodedata
openai.api_key = "sk-dWZKRO"
openai.api_base = "https://"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# openai.api_key = ''
async def dispatch_openai_requests(
    messages_list,#: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
):# -> list[str]:
    """Dispatches requests to OpenAI API asynchronously.
    """
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)

def parse_score(review):
    try:
        score_pair = review.split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        sp = score_pair.split(" ")
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            raise Exception("Invalid score pair.")
    except Exception as e:
        logger.error(
            f"{e}\nContent: {review}\n" "You must manually fix the score pair."
        )
        return [-1, -1]

def gen_prompt(ques, ans1, ans2):

    sys_prompt = "You are a helpful and precise assistant for checking the quality of the answer."
    prompt_template = "[Question]\n{question}\n\n[The Start of Assistant 1's Answer]\n{answer_1}\n\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{answer_2}\n\n[The End of Assistant 2's Answer]\n\n[System]\n{criteria}\n\n"
    #criteria = "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."
    criteria = "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease only output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. Do not explain for your scores."
    prompt = prompt_template.format(
        question=ques, answer_1=ans1, answer_2=ans2, criteria=criteria
    )
    return sys_prompt, prompt


def get_json_list(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "r") as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatGPT-based QA evaluation.")
    parser.add_argument("-i1", "--input1_file")
    parser.add_argument("-i2", "--input2_file")
    parser.add_argument("-k1", "--key_1")
    parser.add_argument("-k2", "--key_2")
    parser.add_argument("-s", "--eval_scorer")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size to call OpenAI GPT",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="maximum number of tokens produced in the output",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The output dir."
    )
    args = parser.parse_args()
    
    i1_jsons = json.load(open(args.input1_file))
    i2_jsons = json.load(open(args.input2_file))
    assert len(i1_jsons) == len(i2_jsons)

    message_list = []
    total_len = len(i1_jsons)
    question_idx_list = list(range(total_len))
    
    if "vicuna" in args.input1_file:
        prompt_key = 'text'
        data_idx = 'question_id'
        dst = 'vicuna' # dst is used for saving the content
    elif "koala" in args.input1_file:
        prompt_key = 'prompt'
        dst = 'koala'
        data_idx = 'id'
    elif "sinstruct" in args.input1_file:
        prompt_key = 'instruction'
        dst = 'sinstruct'
        data_idx = 'id'
    elif "wizardlm" in args.input1_file:
        prompt_key = 'Instruction'
        dst = 'wizardlm'
        data_idx = 'idx'
    elif "anthropic" in args.input1_file:
        prompt_key = "instruction"
        dst = "anthropic"
        data_idx = 'id'
    elif "oasst" in args.input1_file:
        prompt_key = "instruction"
        dst = "oasst"
        data_idx = 'id'
    elif "frontis" in args.input1_file:
        prompt_key = "data"
        dst = "frontis"
        data_idx = 'id'

    for i in question_idx_list:
        assert i1_jsons[i][data_idx] == i2_jsons[i][data_idx]

        instruction = i1_jsons[i][prompt_key]
        if "sinstruct" in args.input1_file:
            instances = i1_jsons[i]['instances']
            assert len(instances) == 1
            if  instances[0]['input']:
                ques = '{instruction} Input: {input}'.format(instruction=instruction,input= instances[0]['input'])
            else:
                ques = instruction
        elif "anthropic" in args.input1_file or "oasst" in args.input1_file:
            input_ = i1_jsons[i]['input']
            if input_:
                ques = '{instruction} Input: {input}'.format(instruction=instruction,input=input_)
            else:
                ques = instruction
        elif "frontis" in args.input1_file:
            ques = instruction[0]
        else:
            ques = instruction

        ans1 = i1_jsons[i]['response']
        ans2 = i2_jsons[i]['response']
        
        sys_prompt, prompt = gen_prompt(ques, ans1, ans2)
        message = [
            {"role": "system", "content": sys_prompt},
            {"role": "user","content": prompt},
        ]
        message_list.append(message)
    
    predictions = []
    i = 0
    wait_base = 1
    retry = 0
    error = 0
    pbar = tqdm(total=len(message_list))
    batch_size = args.batch_size
    saved_josns = []

    assert len(message_list) == len(i1_jsons)

    # "gpt-3.5-turbo-0613" "gpt-4-0613"
    while i < len(message_list):
        try:
            batch_predictions = asyncio.run(
                dispatch_openai_requests(
                    messages_list=message_list[i:i+batch_size],
                    model=args.eval_scorer,
                    temperature=0.0,
                    max_tokens=args.max_tokens,
                    top_p=1.0,
                )
            )
            print("batch_predictions:",batch_predictions)
            predictions += batch_predictions
            saved_josns += i1_jsons[i:i+batch_size]
            retry = 0
            i += batch_size
            wait_base = 1
            pbar.update(batch_size)
        except Exception as e:
            retry += 1
            error += 1
            print("asyncio.run-Exception: {}".format(e),flush=True)
            print("Batch error: {}-{}".format(i, i+batch_size),flush=True)
            print("retry number: {}".format(retry),flush=True)
            print("error number: {}".format(error),flush=True)
            #if retry > 10:
            #    print("retry bigger than 10, ignoring this data.")
            #    i += batch_size
            time.sleep(wait_base)
            wait_base = wait_base*1.05
    pbar.close()

    assert len(saved_josns) == len(predictions)

    output_dir = args.output_dir
    output_review_file = args.key_1 +'-'+args.key_2 +'-'+ dst + '.json'
    if os.path.isdir(output_dir) is not True:
        os.mkdir(output_dir)
    output_review_f = os.path.join(output_dir, output_review_file)
    
    with open(f"{output_review_f}", "w",encoding='utf-8') as fo:
        js_dict = []
        for idx, prediction in enumerate(predictions):
            message = prediction.choices[0].message
            content = unicodedata.normalize('NFKC', message['content'])
            scores = parse_score(content)
            js_dict.append(saved_josns[idx])
            js_dict[-1]['review'] = content
            js_dict[-1]['score'] = scores
            js_dict[-1]['eval_scorer'] = args.eval_scorer
        json.dump(js_dict, fo, indent=4, ensure_ascii=False)

