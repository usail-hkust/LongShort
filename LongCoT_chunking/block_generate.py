import random
import json
import requests
import re
import time
import demjson
from vllm import LLM, SamplingParams
import multiprocessing
multiprocessing.set_start_method('spawn')
from tqdm import tqdm
import sys
model_vllm = None
sampling_params = None


def request_vllm(model_name, message):
    global model_vllm, sampling_params

    if model_vllm is None:
        model_path =  str(model_name)
        model_vllm = LLM(model=model_path, trust_remote_code=True, enforce_eager=True, tensor_parallel_size=4, gpu_memory_utilization=0.9)
    else:
        pass

    sampling_params = SamplingParams(max_tokens=7120, temperature=0.1)

    completion = model_vllm.chat(messages=message, sampling_params=sampling_params, use_tqdm=True)
    response_list = []
    # batch of prompt
    for output in completion:
        # number of sequences
        sequences = output.outputs
        for i in range(len(sequences)):
            response = str(sequences[i].text)
            response_list.append(response)

    return response_list[0]


def request(prompt_completion):

    answer ={}
    response = request_vllm(prompt_completion['model'], prompt_completion['messages'])
    answer['response'] = response

    return answer['response']

def split_by_signals_to_json(text, step_signal):
    # Create a regex pattern that matches any of the step_signal elements
    pattern = '|'.join(re.escape(signal) for signal in step_signal)
    # Use re.split to split the text by the pattern
    split_text = re.split(pattern, text)

    formatted_text = "\n".join([f"Step {i + 1}: {line}" for i, line in enumerate(split_text)])

    return split_text, formatted_text

def block_thought(block, step_level_thought):
    
    result = {}

    for block_name, block_info in block.items():
        start = int(block_info.get('start'))
        end = int(block_info.get('end'))

        
        start_idx = int(start) - 1  
        end_idx = int(end)         

        if 0 <= start_idx < len(step_level_thought) and 0 < end_idx <= len(step_level_thought):
            result[block_name] = '\n\n'.join(step_level_thought[start_idx:end_idx])
        else:
            result[block_name] = ''  

    return result


def jiancha(prompt, block):
    flag = True
    erro = ''
    step_pattern = re.compile(r'Step (\d+): (.*?)$', re.MULTILINE)
    steps = step_pattern.findall(prompt)

    n_block = len(block)
    last_block_index = 0
    for ix in range(n_block):

        k_name = f"block {ix + 1}"
        start = block[k_name]['start']
        end = block[k_name]['end']
        block_type = block[k_name]['block type']

        if start != last_block_index + 1:
            print(f"{k_name} error")
            erro = k_name+"error"
            flag = False
        last_block_index = end

    if last_block_index != len(steps):
        flag = True
        print("length not  match")
        erro = "length not  match"
    return flag, erro

model_name = sys.argv[1]
file_path = sys.argv[2] # json
thought_path = sys.argv[3]  # jsonl

step_signal = ['\n\n']

block_prompt_task_profile = """You are a professional math teacher. When solving math problems, you typically begin by understanding the problem, then break down the solution into several major steps, solving each part in sequence, arriving at the final result, and performing a second verification. Next, I will give you a student's solution process to a math problem. Based on this process, you need to reconstruct their general thinking process during problem-solving — for example, first understanding the problem, then decomposing it, and so on.

Specifically, please first read the problem and their solution. Then summarize the student's thinking process into distinct blocks such as “problem understanding” and so on. Record the start and end step index for each block. Return your answer in JSON format, where the key is "block 1", "block 2", etc., and the value includes the fields start, end, and block type. The start and end should refer to the index of each step, and block type should be a broad category (not too specific), such as "problem understanding".
"""
block_prompt_output_format = """Fill in start and end using step indices, and write block type. Return only the following JSON format without any explanations: {\'block 1\': {\'start\': \'\', \'end\': \'\', \'block type\': \'\'}, ...}"""

with open(file_path, 'r', encoding='utf-8') as file:
    total_lines = sum(1 for line in file)

try:
    with open(thought_path, 'r') as file:
        question_thought_block_thought = json.load(file)
    num = len(question_thought_block_thought)
except:
    question_thought_block_thought = []
    num = 0


with open(file_path, 'r', encoding='utf-8') as file:
    for current_line_number, line in enumerate(tqdm(file, total=total_lines), start=1):
        if current_line_number-1 < num:
            continue
        else:
            question_thought_block_thought_idx = {}
            json_line = json.loads(line)
            try:
                question = json_line['question']
            except:
                question = json_line['problem']
            try:
                thought = json_line['thought']
            except:
                thought = json_line['response']
            answer = json_line['answer']

            step_level_thought, formatted_thought = split_by_signals_to_json(thought, step_signal)

            question_thought_block_thought_idx['question'] = question
            question_thought_block_thought_idx['answer'] = answer
            question_thought_block_thought_idx['thought'] = thought
            ## blocking

            prompt_completion_block = {
                "model": model_name,
                "messages": [
                    {"role": "user",
                     "content": block_prompt_task_profile + 'Problem：\n\n```\n\n' + question + '\n\n```\n\nSolution：\n\n```\n\n' + formatted_thought + '\n\n```\n\n' + block_prompt_output_format},
                ],
                "temperature": 0.1,
            }
            num = 0
            __flag = False
            while __flag!=True and num<5:
                try:
                    response = request(prompt_completion_block)
                    question_thought_block_thought_idx['block_prompt'] = prompt_completion_block['messages'][0]['content']
                    question_thought_block_thought_idx['block_prompt_results'] = response
                    response = response.replace('```json', '').replace('```', '')
                    block = demjson.decode(response)
                    # print(formatted_thought)
                    question_thought_block_thought_idx['block'] = block
                    
                    block_level_thought = block_thought(block, step_level_thought)
                    question_thought_block_thought_idx['block_thought'] = block_level_thought
                    flag, erro = jiancha(question_thought_block_thought_idx['block_prompt'], block)
                    if flag:
                        __flag = True
                        print(block)
                        print('=========')
                        break
                    else:
                        num+=1
                        print('===warning==')
                        print(erro)
                        print(block)
                        print('====warning=====')
                        continue
                except Exception as e:
                    num+=1
                    time.sleep(1)
                    print(e)

            question_thought_block_thought.append(question_thought_block_thought_idx)
            with open(thought_path, 'w', encoding="utf-8") as json_file:
                json.dump(question_thought_block_thought, json_file, indent=4, ensure_ascii=False)

