"""Reward functions for GRPO training."""

import json
import math
import re
from typing import Dict
import requests
import json
import os
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
import subprocess

from .utils import is_e2b_available
if is_e2b_available():
    from dotenv import load_dotenv
    from e2b_code_interpreter import Sandbox

    load_dotenv()

def request_LongCoT(model_name, prompt, temperature=0.3, tokens=2048, stop_symbol=''):
    prompt_completion_list = []
    for i in range(len(prompt)):
        prompt_completion = {
            "model": model_name,
            "messages": [
                {"role": "user",
                 "content": prompt[i]},
            ],
            "temperature": temperature,
            "tokens": tokens,
            "stop_symbol": stop_symbol,
        }
        prompt_completion_list.append(prompt_completion)

    response_list = requests.post("http://localhost:8876", headers={"Content-Type": "application/json"},
                             data=json.dumps(prompt_completion_list))

    return response_list

def request_ShortCoT(model_name, prompt, temperature=0.3, tokens=2048, stop_symbol=''):

    prompt_completion_list = []
    for i in range(len(prompt)):
        prompt_completion = {
            "model": model_name,
            "messages": [
                {"role": "user",
                 "content": prompt[i]},
            ],
            "temperature": temperature,
            "tokens": tokens,
            "stop_symbol": stop_symbol,
        }
        prompt_completion_list.append(prompt_completion)


    response_list = requests.post("http://localhost:8973", headers={"Content-Type": "application/json"},
                             data=json.dumps(prompt_completion_list))

    return response_list

def multi_turn_sampling(LongCoT_dir, ShortCoT_dir,):
    ShortCoT_response = ''
    short_pattern2 = re.compile(r'(?s)^<answer>.*</answer>$')
    turn = 1
    tmp = ''
    while short_pattern2.match(ShortCoT_response) == None:
        if turn == 1:
            print(f"================ Turn {turn}================")
            LongCoT_prompts = sys_LongCoT_prompt + question
            LongCoT_response = str(request_LongCoT(model_name=str(LongCoT_dir), prompt=LongCoT_prompts, temperature=0.3, tokens=2048, stop_symbol = None).json())
            print('=========(LongCoT)==============')
            print(LongCoT_response)
            ShortCoT_prompts = sys_ShortCoT_prompt + question + LongCoT_response
            ShortCoT_response = str(request_ShortCoT(model_name=str(ShortCoT_dir), prompt=ShortCoT_prompts, temperature=0.3, tokens=2048, stop_symbol = None).json())
            print('=========(ShortCoT)========')
            print(ShortCoT_response)
            turn += 1
            tmp = tmp + LongCoT_response + '\n\n' + ShortCoT_response
        else:
            print(f"================ Turn {turn}================")
            LongCoT_prompts = sys_LongCoT_prompt + question + tmp
            LongCoT_response = str(request_LongCoT(model_name=str(LongCoT_dir), prompt=LongCoT_prompts, temperature=0.3, tokens=2048, stop_symbol = None).json())
            print('=========(LongCoT)==============')
            print(LongCoT_response)
            ShortCoT_prompts = sys_ShortCoT_prompt + question + tmp + LongCoT_response
            ShortCoT_response = str(request_ShortCoT(model_name=str(ShortCoT_dir), prompt=ShortCoT_prompts, temperature=0.3, tokens=2048, stop_symbol = None).json())
            print('=========(ShortCoT)========')
            print(ShortCoT_response)
            turn += 1
            tmp = tmp + LongCoT_response + '\n\n' + ShortCoT_response

    return tmp

def LongCoT_reward(prompts, completions, solution, **kwargs):
    SYS_prompt = ''
    contents_pre = [completion[0]["content"] for completion in completions]
    questions =[prompt[1]["content"] for prompt in prompts] 

    for k in range(len(response_list)):
        response = response_list[k]
        contents.append(response)

    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            try:
                reward = float(verify(answer_parsed, gold_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = 0.0
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 1.0
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards

def ShortCoT_reward(prompts, completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""

    SYS_prompt = ''
    contents_pre = [completion[0]["content"] for completion in completions]
    questions =[prompt[1]["content"] for prompt in prompts] 
    for k in range(len(response_list)):
        response = response_list[k]
        contents.append(response)

    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            try:
                reward = float(verify(answer_parsed, gold_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = 0.0
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 1.0
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)
    return rewards

def is_valid_pattern(s):
    whitespace = r'\s*'
    

    content = r'[^<]*'

    think_think = f'{whitespace}<think>{content}</think>{whitespace}<answer>'
    
    answer_rethink = f'{content}</rethink>'
    answer_answer = f'{content}</answer>{whitespace}'

    middle_think = f'(?:{whitespace}<think>{content}</think>{whitespace}<answer>|{whitespace}<think>{content}</overthink>{whitespace}<answer>)'
    
    complete_pattern = f"""^
        (?:{think_overthink}|{think_think})  
        (?:{answer_rethink}{middle_think})*   
        {answer_answer}$                      
    """
    
    return bool(re.match(complete_pattern, s, re.VERBOSE))

def Multi_turn_LongCoT_format_reward(completions, **kwargs):
    
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = is_valid_multi_turn_LongCoT_pattern(completion_contents)
    return [1.0 if match else 0.0 for match in matches]

def is_valid_multi_turn_ShortCoT_pattern(s):
    pattren_short = re.compile(r'(?s)^<answer>[^<]*</rethink>$|^<answer>[^<]*</answer>$')
    short_pattern1 = re.compile(r'(?s)^<answer>.*</rethink>$')
    short_pattern2 = re.compile(r'(?s)^<answer>.*</answer>$')
    reward_multi_turn_ShortCoT = []
    for k in range(len(s)):
        content = str(s[k])
        empty_check_short = re.fullmatch(r'(?s)<answer>\s*</(answer|rethink)>', content)

        if pattren_short.match(str(s[k])) and (short_pattern1.match(str(s[k])) or short_pattern2.match(str(s[k]))) and not empty_check_short:
            reward_multi_turn_ShortCoT.append(True)
        else:
            reward_multi_turn_ShortCoT.append(False)
    return reward_multi_turn_ShortCoT

def Multi_turn_ShortCoT_format_reward(completions, **kwargs):
   
    completion_contents = [completion[0]["content"] for completion in completions]
    # matches = [is_valid_multi_turn_ShortCoT_pattern(content) for content in completion_contents]
    matches = is_valid_multi_turn_ShortCoT_pattern(completion_contents)
    return [1.0 if match else 0.0 for match in matches]


def len_reward(completions: list[Dict[str, str]], solution: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards