from fastapi import FastAPI, Request
from vllm import LLM, SamplingParams
import json
import multiprocessing
multiprocessing.set_start_method('spawn')

app = FastAPI()

model_name = None
model_vllm = None
sampling_params = None


@app.post("/")
async def create_item(request: Request):
    global model_name, model_vllm, sampling_params
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    model_name = json_post_list.get('model')
    message = json_post_list.get('messages')
    prompt = json_post_raw['messages'][0]['content']
    print('=======')
    print(prompt)
    print('=======')
    temperature_ = json_post_list.get('temperature')
    tokens = json_post_list.get('tokens')
    stop_symbol = json_post_list.get('stop_symbol')
    # n = json_post_list.get('n')
    if model_vllm is None:
        model_path = str(model_name)
        model_vllm = LLM(model=model_path, trust_remote_code=True, gpu_memory_utilization=0.9)
    else:
        pass

    if stop_symbol != None:
        sampling_param = SamplingParams(max_tokens=tokens, temperature=temperature_, include_stop_str_in_output=True, stop=str(stop_symbol),ignore_eos=False)
        print('====')
        completion = model_vllm.generate(prompt_list, sampling_params=sampling_param, use_tqdm=True)
        print(completion)
        response_list = []
        for output in completion:
            # number of sequences
            sequences = output.outputs
            for i in range(len(sequences)):
                response = str(sequences[i].text)
                response_list.append(response)

        return response_list
    else:
        sampling_params = SamplingParams(max_tokens=tokens, temperature=temperature_)

        completion = model_vllm.chat(messages=message_list, sampling_params=sampling_params, use_tqdm=True)
        response_list = []
        # batch of prompt
        for output in completion:
            # number of sequences
            sequences = output.outputs
            for i in range(len(sequences)):
                response = str(sequences[i].text)
                response_list.append(response)

        return response_list
