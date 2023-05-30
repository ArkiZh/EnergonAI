import argparse
import logging
import random
from typing import Optional

import uvicorn
from energonai import QueueFullError, launch_engine, AsyncEngine
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from batch import BatchManagerForGeneration
from cache import ListCache, MissCacheError


class GenerationTaskReq(BaseModel):
    max_tokens: int = Field(gt=0, le=256, example=64)
    prompt: str = Field(
        min_length=1, example='Question: Where were the 2004 Olympics held?\nAnswer: Athens, Greece\n\nQuestion: What is the longest river on the earth?\nAnswer:')
    top_k: Optional[int] = Field(default=None, gt=0, example=50)
    top_p: Optional[float] = Field(default=None, gt=0.0, lt=1.0, example=0.5)
    temperature: Optional[float] = Field(default=None, gt=0.0, lt=1.0, example=0.7)


app = FastAPI()

engine :AsyncEngine = None

@app.post('/generation')
async def generate(data: GenerationTaskReq, request: Request):
    logger.info(f'{request.client.host}:{request.client.port} - "{request.method} {request.url.path}" - {data}')
    key = (data.prompt, data.max_tokens)
    try:
        if cache is None:
            raise MissCacheError()
        outputs = cache.get(key)
        output = random.choice(outputs)
        logger.info('Cache hit')
    except MissCacheError:
        try:
            uid = id(data)
            engine.submit(uid, data=data.prompt)
            output = await engine.wait(uid)
            if cache is not None:
                cache.add(key, output)
        except QueueFullError as e:
            raise HTTPException(status_code=406, detail=e.args[0])
        except Exception as e:
            import traceback
            traceback.print_exc()


    return {'text': output}


@app.on_event("shutdown")
async def shutdown(*_):
    engine.shutdown()
    server.should_exit = True
    server.force_exit = True
    await server.shutdown()


def print_args(args: argparse.Namespace):
    print('\n==> Args:')
    for k, v in args.__dict__.items():
        print(f'{k} = {v}')


FIXED_CACHE_KEYS = [
    ('Question: What is the name of the largest continent on earth?\nAnswer: Asia\n\nQuestion: What is at the center of the solar system?\nAnswer:', 64),
    ('A chat between a salesman and a student.\n\nSalesman: Hi boy, are you looking for a new phone?\nStudent: Yes, my phone is not functioning well.\nSalesman: What is your budget? \nStudent: I have received my scholarship so I am fine with any phone.\nSalesman: Great, then perhaps this latest flagship phone is just right for you.', 64),
    ("English: I am happy today.\nChinese: 我今天很开心。\n\nEnglish: I am going to play basketball.\nChinese: 我一会去打篮球。\n\nEnglish: Let's celebrate our anniversary.\nChinese:", 64)
]

from transformers import LlamaTokenizer, AutoModelForCausalLM

def create_model(tp_size, pretrained_model_name_or_path, ignore_mismatched_sizes):
    if tp_size<=1:
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path, ignore_mismatched_sizes=ignore_mismatched_sizes)
        model = model.half()
    else:
        from colossalai.tensor import (
            ColoParameter,
            ComputePattern,
            ComputeSpec,
            ProcessGroup,
            ReplicaSpec,
            ShardSpec,
        )
        from colossalai.utils import get_current_device
        from colossalai.zero import ColoInitContext
        import torch
        from transformers import LlamaForCausalLM
        with ColoInitContext(device=get_current_device(),
                        dtype=torch.half,
                        default_dist_spec=None,
                        default_pg=None):
            model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path, ignore_mismatched_sizes=ignore_mismatched_sizes)
        import tp
        tp_pg = ProcessGroup(tp_degree=tp_size)    
        tp.tensor_parallelize(model, tp_pg)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default="/home/lumk/zk/llama-project/train-llama/logs/05-22_15:56/checkpoint_200")
    parser.add_argument("--tokenizer_path", default="/home/lumk/zk/llama-project/train-llama/logs/05-22_15:56/checkpoint_200")
    parser.add_argument('--tp', type=int, default=8)
    parser.add_argument('--master_host', default='localhost')
    parser.add_argument('--master_port', type=int, default=19990)
    parser.add_argument('--rpc_port', type=int, default=19980)
    parser.add_argument('--max_batch_size', type=int, default=8)
    parser.add_argument('--pipe_size', type=int, default=1)
    parser.add_argument('--queue_size', type=int, default=0)
    parser.add_argument('--http_host', default='0.0.0.0')
    parser.add_argument('--http_port', type=int, default=7070)
    parser.add_argument('--cache_size', type=int, default=5)
    parser.add_argument('--cache_list_size', type=int, default=1)
    args = parser.parse_args()
    print_args(args)

    logger = logging.getLogger(__name__)
    # import os
    # os.environ["TRANSFORMERS_OFFLINE"] = "0"
    
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)
    if args.cache_size > 0:
        cache = ListCache(args.cache_size, args.cache_list_size, fixed_keys=FIXED_CACHE_KEYS)
    else:
        cache = None
    engine = launch_engine(args.tp, args.pipe_size, args.master_host, args.master_port, args.rpc_port, create_model,
                           batch_manager=BatchManagerForGeneration(tokenizer=tokenizer, max_batch_size=2),
                           pipe_size=args.pipe_size,
                           queue_size=args.queue_size,
                           pretrained_model_name_or_path=args.model_path,
                           ignore_mismatched_sizes=True,
                           tp_size = args.tp
                           )
    config = uvicorn.Config(app, host=args.http_host, port=args.http_port)
    server = uvicorn.Server(config=config)
    server.run()
