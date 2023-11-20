# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import random
import timeit
import json
import timeit
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm, trange

from transformers import (
    BertConfig,
    BertTokenizer,
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification
)
from utils.modeling_xlmroberta import XLMRForSequenceClassification


logger = logging.getLogger(__name__)


def convert_type(tensor, data_type):
    if data_type == 'fp16':
        return tensor.half()
    elif data_type == 'fp32':
        return tensor.float()
    elif data_type == 'bf16':
        return tensor.bfloat16()


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    parser.add_argument("--model_type", type=str, help="ori, ths, thsext")
    parser.add_argument('--data_type', type=str, choices=['fp32', 'fp16', 'bf16'], default='fp16')
    parser.add_argument('--ths_path', type=str, default='./lib/libth_transformer.so',
                        help='path of the pyt_fastertransformer dynamic lib file')
    parser.add_argument('--remove_padding', action='store_true',
                        help='Remove the padding of sentences of encoder.')

    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s",
        args.local_rank,
        device,
        args.n_gpu,
    )

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    logger.info("Parameters %s", args)

    # load model
    model = XLMRForSequenceClassification.from_pretrained(args.model_name_or_path, torchscript=True)
    model.to(args.device)
    
    hf_model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path).to(args.device)

    if args.data_type == 'fp16':
        logger.info("Use fp16")
        model.half()
        hf_model.half()
    elif args.data_type == 'bf16':
        logger.info("Use bf16")
        model.bfloat16()
    
    if args.model_type == 'thsext':
        logger.info("Use custom xlmRoBERTa encoder for TorchScript")
        from utils.encoder import EncoderWeights, CustomEncoder
        weights = EncoderWeights(
            model.config.num_hidden_layers, model.config.hidden_size,
            torch.load(os.path.join(args.model_name_or_path, 'pytorch_model.bin'), map_location='cpu'))
        weights.to_cuda()
        if args.data_type == 'fp16':
            weights.to_half()
        elif args.data_type == 'bf16':
            weights.to_bfloat16()
        enc = CustomEncoder(model.config.num_hidden_layers,
                            model.config.num_attention_heads,
                            model.config.hidden_size//model.config.num_attention_heads,
                            weights,
                            remove_padding=args.remove_padding,
                            path=os.path.abspath(args.ths_path))
        enc_ = torch.jit.script(enc)
        model.replace_encoder(enc_)
    
    logger.info("Use TorchScript mode")
    with open('/workspace/code/profile_flagembedding/rerank_input.json') as f:
        data = json.load(f)
        sentence_pairs = data["qp_pairs"]
    
    inputs = tokenizer(
        sentence_pairs,
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=512,
    ).to(args.device)
    # print(inputs["input_ids"].size())
    
    model.eval()
    hf_model.eval()
    with torch.no_grad():
        model_ = torch.jit.trace(model, (inputs['input_ids'], inputs['attention_mask']))
        model = model_
        
        for i in range(100):
            res_result1 = model(inputs['input_ids'], inputs['attention_mask'])[0]
        
        t10 = timeit.default_timer()
        for i in range(100):
            res_eff_encoder = model(inputs['input_ids'], inputs['attention_mask'])[0]
        t_jit = timeit.default_timer() - t10
        print("eff encoder time_cost:", t_jit/100)
        
        for i in range(100):
            res_result1 = hf_model(inputs['input_ids'], inputs['attention_mask'])[0]
        
        t10 = timeit.default_timer()
        for i in range(100):
            res_hug = hf_model(inputs['input_ids'], inputs['attention_mask'])[0]
        t_hug = timeit.default_timer() - t10
        print("huggingface time_cost:", t_hug/100)
        diff = torch.abs(res_hug - res_eff_encoder)
        print('EFT Mean diff: {}'.format(torch.mean(diff)))
        print('EFT Max diff:  {}'.format(torch.max(diff)))
        print('EFT Min diff:  {}'.format(torch.min(diff)))

if __name__ == "__main__":
    main()
