import os
import sys

import argparse
import torch
import transformers
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional, Dict, Sequence, List
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import json
from collect import create_sft
from tqdm import tqdm
import copy
from torch.cuda.amp import autocast
import random
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: Let's think step by step."
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    ),
    "prompt_no_input_v2": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{question}\n\n### Response: Let's think step by step."
    ),
}

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings, tokenizer: transformers.PreTrainedTokenizer):
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources,
    targets,
    tokenizer: transformers.PreTrainedTokenizer,
):
    sources_tokenized = _tokenize_fn(sources, tokenizer)
    input_ids = sources_tokenized["input_ids"]
    return dict(input_ids=input_ids, labels=copy.deepcopy(input_ids))


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, max_sample: int):
        super(SupervisedDataset, self).__init__()

        # dataset_for_eval = load_dataset(data_path)['train']
        try:
            list_data_dict = jload(data_path)
        except BaseException:
            with open(data_path, 'r') as f:
                lines = f.readlines()
            list_data_dict = [json.loads(line.strip()) for line in lines]

        # logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        # print(list_data_dict[0])
        if 'instruction' in list_data_dict[0]:
            pass
        else:
            def get_input(query):
                if query.find('\n') == -1:
                    return ''
                return '\n'.join(query.split('\n')[1:])
            list_data_dict = [{'instruction':data['query'].split('\n')[0], 'input':get_input(data['query']), 'output':data['response']} for data in list_data_dict[0]]
        # import ipdb; ipdb.set_trace()
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        if max_sample is not None:
            print(f"only use {max_sample} samples")
            if max_sample <= len(sources):
                # 如果max_sample小于等于数组长度，直接使用random.sample函数
                random_indices = random.sample(range(len(sources)), max_sample)
                sources = [sources[i] for i in random_indices]
                targets = [targets[i] for i in random_indices]
            else:
                # 如果max_sample大于数组长度，将数组长度设置为max_sample
                print("max_sample exceeds the length of the array. Using the entire array.")
                sources = sources
                targets = targets
        else:
            print(f"using {len(sources)} samples")
            sources = sources
            targets = targets

        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i], id=i)

def padding(inputs, padding_token, cutoff = None):
    num_elems = len(inputs)
    if cutoff is None:
        cutoff = max([len(item) for item in inputs])
    else:
        cutoff = min(max([len(item) for item in inputs]), cutoff)
    
    tokens = torch.ones(num_elems, cutoff).long().to(inputs[0].device) * padding_token
    for i in range(num_elems):
        toks = inputs[i]
        length = min(cutoff, len(toks))
        tokens[i, -length:] = toks[-length:]
    return tokens

def sequence_gather(s, world_size, pad_tok_id):
    local_size = torch.tensor(s.size(), device=s.device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_length = max(size[1] for size in all_sizes)
    length_diff = max_length.item() - local_size[1].item()
    if length_diff:
        pad_size = (*s.shape[:-1], length_diff)
        padding = torch.ones(pad_size, device=s.device, dtype=s.dtype) * pad_tok_id
        s = torch.concat((s, padding), dim = -1)
    gathered_s = [torch.ones_like(s)*pad_tok_id for _ in range(world_size)]
    dist.all_gather(gathered_s, s)

    return gathered_s

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", 'id'))
        input_ids = padding(input_ids, self.tokenizer.pad_token_id, cutoff = 256)
        labels = padding(labels, IGNORE_INDEX, cutoff = 256)

        return dict(
            input_ids=input_ids,
            labels=labels,
            id=torch.tensor(ids).to(input_ids.device),
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_path, max_sample) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_path, max_sample=max_sample)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return eval_dataset, data_collator


def main(rank, args):
    dist.init_process_group("nccl")
    torch.manual_seed(args.seed)
    world_size = torch.cuda.device_count()
    base_model = args.base_model
    data_path = args.data_path
    batch_size = args.batch_size
    
    n_gpus = torch.cuda.device_count()
    max_memory = f'80000MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    model = LlamaForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16
        )
        
    # model.half()
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model, use_fast=False)

    torch.cuda.set_device(rank)
    model.to(torch.cuda.current_device())
    model = DDP(model, device_ids=[torch.cuda.current_device()])
    model.eval()

    eval_dataset, data_collator = make_supervised_data_module(tokenizer, data_path, args.max_sample)
    # dataset_for_eval = load_dataset(data_path)['train']
    return_seq_num = 1
    for tempera in [0.7]:
        sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        dataloader = DataLoader(
            eval_dataset, 
            shuffle=False, 
            collate_fn=data_collator, 
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True
        )
        generation_config = GenerationConfig(
            # temperature=0.8 if args.diverse_beam > 1 else 1.0,
            temperature=tempera,
            # num_beam_groups=args.diverse_beam,
            # diversity_penalty=1.0,
            do_sample=True,
            num_beams=return_seq_num,
            max_new_tokens=512,
            num_return_sequences=return_seq_num,
            top_p=1.0
        )

        all_outputs = []
        for step, batch in tqdm(enumerate(dataloader), total=len(eval_dataset)):
            # if step > 10:
            #     break
            # print(batch.pop('id'))
            # print(dataset_for_eval[step]['prompt'])
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            # with autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                generation_output = model.module.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    synced_gpus=True,
                )
            s = generation_output.sequences
            gather_outputs = sequence_gather(s, world_size, tokenizer.pad_token_id)
            gathered_inputs = sequence_gather(input_ids, world_size, tokenizer.pad_token_id)
            gather_outputs = torch.stack(gather_outputs).reshape(world_size, batch_size,return_seq_num,-1)
            gathered_inputs = torch.stack(gathered_inputs)
            gather_outputs = gather_outputs.transpose(0,1).reshape(batch_size*world_size*return_seq_num, -1)
            gathered_inputs = gathered_inputs.transpose(0,1).reshape(batch_size*world_size,-1)
            # try:
            outputs_string = tokenizer.batch_decode(gather_outputs, skip_special_tokens=True)
            inputs_string = tokenizer.batch_decode(gathered_inputs, skip_special_tokens=True)
            
            # for item in range(len(gather_outputs)):
            #     if rank ==0:
            #         print(outputs_string[item])
            #         print(gather_outputs[item])
            #     input()

            # except:
            #     print(gather_outputs)
            #     print(gather_outputs.sum(-1))
            #     print(gather_outputs.shape)
            #     print(torch.max(gather_outputs), torch.min(gather_outputs))
            #     raise RuntimeError
            # if rank == 0: 
            #     print(inputs_string)
            #     print('+'*10)
            #     print(outputs_string)
            
            for idx in range(len(inputs_string)):
                temp = []
                for i in range(return_seq_num):
                    temp.append([inputs_string[idx], outputs_string[return_seq_num*idx+i].replace(inputs_string[idx], '')])
                    # if rank ==0:
                    #     print(temp[-1][1])
                    # input()
                all_outputs.append(temp)
            # input()
        if rank == 0:
            import json
            with open(args.out_path + f'/raw_generation_{tempera}_{args.seed}.json', 'w') as f:
                for item in all_outputs[:len(eval_dataset)]:
                    f.write(json.dumps(item) + '\n')
                    # json.dump(all_outputs[:len(eval_dataset)], f)
            json_path = args.out_path + f'/raw_generation_{tempera}_{args.seed}.json'
            print(f"save json to {json_path}")
            create_sft(json_path, args.out_path + "/gen_data.json")
        dist.barrier()
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--base_model", default="", type=str, help="model path")
    parser.add_argument("--data_path", default="", type=str, help="config path")
    parser.add_argument("--batch_size", type=int, default=0, help="batch size")
    parser.add_argument("--port", type=int, default=0, help="batch size")
    parser.add_argument("--diverse_beam", type=int, default=1, help="batch size")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument("--out_path", default="", type=str, help="config path")
    parser.add_argument("--gptq", type=str, default=None, help="batch size")
    parser.add_argument("--awq", type=str, default=None, help="awq")
    parser.add_argument('--bit_4', action='store_true', help='Quant to 4 bit')
    parser.add_argument('--quant_type', type=str, default='nf4', help='quantization type')
    parser.add_argument('--rtn', type=str, default=None, help='rtn')
    parser.add_argument("--max_sample", type=int, default=None, help="max_sample")
    parser.add_argument('--max_memory', type=str, nargs='*',
                    help="List of device_id:max_memory pairs to be parsed into a dictionary; " \
                        + "Example: 0:10GiB 1:10GiB cpu:30GiB; " \
                        + "mode details here: " \
                        + "https://huggingface.co/docs/accelerate/usage_guides/big_modeling")
    args = parser.parse_args()
    if not os.path.exists(args.out_path):
        try:
            os.makedirs(args.out_path)
            print(f"dir {args.out_path} create successfully")
        except:
            pass
    else:
        print(f"dir {args.out_path} has existed")
    local_rank = int(os.environ["LOCAL_RANK"])
    # from ppq.api import ENABLE_CUDA_KERNEL
    # with ENABLE_CUDA_KERNEL():
    main(local_rank, args)
