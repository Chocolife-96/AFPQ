import os
import sys, random

import argparse
import torch
import transformers
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional, Dict, Sequence, List
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import json
from torch.cuda.amp import autocast
from tqdm import tqdm
import copy

DEV = "cuda:0"

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: Let's think step by step."
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{query}\n\n### Response: Let's think step by step."
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

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, shard: int):
        super(SupervisedDataset, self).__init__()

        # dataset_for_eval = load_dataset(data_path)['train']

        
        with open(data_path, 'r') as f:
            dataset_for_eval = f.readlines()
        
        # stride = len(dataset_for_eval) // 8 + 1
        # dataset_for_eval = dataset_for_eval[stride*shard: stride*(shard+1)]
        # print(f'shard from {stride*shard} to {stride*(shard+1)}')
        print(f'eval data number {len(dataset_for_eval)}')
        dataset_for_eval = [json.loads(item.strip()) for item in dataset_for_eval]
        sources = [PROMPT_DICT["prompt_no_input"].format_map(item) for item in dataset_for_eval]
        targets = [item['response'] for item in dataset_for_eval]
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
        cutoff = min(cutoff, max([len(item) for item in inputs]))
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


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_path, shard) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_path, shard=shard)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return eval_dataset, data_collator


def main(rank, args):

    # dist.init_process_group("nccl")
    # world_size = torch.cuda.device_count()
    base_model = args.base_model
    data_path = args.data_path
    batch_size = args.batch_size

    if args.gptq:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        model = AutoGPTQForCausalLM.from_quantized(args.gptq, device="cuda:0", use_safetensors=True, use_triton=False, 
        inject_fused_attention=False, inject_fused_mlp=False) 
    else:
        model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                load_in_4bit=args.bit_4,
                device_map='auto',
                quantization_config=BitsAndBytesConfig(
                    bnb_4bit_quant_type=args.quant_type,
                    load_in_4bit=args.bit_4,
                    bnb_4bit_compute_dtype=torch.bfloat16
                ),
            )
        if args.rtn is not None:
            from rtn import pseudo_quantize_model_weight
            q_config = {
                "zero_point": True,  # by default True
                "q_group_size": 128,  # whether to use group quantization
            }
            pseudo_quantize_model_weight(
                model, w_bit=3, q_config=q_config, quant_type=args.rtn, model_path=base_model
            )
            
        elif args.awq:
            q_config = {
                "zero_point": True,  # by default True
                "q_group_size": 128,  # whether to use group quantization
            }
            import sys
            sys.path.append("/root/model/llm-awq")
            from awq.quantize.pre_quant import run_awq, apply_awq
            from awq.quantize.quantizer import pseudo_quantize_model_weight, pseudo_quantize_n2f3_tensor
            from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model, load_checkpoint_in_model
            print("Loading pre-computed AWQ results from", args.awq)
            awq_results = torch.load(args.awq, map_location="cpu")
            apply_awq(model, awq_results)
            from rtn import pseudo_quantize_model_weight
            pseudo_quantize_model_weight(
                model, w_bit=3, q_config=q_config, quant_type="n2f3"
            )
            # pseudo_quantize_model_weight(
            #     model, w_bit=3, q_config=q_config
            # )
            max_memory = [v.split(':') for v in (args.max_memory or [])]
            max_memory = {(int(k) if k.isdigit() else k):v for k,v in max_memory}
            kwargs = {"max_memory": max_memory} if len(max_memory) else {}
            device_map = infer_auto_device_map(
                model,
                # TODO: can we remove this?
                no_split_module_classes=[
                    "OPTDecoderLayer", "LlamaDecoderLayer", "BloomBlock", "MPTBlock", "DecoderLayer"],
                **kwargs
            )
            model = dispatch_model(model, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    tokenizer.truncation_side = 'left'

    # torch.cuda.set_device(rank)
    # model = DDP(model, device_ids=[torch.cuda.current_device()])
    model.eval()

    if args.seed_range == 100:
        start = 0
        end = 100
    else:
        start = args.seed_range
        end = min(args.seed_range + 10, 100)

    for seed in range(start, end):
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)

        eval_dataset, data_collator = make_supervised_data_module(tokenizer, data_path, shard=args.test_shard)
        return_seq_num = 1
        tempera = args.tempera

        if 'train' in args.data_path:
            sample_set = 'train'
        else:
            sample_set = 'test'
        
        if args.do_sample and os.path.exists(args.out_path + f'/raw_generation_{tempera}sampled_on_{sample_set}_seed_{seed}_shard_{args.test_shard}.json'):
            continue
        if not args.do_sample and os.path.exists(args.out_path + f'/raw_generation_greedy_on_{sample_set}_shard_{args.test_shard}.json'):
            print(args.out_path + f'/raw_generation_greedy_on_{sample_set}_shard_{args.test_shard}.json')
            continue

        print('working on......')
        if args.do_sample:
            print(args.out_path + f'/raw_generation_{tempera}sampled_on_{sample_set}_seed_{seed}_shard_{args.test_shard}.json')
        if not args.do_sample:
            print(args.out_path + f'/raw_generation_greedy_on_{sample_set}_shard_{args.test_shard}.json')
        print('........')
            
        # sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        
        dataloader = DataLoader(
            eval_dataset, 
            shuffle=False, 
            collate_fn=data_collator, 
            batch_size=batch_size,
            drop_last=True,
        )
        generation_config = GenerationConfig(
            temperature=tempera,
            do_sample=args.do_sample,
            num_beams=return_seq_num,
            max_new_tokens=512,
            num_return_sequences=return_seq_num,
        )
        
        all_outputs = []
        for step, batch in tqdm(enumerate(dataloader)):
            # if step > 10:
            #     break
            # print(batch.pop('id'))
            # print(dataset_for_eval[step]['prompt'])
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            with torch.no_grad():
                with autocast(dtype=torch.bfloat16): 
                    generation_output = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        generation_config=generation_config,
                        return_dict_in_generate=True,
                    )
            s = generation_output.sequences
            outputs_string = tokenizer.batch_decode(s, skip_special_tokens=True)
            inputs_string = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            
            for idx in range(len(inputs_string)):
                temp = []
                for i in range(return_seq_num):
                    temp.append([inputs_string[idx], outputs_string[return_seq_num*idx+i].replace(inputs_string[idx], '')])
                all_outputs.append(temp)


        print('finish......')
        if args.do_sample:
            print(args.out_path + f'/raw_generation_{tempera}sampled_on_{sample_set}_seed_{seed}_shard_{args.test_shard}.json')
        if not args.do_sample:
            print(args.out_path + f'/raw_generation_greedy_on_{sample_set}_shard_{args.test_shard}.json')
        print('........')

        import json
        if args.do_sample:
            
            with open(args.out_path + f'/raw_generation_{tempera}sampled_on_{sample_set}_seed_{seed}_shard_{args.test_shard}.json', 'w') as f:
                for item in all_outputs:
                    f.write(json.dumps(item) + '\n')
        
        else:
            
            with open(args.out_path + f'/raw_generation_greedy_on_{sample_set}_shard_{args.test_shard}.json', 'w') as f:
                for item in all_outputs:
                    f.write(json.dumps(item) + '\n')
            break
    # dist.barrier()
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--base_model", default="", type=str, help="model path")
    parser.add_argument("--data_path", default="", type=str, help="config path")
    parser.add_argument("--batch_size", type=int, default=0, help="batch size")
    parser.add_argument("--port", type=int, default=0, help="batch size")
    parser.add_argument("--diverse_beam", type=int, default=1, help="batch size")
    parser.add_argument("--use_diverse_beam", type=bool, default=False, help="batch size")
    parser.add_argument("--out_path", default="", type=str, help="config path")
    parser.add_argument("--do_sample", default=False, type=bool, help="config path")
    parser.add_argument("--test_shard", default=0, type=int, help="test shard")
    parser.add_argument("--seed", default=0, type=int, help="test shard")
    parser.add_argument("--seed_range", default=100, type=int, help="test shard")
    parser.add_argument("--tempera", type=float, default=0.2, help="batch size")
    parser.add_argument("--gptq", type=str, default=None, help="batch size")
    parser.add_argument("--awq", type=str, default=None, help="awq")
    parser.add_argument('--bit_4', action='store_true', help='Quant to 4 bit')
    parser.add_argument('--quant_type', type=str, default='nf4', help='quantization type')
    parser.add_argument('--rtn', type=str, default=None, help='rtn')
    parser.add_argument('--max_memory', type=str, nargs='*',
                    help="List of device_id:max_memory pairs to be parsed into a dictionary; " \
                        + "Example: 0:10GiB 1:10GiB cpu:30GiB; " \
                        + "mode details here: " \
                        + "https://huggingface.co/docs/accelerate/usage_guides/big_modeling")
    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
        print(f"dir {args.out_path} create successfully")
    else:
        print(f"dir {args.out_path} has existed")
    # print(args.do_sample)

    # local_rank = int(os.environ["LOCAL_RANK"])
    main(-1, args)

