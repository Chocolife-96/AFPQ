import argparse
import pprint
import sys
import os
import re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from human_eval.data import write_jsonl, read_problems, stream_jsonl
from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model, load_checkpoint_in_model
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

def generate_prompt(input):
    INSTRUCTION = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.


### Instruction:
Create a Python script for this problem:
{input}

### Response:"""
    return INSTRUCTION

def get_model(
    load_8bit: bool = False,
    bit_4: bool = False,
    rtn: str = None,
    gptq: str = None,
    awq: str = None,
    quant_type: str = None,
    group_size: int = 128,
    bits: int = 4,
    base_model: str = "bigcode/starcoder",
):
    assert base_model, (
        "Please specify a --base_model, e.g. --base_model='bigcode/starcoder'"
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if device == "cuda":
        if gptq:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
            dev = torch.cuda.current_device()
            # print("current dev", dev)
            model = AutoGPTQForCausalLM.from_quantized(gptq, device=dev, use_safetensors=True, inject_fused_attention=False)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                load_in_4bit=bit_4,
                quantization_config=BitsAndBytesConfig(
                    bnb_4bit_quant_type=quant_type,
                    load_in_4bit=bit_4,
                    bnb_4bit_compute_dtype=torch.bfloat16
                ),
            )
        model.config.pad_token_id = tokenizer.pad_token_id
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    

    if rtn is not None:
        from rtn import pseudo_quantize_model_weight
        q_config = {
            "zero_point": True,  # by default True
            "q_group_size": group_size,  # whether to use group quantization
        }
        pseudo_quantize_model_weight(
            model, w_bit=bits, q_config=q_config, quant_type=rtn, model_path=base_model
        )
    elif awq is not None:
        sys.path.append("../llm-awq")
        q_config = {
            "zero_point": True,  # by default True
            "q_group_size": group_size,  # whether to use group quantization
        }
        from awq.quantize.pre_quant import run_awq, apply_awq
        from rtn import pseudo_quantize_model_weight
        print("Loading pre-computed AWQ results from", awq)
        awq_results = torch.load(awq, map_location="cpu")
        apply_awq(model, awq_results)

        pseudo_quantize_model_weight(
            model, w_bit=bits, q_config=q_config, quant_type=quant_type
        )
        max_memory = [v.split(':') for v in []]
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
    

    # if not load_8bit and not bit_4:
    #     model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    return tokenizer, model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='bigcode/starcoder', help="")
    parser.add_argument('--output_path', type=str, help="")
    parser.add_argument('--start_index', type=int, default=0, help="")
    parser.add_argument('--end_index', type=int, default=164, help="")
    parser.add_argument('--temperature', type=float, default=0.8, help="")
    parser.add_argument('--N', type=int, default=200, help="")
    parser.add_argument('--max_len', type=int, default=512, help="")
    parser.add_argument('--decoding_style', type=str, default='sampling', help="")
    parser.add_argument('--num_seqs_per_iter', type=int, default=50, help='')
    parser.add_argument('--greedy_decode', action='store_true', help='')
    parser.add_argument('--overwrite', action='store_true', help='')
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument('--rtn', type=str, default=None, help='rtn')
    parser.add_argument('--gptq', type=str, default=None)
    parser.add_argument('--awq', type=str, default=None)
    parser.add_argument('--bit_4', action='store_true', help='Quant to 4 bit')
    parser.add_argument('--quant_type', type=str, default='n2f3', help='quantization type')
    parser.add_argument('--bits', type=int, default=4, help="")
    parser.add_argument('--group_size', type=int, default=128, help="")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    problems = read_problems()

    task_ids = sorted(problems.keys())[args.start_index: args.end_index]
    prompts = [problems[task_id]['prompt'] for task_id in task_ids]
    num_samples = len(prompts)
    print("Number of samples: {}".format(num_samples))

    tokenizer, model = get_model(base_model=args.model, rtn=args.rtn, gptq=args.gptq, awq=args.awq, bit_4=args.bit_4, quant_type=args.quant_type, group_size=args.group_size, bits=args.bits)
    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False if args.greedy_decode else True,
        temperature=args.temperature,
        max_length=args.max_len,
        num_return_sequences=args.num_seqs_per_iter,
        eos_token_id=tokenizer.eos_token_id,
        top_p=0.95
    )

    print(f"Loaded {args.model}.")
    for i in tqdm(range(num_samples), ncols=0, total=num_samples):
        output_file = args.output_path + '/{}.jsonl'.format(args.start_index + i)

        if os.path.exists(output_file) and not args.overwrite:
            print(f'Skip {output_file} as it already exists')
            continue

        prompt = prompts[i].replace('    ', '\t')
        prompt_batch = [generate_prompt(prompt)]

        ids_batch = [task_ids[i]]

        completion_seqs = []

        encoding = tokenizer(prompt_batch, return_tensors="pt", truncation=True, max_length=args.max_len).to(device)

        if args.decoding_style == 'sampling':
            loops = int(args.N / args.num_seqs_per_iter)
        else:
            loops = 1

        for _ in tqdm(range(loops), total=loops, leave=False, ncols=0):

            with torch.no_grad():
                try:
                    gen_tokens = model.generate(
                        **encoding,
                        generation_config=generation_config
                    )
                except:
                    gen_tokens = None

            if gen_tokens is not None:
                gen_seqs = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            else:
                gen_seqs = None

            if gen_seqs is not None:
                assert len(ids_batch) == 1
                task_id = ids_batch[0]

                for seq_idx, gen_seq in enumerate(gen_seqs):
                    completion_seq = gen_seq.split("### Response:")[1]
                    completion_seq = completion_seq.replace('\t', '    ')
                    all_code = gen_seq.replace('\t', '    ')

                    completion_seqs.append(
                        {'task_id': task_id,
                         'completion': completion_seq,
                         'all_code': all_code,
                         }
                    )

        print("Saving results to {}".format(output_file))
        write_jsonl(output_file, completion_seqs)


if __name__ == '__main__':
    main()