import json
import re
from tqdm import tqdm

def create_sft(data_path, out_path):
    data_path=data_path

    with open(data_path, 'r') as f:
        dataset_for_eval = f.readlines()

    match_str = "Below is an instruction that describes a task. Write a response"
    instructions = []
    responses = []
    for line in dataset_for_eval:
        json_data = json.loads(line)
        if json_data and isinstance(json_data[0], list):
            if match_str in json_data[0][0]:
                instructions.append(json_data[0][0])
                responses.append(json_data[0][1])

    instruction = []
    for i in range(0,len(instructions)):
        pattern = r'### Instruction:\n(.*?)\n\n### Response:'
        match = re.search(pattern, instructions[i], re.DOTALL)

        if match:
            instruction_content = match.group(1)
            instruction.append(instruction_content)
        else:
            raise 1

    print(len(instruction))
    print(len(responses))

    # print(instruction[1])
    # print(responses[1])
    with open(out_path, 'w') as f:
        for idx in range(len(instruction)):
            f.write(json.dumps({'query':instruction[idx], 'response':responses[idx]}).strip() + "\n")

# create_sft()