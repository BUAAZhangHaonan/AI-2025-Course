import json
import argparse

def modify_instructions(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            print("1")

            new_instruction = ""
            if 'instruction' in data:
                instruction = data['instruction']

                target = "answer the following question:"
                idx = instruction.find(target)
                if idx != -1:

                    Question = "Question:" + instruction[idx + len(target):]
                    new_instruction = (
                        Question + 
                        "\nPlease provide only the final answer, wrapped in <answer> tags  /no_think\n"
                    )
                    data['instruction'] = new_instruction
            if "input" in data:
                data['instruction'] = "Documents: " + data['input'] + "\n"
            data['prompt'] = data['instruction'] + new_instruction + "Answer:"
            del data['instruction']
            del data['input']

            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"处理完成，结果已保存至 {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='修改jsonl文件中的instruction字段')
    parser.add_argument('--input', type=str, required=True, help='输入jsonl文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出jsonl文件路径')
    args = parser.parse_args()
    
    modify_instructions(args.input, args.output)