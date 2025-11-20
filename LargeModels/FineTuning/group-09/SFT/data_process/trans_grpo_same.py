import json
import argparse

def add_no_think(input_file, output_file):

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            data = json.loads(line.strip())
            
            if "instruction" in data:
                if isinstance(data["instruction"], str):
                    if not data["instruction"].strip().endswith("/no_think"):
                        data["instruction"] = f"{data['instruction'].rstrip()} /no_think"
            
            if "input" in data:
                if isinstance(data["input"], str):
                   
                    data["input"] = f"{data['input'].rstrip()}\nPlease provide only the final answer, wrapped in <answer> tags.\nAnswer:"
            data['prompt'] = data['instruction'] + '\n' + data['input']
            del data['instruction']
            del data['input']
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

    
    print(f"处理完成，结果已保存到 {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='修改jsonl文件中的instruction字段')
    parser.add_argument('--input', type=str, required=True, help='输入jsonl文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出jsonl文件路径')
    args = parser.parse_args()
    
    add_no_think(args.input, args.output)