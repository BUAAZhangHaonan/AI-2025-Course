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
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='给JSONL文件的instruction字段添加/no_think')
    parser.add_argument('--input', required=True, help='输入JSONL文件路径')
    parser.add_argument('--output', required=True, help='输出JSONL文件路径')
    args = parser.parse_args()
    

    add_no_think(args.input, args.output)
