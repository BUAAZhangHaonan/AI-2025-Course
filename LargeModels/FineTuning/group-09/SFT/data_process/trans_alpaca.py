import json
from typing import List, Dict

def convert_to_alpaca(original_data: Dict) -> Dict:

    documents_combined = "\n\n".join(original_data.get("documents", []))

    answer_combined = "; ".join(original_data.get("answer", []))
    question = original_data.get("question", "")
    instruction = f"Based on the provided reference documents, answer the following question: {question}"  #
    # 构建Alpaca格式数据
    alpaca_format = {
        "instruction": instruction,
        "input": documents_combined,
        "output": answer_combined
    }
    
    return alpaca_format

def process_json_file(input_file: str, output_file: str) -> None:
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            # 解析原始数据
            original_data = json.loads(line)
            # 转换为Alpaca格式
            alpaca_data = convert_to_alpaca(original_data)
            json.dump(alpaca_data, f_out, ensure_ascii=False)
            f_out.write('\n')

if __name__ == "__main__":
    input_filename = "test_top5.json"
    output_filename = "test_top5_alpaca.json" 
    
    process_json_file(input_filename, output_filename)
    print(f"转换完成，结果已保存到 {output_filename}")
