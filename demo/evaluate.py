import os
import json
import argparse
from vlmeval.config import supported_VLM
import tqdm
import torch



def main():
    parser = argparse.ArgumentParser(description="VLM Evaluation Script")
    parser.add_argument('--model', type=str, required=True, help="Model name from the supported_VLM dictionary")
    parser.add_argument('--prepend_text', action='store_true', help='Include prepend text if available')
    args = parser.parse_args()

    model_name = args.model

    if model_name not in supported_VLM:
        print(f"Model '{model_name}' is not supported.")
        return

    json_dir = './sample_benchmark/'
    

    output_dir = os.path.join('./results', model_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Evaluating model: {model_name}")
    model = supported_VLM[model_name]()
    print("Model Loaded")
    
    
    jsons = ['pattern_single_all_data.json','pattern_multiple_all_data.json']

    for json_file in jsons:
        print(f"Processing JSON file: {json_file}")
        json_path = os.path.join(json_dir, json_file)
        with open(json_path, 'r') as f:
            data = json.load(f)

        correct_answers = 0
        total_questions = len(data['questions'])
        option_labels = ['A', 'B', 'C', 'D'] 
        if 'ace' in json_file: option_labels = ['A', 'B', 'C'] 

        output_path = os.path.join(output_dir, json_file.split('/')[-1])

        for q_id, question in tqdm.tqdm(data['questions'].items()):
            question_text = ''
            if data["category"] == "tools_use":
                question_text += data['context'] + '\n'

            if args.prepend_text and data.get('prepend_text'):
                question_text += data['prepend_text'] + '\n'

            question_text += question['question_text'] + '\n'

            if data.get('postpend_text'):
                question_text += data['postpend_text'] + '\n'

            options_text = ''
            
            num_options = len(question['options'])
            option_labels = [chr(ord('A') + i) for i in range(num_options)]

            for idx, option in enumerate(question['options']):
                if idx >= len(option_labels):
                    print(f"Warning: More options than labels available for question {q_id}")
                    break
                options_text += f"({option_labels[idx]}) {option}\n"
                if question["answer"] == option:
                    question["correct_answer_option"] = option_labels[idx]

            question_text += options_text
            question_input = question['image_paths'] + [question_text]
            
            try:
                torch.cuda.empty_cache()
                ret = model.generate(question_input, dataset="MCQ")
                model_output = ret.strip()
                model_answer = model_output
                question['prediction'] = model_answer
            except Exception as e:
                print(e)
            
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)

if __name__ == '__main__':
    main()