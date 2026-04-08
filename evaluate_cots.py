import os
import json
import argparse
from dotenv import load_dotenv
from vlmeval.config import supported_VLM
import warnings
warnings.filterwarnings('ignore')
import tqdm


load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="VLM Evaluation Script")
    parser.add_argument('--model', type=str, required=True, help="Model name from the supported_VLM dictionary")
    parser.add_argument('--prepend_text', action='store_true', help='Include prepend text if available')
    parser.add_argument('--cat', type=str, choices=['pattern','matching','minutiae','sensor','orientation','realvssyn','ace','tools','all'], required=True, help='Category of questions')
    args = parser.parse_args()

    model_names = args.model

    for model_name in [model_names]:

        if model_name not in supported_VLM:
            print(f"Model '{model_name}' is not supported.")
            return

        json_dir = './benchmark/'

        output_dir = os.path.join('./results_test_cot', model_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"Evaluating model: {model_name}")
        model = supported_VLM[model_name]()
        print("Model Loaded")
        all = []
        if args.cat == 'pattern':
            jsons = ['pattern_single_all_data_cot.json','pattern_multiple_all_data_cot.json']
        elif args.cat == 'matching':
            jsons = ['matching_multiple_all_data_cot.json']
        elif args.cat == 'minutiae':
            jsons = ['minutiae_multiple_min_all_data_cot.json']
        elif args.cat == 'sensor':
            jsons = ['sensor_single_all_data_cot.json','sensor_multiple_all_data_cot.json']
        elif args.cat == 'realvssyn':
            jsons = ['realvssyn_multiple_alldata_cot.json']
        elif args.cat == 'orientation':
            jsons = ['orientation_single_or_all_data_cot.json','orientation_pattern_single_or_all_data_cot.json']
        elif args.cat == 'ace':
            jsons = ['ace_multiple_all_data_final_cot.json']
        elif args.cat == 'tools':
            jsons = ['tools_retrieval_cot.json']
        else:
            jsons = ['pattern_single_all_data_cot.json','pattern_multiple_all_data_cot.json','matching_multiple_all_data_cot.json','minutiae_single_min_all_data_cot.json','minutiae_multiple_min_all_data_cot.json','sensor_single_all_data_cot.json','sensor_multiple_all_data_cot.json','realvssyn_multiple_alldata_cot.json','orientation_single_or_all_data_cot.json','orientation_pattern_single_or_all_data_cot.json','ace_multiple_all_data_final_cot.json', 'tools_retrieval_cot.json']
            
            

        
        for json_file in jsons:
            output_path = os.path.join(output_dir, json_file.split('/')[-1])

            print(f"Processing JSON file: {json_file}")
            json_path = os.path.join(json_dir, json_file)
            with open(json_path, 'r') as f:
                data = json.load(f)

            correct_answers = 0
            total_questions = len(data['questions'])
            option_labels = ['A', 'B', 'C', 'D'] 
            if 'ace' in json_file: option_labels = ['A', 'B', 'C'] 

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
                for idx, option in enumerate(question['options']):
                    if idx >= len(option_labels):
                        print(f"Warning: More options than labels available for question {q_id}")
                        break
                    options_text += f"({option_labels[idx]}) {option}\n"

                question_text += options_text

                question_input = question['image_paths'] + [question_text]
                

                
                try:
                    ret = model.generate(question_input, dataset="MCQ")
                    model_output = ret.strip()
                    model_answer = model_output
                    question['prediction'] = model_answer

                    
                except Exception as e:
                    print(e)
                    with open(output_path, 'w') as f:
                        json.dump(data, f, indent=4)
                

            with open(output_path, 'w') as f:
                json.dump(data, f, indent=4)

if __name__ == '__main__':
    main()