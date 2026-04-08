import os
import json
import argparse
import tiktoken
from PIL import Image
import math
import tqdm


import base64
from openai import OpenAI

client = OpenAI(api_key='YOUR_API_KEY_HERE')

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


json_dir = './benchmark'

parser = argparse.ArgumentParser(description="VLM Evaluation Script")
parser.add_argument('--model', type=str, default="gpt-5", help="Model name from the supported_VLM dictionary")
parser.add_argument('--prepend_text', action='store_true', help='Include prepend text if available')
parser.add_argument('--cat', type=str, choices=['pattern','matching','minutiae','sensor','orientation','realvssyn','ace','tools','all'], required=True, help='Category of questions')
args = parser.parse_args()


output_dir = os.path.join('./results_test', 'gpt-5')
os.makedirs(output_dir, exist_ok=True)

if args.cat == 'pattern':
    jsons = ['pattern_single_all_data.json','pattern_multiple_all_data.json']
elif args.cat == 'matching':
    jsons = ['matching_multiple_all_data.json']
elif args.cat == 'minutiae':
    jsons = ['minutiae_single_min_all_data.json','minutiae_multiple_min_all_data.json']
elif args.cat == 'sensor':
    jsons = ['sensor_single_all_data.json','sensor_multiple_all_data.json']
elif args.cat == 'realvssyn':
    jsons = ['realvssyn_multiple_alldata.json']
elif args.cat == 'orientation':
    jsons = ['orientation_single_or_all_data.json','orientation_pattern_single_or_all_data.json']
elif args.cat == 'ace':
    jsons = ['ace_multiple_all_data_final.json']
elif args.cat == 'tools':
    jsons = ['tools_retrieval.json']
else:
    jsons = ['pattern_single_all_data.json','pattern_multiple_all_data.json','matching_multiple_all_data.json','minutiae_single_min_all_data.json','minutiae_multiple_min_all_data.json','sensor_single_all_data.json','sensor_multiple_all_data.json','realvssyn_multiple_alldata.json','orientation_single_or_all_data.json','orientation_pattern_single_or_all_data.json','ace_multiple_all_data_final.json','tools_retrieval.json']

for json_file in jsons:
    output_path = os.path.join(output_dir, json_file.split('/')[-1])
    
    print(f"Processing JSON file: {json_file}")
    json_path = os.path.join(json_dir, json_file)
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    option_labels = ['A', 'B', 'C', 'D'] 

    for q_id, question in tqdm.tqdm(data['questions'].items()):
        question_text = ''
        if data["category"] == "tools_use":
            question_text += data['context']

        if data.get('prepend_text'):
            question_text += data['prepend_text']
        
        question_text += question['question_text']+" "

        if data.get('postpend_text'):
            question_text += data['postpend_text'] + " No inner monologue."
            if data["category"] == "tools_use":
                question_text = question_text.replace("Ignore the black image provided.\n", "")

        options_text = ''
        for idx, option in enumerate(question['options']):
            if idx >= len(option_labels):
                print(f"Warning: More options than labels available for question {q_id}")
                break
            options_text += f"({option_labels[idx]}) {option} "

        question_text += options_text

        messages = [{
            "role": "system",
            "content": [{"type": "input_text", "text": "You are an expert fingerprint examiner"}]
        },
        {
            "role": "user",
            "content": []
        }]
            
        if 'tool' not in json_file:
            for img_path in question['image_paths']:
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                messages[1]['content'].append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"})
            
        messages[1]['content'].append({"type": "input_text", "text": question_text})
        
        try:
            response = client.responses.create(
            model="gpt-5",
            input=messages,
            max_output_tokens=32,
            reasoning= {
            "effort": "minimal"}
            )
            question['prediction'] = response.output_text
            
        except Exception as e:
            print(e)
            break
        
        
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

