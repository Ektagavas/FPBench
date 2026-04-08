import os
import json
import base64
from io import BytesIO
from google import genai
from google.genai import types
from PIL import Image
import math
import argparse
import tqdm


try:
    client = genai.Client(api_key="YOUR_API_KEY_HERE")
except Exception as e:
    print(f"Error initializing Gemini client. Ensure GEMINI_API_KEY is set. Error: {e}")
    exit()


GEMINI_MODEL = "gemini-2.5-pro"
SYSTEM_INSTRUCTION_TEXT = "You are an expert fingerprint examiner"


# Function to read the image file and convert it to a Part object for Gemini
def file_to_part(image_path, mime_type="image/jpeg"):
    """Reads a local file and returns a types.Part object."""
    try:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
        return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    except Exception as e:
        print(f"Error reading image file {image_path}: {e}")
        return None


json_dir = './benchmark' 

parser = argparse.ArgumentParser(description="VLM Evaluation Script")
parser.add_argument('--model', type=str, default="GeminiPro2-5", help="Model name from the supported_VLM dictionary")
parser.add_argument('--prepend_text', action='store_true', help='Include prepend text if available')
parser.add_argument('--cat', type=str, choices=['pattern','matching','minutiae','sensor','orientation','realvssyn','ace','tools','all'], required=True, help='Category of questions')
args = parser.parse_args()


output_dir = os.path.join('./results_1104', args.model)
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
    jsons = ['pattern_single_all_data.json','pattern_multiple_all_data.json','matching_multiple_all_data.json','minutiae_single_min_all_data.json','minutiae_multiple_min_all_data.json','sensor_single_all_data.json','sensor_multiple_all_data.json','realvssyn_multiple_alldata.json','orientation_single_or_all_data.json','orientation_pattern_single_or_all_data.json','ace_multiple_all_data_final.json']


for json_file in jsons:
    print(f"Processing JSON file: {json_file}")
    json_path = os.path.join(json_dir, json_file)
    with open(json_path, 'r') as f:
        data = json.load(f)

    output_path = os.path.join(output_dir, json_file.split('/')[-1])
    
    option_labels = ['A', 'B', 'C', 'D'] 

    for q_id, question in tqdm.tqdm(data['questions'].items()):
        
        question_text = ''
        if data["category"] == "tools_use":
            question_text += data['context'] + '\n'

        if data.get('prepend_text'):
            question_text += data['prepend_text'] + '\n'
        
        question_text += question['question_text'] + '\n'

        
        if data.get('postpend_text'):
            p = "Provide concise reasoning steps in brief before giving the final answer. Include the final correct answer option e.g., A, B, C, D at the end of your answer. No inner monologue."
            question_text += p + '\n'

        options_text = ''
        for idx, option in enumerate(question['options']):
            if idx >= len(option_labels):
                print(f"Warning: More options than labels available for question {q_id}")
                break
            options_text += f"({option_labels[idx]}) {option}\n"

        question_text += options_text
        contents = []
        
        if 'tools' not in json_file:
            for img_path in question['image_paths']:
                image_part = file_to_part(img_path, mime_type="image/jpeg") 
                if image_part:
                    contents.append(image_part)
        
        contents.append(question_text)
        
        generation_config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION_TEXT,
            thinking_config=types.ThinkingConfig(thinking_budget=128),
            temperature=0.,
            maxOutputTokens=136
        )
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=generation_config
            )
            
            question['prediction'] = response.text

        except Exception as e:
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"An error occurred during API call for question {q_id}: {e}")
            break 
        
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

