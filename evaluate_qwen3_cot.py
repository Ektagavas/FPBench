import os
import json
import argparse
import torch
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig



def main():
    parser = argparse.ArgumentParser(description="VLM Evaluation Script")
    parser.add_argument('--model', type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="Model name from the supported_VLM dictionary")
    parser.add_argument('--prepend_text', action='store_true', help='Include prepend text if available')
    parser.add_argument('--cat', type=str, choices=['pattern','matching','minutiae','sensor','orientation','realvssyn','ace','tools','all'], required=True, help='Category of questions')
    args = parser.parse_args()

    model_name = args.model

    json_dir = './benchmark'
    
    output_dir = os.path.join('./results_test_cot', model_name.split('/')[-1])
    os.makedirs(output_dir, exist_ok=True)
    print(output_dir)

    
    print(f"Evaluating model: {model_name}")
    
    if torch.cuda.get_device_capability()[0] >= 8:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16
    
    model_kwargs = dict(
        attn_implementation="eager", # Use "flash_attention_2" when running on Ampere or newer GPU
        dtype=torch_dtype,
        device_map="auto",
    )

    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=model_kwargs['dtype'],
        bnb_4bit_quant_storage=model_kwargs['dtype'],
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        **model_kwargs
        )
    
    model.eval()

    processor = AutoProcessor.from_pretrained(model_name)

    print("Model Loaded")

    all = []
    if args.cat == 'pattern':
        jsons = ['pattern_single_all_data_cot.json','pattern_multiple_all_data_cot.json']
    elif args.cat == 'matching':
        jsons = ['matching_multiple_all_data_cot.json']
    elif args.cat == 'minutiae':
        jsons = ['minutiae_single_min_all_data_cot.json','minutiae_multiple_min_all_data_cot.json']
    elif args.cat == 'sensor':
        jsons = ['sensor_single_all_data_cot.json','sensor_multiple_all_data_cot.json']
    elif args.cat == 'realvssyn':
        jsons = ['realvssyn_multiple_alldata_cot.json']
    elif args.cat == 'orientation':
        jsons = ['orientation_single_or_all_data_cot.json','orientation_pattern_single_or_all_data_cot.json']
    elif args.cat == 'ace':
        jsons = ['ace_multiple_all_data_final_cot.json']
    elif args.cat == 'tools':
        jsons = ['tools_retrieval.json']
    else:
        jsons = ['pattern_single_all_data_cot.json','pattern_multiple_all_data_cot.json','matching_multiple_all_data_cot.json','minutiae_single_min_all_data_cot.json','minutiae_multiple_min_all_data_cot.json','sensor_single_all_data_cot.json','sensor_multiple_all_data_cot.json','realvssyn_multiple_alldata_cot.json','orientation_single_or_all_data_cot.json','orientation_pattern_single_or_all_data_cot.json','ace_multiple_all_data_final_cot.json',
        'tools_retrieval_cot.json']
        
    for json_file in jsons:
        print(f"Processing JSON file: {json_file}")
        
        json_path = os.path.join(json_dir, json_file)
        output_path = os.path.join(output_dir, json_file.split('/')[-1])
        with open(json_path, 'r') as f:
            data = json.load(f)
        

        correct_answers = 0
        total_questions = len(data['questions'])
        option_labels = ['A', 'B', 'C', 'D'] 

        for q_id, question in tqdm(data['questions'].items()):
            
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

            messages = [{
                "role": "system",
                "content": [{"type": "text", "text": "You are an expert fingerprint examiner"}]
            },
            {
                "role": "user",
                "content": []
            }]
             
            for img_path in question['image_paths']:
                messages[1]['content'].append({"type": "image", "image": img_path})

            question_input = question['image_paths'] + [question_text]
            messages[1]['content'].append({"type": "text", "text": question_text})

            
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            
            inputs = inputs.to("cuda")

            torch.cuda.empty_cache()
            try:
                generated_ids = model.generate(**inputs, max_new_tokens=4096, do_sample=False, temperature=0.)
            except Exception as e:
                print(q_id)
                print(e)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            model_answer = output_text[0]
            question['prediction'] = model_answer

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)

if __name__ == '__main__':
    main()