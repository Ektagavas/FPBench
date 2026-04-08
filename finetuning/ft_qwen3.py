import os
import json
import argparse
import torch
from tqdm import tqdm
from unsloth.trainer import UnslothVisionDataCollator
from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig


def main():
    parser = argparse.ArgumentParser(description="VLM Evaluation Script")
    parser.add_argument('--model', type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="Model name from the supported_VLM dictionary")
    parser.add_argument('--prepend_text', action='store_true', help='Include prepend text if available')
    parser.add_argument('--vision', action='store_true', help='Finetune vision layers')
    parser.add_argument('--llm', action='store_true', help='Finetune language layers')
    parser.add_argument('--output', type=str, default="qwen3_pattern")
    args = parser.parse_args()

    model_name = args.model

    json_dir = './ft_json'

    output_dir = os.path.join('./results_0301', model_name.split('/')[-1])
    os.makedirs(output_dir, exist_ok=True)
    print(output_dir)
    
    
    print(f"Evaluating model: {model_name}")
    
    if torch.cuda.get_device_capability()[0] >= 8:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16

    print("Model Loaded")

    # Change dataset here
    jsons = ['pattern_mixed_all_data.json'] 
    # jsons = ['matching_multiple_all_data.json']
    # jsons = ['minutiae_mixed_min_all_data.json'] 
    # jsons = ['realvssyn_multiple_alldata.json'] 
    # jsons = ['ace_multiple_all_data_final.json'] 
    # jsons = ['orientation_mixed_single_or_all_data.json']
    # jsons = ['sensor_mixed_all_data.json']

    dataset = []   
    
    for json_file in jsons:
        print(f"Processing JSON file: {json_file}")
        json_path = os.path.join(json_dir, json_file)
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
            },
            {
                "role": "assistant",
                "content": [],
            },]
             
            
            for img_path in question['image_paths']:
                messages[1]['content'].append({"type": "image", "image": Image.open(img_path).convert('RGB')})

            messages[2]['content'].append({"type": "text", "text": question['correct_answer_option']})
            

            question_input = question['image_paths'] + [question_text]
            messages[1]['content'].append({"type": "text", "text": question_text})
            dataset.append({"messages":messages})
            

    
    max_seq_length = 16384 
    lora_rank = 16 

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = "unsloth/Qwen3-VL-8B-Instruct",
        max_seq_length = max_seq_length,
        load_in_4bit = True,
        fast_inference = False,
        gpu_memory_utilization = 0.8, # Reduce if out of memory
    )

    model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = args.vision, # False if not finetuning vision layers
    finetune_language_layers   = args.llm,  # False if not finetuning language layers
    finetune_attention_modules = False,  # False if not finetuning attention layers
    finetune_mlp_modules       = True,  # False if not finetuning MLP layers
    r = 16, 
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False, 
    loftq_config = None, 
    use_gradient_checkpointing = "unsloth", # Reduces memory usage
    # target_modules = "all-linear",
    )

    FastVisionModel.for_training(model)

    dir_name = args.output

    trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
    train_dataset = dataset,
    args = SFTConfig(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = 1,
            learning_rate = 1e-4,
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.001,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = dir_name,
            report_to = "none",
            remove_unused_columns = False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
            max_length = 2048,
        ),
    )

    trainer_stats = trainer.train()
    
            

if __name__ == '__main__':
    main()