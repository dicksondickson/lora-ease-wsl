import sys
import subprocess
from safetensors.torch import load_file
from diffusers import AutoPipelineForText2Image
from datasets import load_dataset
from huggingface_hub.repocard import RepoCard
from huggingface_hub import HfApi
import torch
import re
import argparse
import os
import zipfile

def do_preprocess(class_data_dir):
    print("Unzipping dataset")
    zip_file_path = f"{class_data_dir}/class_images.zip"
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(class_data_dir)
    os.remove(zip_file_path)

def do_train(script_args):
    # Pass all arguments to trainer.py
    print("Starting training...")
    result = subprocess.run(['python', 'trainer.py'] + script_args)
    if result.returncode != 0:
        raise Exception("Training failed.")

def replace_output_dir(text, output_dir, replacement):
            # Define a pattern that matches the output_dir followed by whitespace, '/', new line, or "'"
            # Add system name from HF only in the correct spots
            pattern = rf"{output_dir}(?=[\s/'\n])"
            return re.sub(pattern, replacement, text)
    
def do_inference(dataset_name, output_dir, num_tokens):
    widget_content = []
    try:
        print("Starting inference to generate example images...")
        dataset = load_dataset(dataset_name)
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        )
        pipe = pipe.to("cuda")
        pipe.load_lora_weights(f'{output_dir}/pytorch_lora_weights.safetensors')
        
        prompts = dataset["train"]["prompt"]
        if(num_tokens > 0):
            tokens_sequence = ''.join(f'<s{i}>' for i in range(num_tokens))
            tokens_list = [f'<s{i}>' for i in range(num_tokens)]
        
            state_dict = load_file(f"{output_dir}/{output_dir}_emb.safetensors")
            pipe.load_textual_inversion(state_dict["clip_l"], token=tokens_list, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
            pipe.load_textual_inversion(state_dict["clip_g"], token=tokens_list, text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)
            
            prompts = [prompt.replace("TOK", tokens_sequence) for prompt in prompts]    

        for i, prompt in enumerate(prompts):
            image = pipe(prompt, num_inference_steps=25, guidance_scale=7.5).images[0]
            filename = f"image-{i}.png"
            image.save(f"{output_dir}/{filename}")
            card_dict = {
                "text": prompt,
                "output": {
                    "url": filename
                }
            }
            widget_content.append(card_dict)
    except Exception as e:
        print("Something went wrong with generating images, specifically: ", e)
    
    try:
        api = HfApi()
        username = api.whoami()["name"]
        repo_id = api.create_repo(f"{username}/{output_dir}", exist_ok=True, private=True).repo_id
        
        with open(f'{output_dir}/README.md', 'r') as file:
            readme_content = file.read()
        
    
        readme_content = replace_output_dir(readme_content, output_dir, f"{username}/{output_dir}")
        
        card = RepoCard(readme_content)
        if widget_content: 
            card.data["widget"] = widget_content
            card.save(f'{output_dir}/README.md')
    
        print("Starting upload...")
        api.upload_folder(
            folder_path=output_dir,
            repo_id=f"{username}/{output_dir}",
            repo_type="model",
        )
    except Exception as e:
        print("Something went wrong with uploading your model, specificaly: ", e)
    else:
        print("Upload finished!")

import sys
import argparse

def main():
    # Capture all arguments except the script name
    script_args = sys.argv[1:]

    # Create the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_new_tokens_per_abstraction', type=int, default=0)
    parser.add_argument('--train_text_encoder_ti', action='store_true')
    parser.add_argument('--class_data_dir', help="Name of the class images dataset")

    # Parse known arguments
    args, _ = parser.parse_known_args(script_args)

    # Set num_tokens to 0 if '--train_text_encoder_ti' is not present
    if not args.train_text_encoder_ti:
        args.num_new_tokens_per_abstraction = 0

    # Proceed with training and inference
    if args.class_data_dir:
        do_preprocess(args.class_data_dir)
        print("Pre-processing finished!")
    do_train(script_args)
    print("Training finished!")
    do_inference(args.dataset_name, args.output_dir, args.num_new_tokens_per_abstraction)
    print("All finished!")

if __name__ == "__main__":
    main()