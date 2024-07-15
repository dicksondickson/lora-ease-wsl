"""

LoRA Ease

v1.1

dicksondickson
https://huggingface.co/dicksondickson/lora-ease-wsl


multimodalart
https://huggingface.co/spaces/multimodalart/lora-ease


"""

import gradio as gr
from PIL import Image
import requests
import subprocess
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from huggingface_hub import snapshot_download, HfApi
import torch
import uuid
import os
import shutil
import json
import random
from slugify import slugify
import argparse 
import importlib
import sys
from pathlib import Path
import spaces
import zipfile

MAX_IMAGES = 150

is_spaces = True if os.environ.get('SPACE_ID') else False

training_script_url = "https://raw.githubusercontent.com/huggingface/diffusers/ba28006f8b2a0f7ec3b6784695790422b4f80a97/examples/advanced_diffusion_training/train_dreambooth_lora_sdxl_advanced.py"
subprocess.run(['wget', '-N', training_script_url])
orchestrator_script_url = "https://huggingface.co/datasets/multimodalart/lora-ease-helper/raw/main/script.py"
subprocess.run(['wget', '-N', orchestrator_script_url])


# device = "cuda" if torch.cuda.is_available() else "cpu"


# Check for available GPU devices
if torch.cuda.is_available():
    # If CUDA (NVIDIA GPU) is available, use it
    device = torch.device("cuda")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    # If MPS (Apple Silicon GPU) is available, use it
    device = torch.device("mps")
else:
    # If no GPU is available, use CPU
    device = torch.device("cpu")
        
        
        

FACES_DATASET_PATH = snapshot_download(repo_id="multimodalart/faces-prior-preservation", repo_type="dataset")
#Delete .gitattributes to process things properly
Path(FACES_DATASET_PATH, '.gitattributes').unlink(missing_ok=True)

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", device_map={"": 0}, torch_dtype=torch.float16
)

training_option_settings = {
    "face": {
        "rank": 32,
        "lr_scheduler": "constant",
        "with_prior_preservation": True,
        "class_prompt": "a photo of a person",
        "train_steps_multiplier": 75,
        "file_count": 150,
        "dataset_path": FACES_DATASET_PATH
    },
    "style": {
        "rank": 32,
        "lr_scheduler": "constant",
        "with_prior_preservation": False,
        "class_prompt": "",
        "train_steps_multiplier": 120
    },
    "character": {
        "rank": 32,
        "lr_scheduler": "constant",
        "with_prior_preservation": False,
        "class_prompt": "",
        "train_steps_multiplier": 180
    },
    "object": {
        "rank": 16,
        "lr_scheduler": "constant",
        "with_prior_preservation": False,
        "class_prompt": "",
        "train_steps_multiplier": 50
    },
    "custom": {  
        "rank": 32,
        "lr_scheduler": "constant",
        "with_prior_preservation": False,
        "class_prompt": "",
        "train_steps_multiplier": 150
    }
}

num_images_settings = { 
    #>24 images, 1 repeat; 10<x<24 images 2 repeats; <10 images 3 repeats
    "repeats": [(24, 1), (10, 2), (0, 3)],
    "train_steps_min": 500,
    "train_steps_max": 1500
}

def load_captioning(uploaded_images, option):
    updates = []
    if len(uploaded_images) <= 1:
        raise gr.Error(
            "Please upload at least 2 images to train your model (the ideal number with default settings is between 4-30)"
        )
    elif len(uploaded_images) > MAX_IMAGES:
        raise gr.Error(
            f"For now, only {MAX_IMAGES} or less images are allowed for training"
        )
    # Update for the captioning_area
    for _ in range(3):
        updates.append(gr.update(visible=True))
    # Update visibility and image for each captioning row and image
    for i in range(1, MAX_IMAGES + 1):
        # Determine if the current row and image should be visible
        visible = i <= len(uploaded_images)

        # Update visibility of the captioning row
        updates.append(gr.update(visible=visible))

        # Update for image component - display image if available, otherwise hide
        image_value = uploaded_images[i - 1] if visible else None
        updates.append(gr.update(value=image_value, visible=visible))

        text_value = option if visible else None
        updates.append(gr.update(value=text_value, visible=visible))
    return updates

def check_removed_and_restart(images):
    visible = len(images) > 1 if images is not None else False
    if(is_spaces):
        captioning_area = gr.update(visible=visible)
        advanced = gr.update(visible=visible)
        cost_estimation = gr.update(visible=visible)
        start = gr.update(visible=False)
    else:
        captioning_area = gr.update(visible=visible)
        advanced = gr.update(visible=visible)
        cost_estimation = gr.update(visible=False)
        start = gr.update(visible=True)
    return captioning_area, advanced,cost_estimation, start

def make_options_visible(option):
    if (option == "object") or (option == "face"):
        sentence = "A photo of TOK"
    elif option == "style":
        sentence = "in the style of TOK"
    elif option == "character":
        sentence = "A TOK character"
    elif option == "custom":
        sentence = "TOK"
    return (
        gr.update(value=sentence, visible=True),
        gr.update(visible=True),
    )
    
def change_defaults(option, images):
    settings = training_option_settings.get(option, training_option_settings["custom"])
    num_images = len(images)

    # Calculate max_train_steps
    train_steps_multiplier = settings["train_steps_multiplier"]
    max_train_steps = max(num_images * train_steps_multiplier, num_images_settings["train_steps_min"])
    max_train_steps = min(max_train_steps, num_images_settings["train_steps_max"])

    # Determine repeats based on number of images
    repeats = next(repeats for num, repeats in num_images_settings["repeats"] if num_images > num)

    random_files = []
    if settings["with_prior_preservation"]:
        directory = settings["dataset_path"]
        file_count = settings["file_count"]
        files = [os.path.join(directory, file) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
        random_files = random.sample(files, min(len(files), file_count))

    return max_train_steps, repeats, settings["lr_scheduler"], settings["rank"], settings["with_prior_preservation"], settings["class_prompt"], random_files
    
def create_dataset(*inputs):
    print("Creating dataset")
    images = inputs[0]
    destination_folder = str(uuid.uuid4())
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    jsonl_file_path = os.path.join(destination_folder, 'metadata.jsonl')
    with open(jsonl_file_path, 'a') as jsonl_file:
        for index, image in enumerate(images):
            new_image_path = shutil.copy(image, destination_folder)
            
            original_caption = inputs[index + 1]
            file_name = os.path.basename(new_image_path)

            data = {"file_name": file_name, "prompt": original_caption}

            jsonl_file.write(json.dumps(data) + "\n")
    
    return destination_folder

def start_training(
    lora_name,
    training_option,
    concept_sentence,
    optimizer,
    use_snr_gamma,
    snr_gamma,
    mixed_precision,
    learning_rate,
    train_batch_size,
    max_train_steps,
    lora_rank,
    repeats,
    with_prior_preservation,
    class_prompt,
    class_images,
    num_class_images,
    train_text_encoder_ti,
    train_text_encoder_ti_frac,
    num_new_tokens_per_abstraction,
    train_text_encoder,
    train_text_encoder_frac,
    text_encoder_learning_rate,
    seed,
    resolution,
    num_train_epochs,
    checkpointing_steps,
    prior_loss_weight,
    gradient_accumulation_steps,
    gradient_checkpointing,
    enable_xformers_memory_efficient_attention,
    adam_beta1,
    adam_beta2,
    use_prodigy_beta3,
    prodigy_beta3,
    prodigy_decouple,
    adam_weight_decay,
    use_adam_weight_decay_text_encoder,
    adam_weight_decay_text_encoder,
    adam_epsilon,
    prodigy_use_bias_correction,
    prodigy_safeguard_warmup,
    max_grad_norm,
    scale_lr,
    lr_num_cycles,
    lr_scheduler,
    lr_power,
    lr_warmup_steps,
    dataloader_num_workers,
    local_rank,
    dataset_folder,
    token,
    progress = gr.Progress(track_tqdm=True)
):
    if not lora_name:
        raise gr.Error("You forgot to insert your LoRA name! This name has to be unique.")
    print("Started training")
    slugged_lora_name = slugify(lora_name)
    spacerunner_folder = str(uuid.uuid4())
    commands = [
        "pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0",
        "pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix",
        f"instance_prompt={concept_sentence}",
        f"dataset_name=./{dataset_folder}",
        "caption_column=prompt",
        f"output_dir={slugged_lora_name}",
        f"mixed_precision={mixed_precision}",
        f"resolution={int(resolution)}",
        f"train_batch_size={int(train_batch_size)}",
        f"repeats={int(repeats)}",
        f"gradient_accumulation_steps={int(gradient_accumulation_steps)}",
        f"learning_rate={learning_rate}",
        f"text_encoder_lr={text_encoder_learning_rate}",
        f"adam_beta1={adam_beta1}",
        f"adam_beta2={adam_beta2}",
        f"optimizer={'adamW' if optimizer == '8bitadam' else optimizer}",
        f"train_text_encoder_ti_frac={train_text_encoder_ti_frac}",
        f"lr_scheduler={lr_scheduler}",
        f"lr_warmup_steps={int(lr_warmup_steps)}",
        f"rank={int(lora_rank)}",
        f"max_train_steps={int(max_train_steps)}",
        f"checkpointing_steps={int(checkpointing_steps)}",
        f"seed={int(seed)}",
        f"prior_loss_weight={prior_loss_weight}",
        f"num_new_tokens_per_abstraction={int(num_new_tokens_per_abstraction)}",
        f"num_train_epochs={int(num_train_epochs)}",
        f"adam_weight_decay={adam_weight_decay}",
        f"adam_epsilon={adam_epsilon}",
        f"prodigy_decouple={prodigy_decouple}",
        f"prodigy_use_bias_correction={prodigy_use_bias_correction}",
        f"prodigy_safeguard_warmup={prodigy_safeguard_warmup}",
        f"max_grad_norm={max_grad_norm}",
        f"lr_num_cycles={int(lr_num_cycles)}",
        f"lr_power={lr_power}",
        f"dataloader_num_workers={int(dataloader_num_workers)}",
        f"local_rank={int(local_rank)}",
        "cache_latents",
        #"push_to_hub",
    ]
    # Adding optional flags
    if optimizer == "8bitadam":
        commands.append("use_8bit_adam")
    if gradient_checkpointing:
        commands.append("gradient_checkpointing")
    
    if train_text_encoder_ti:
        commands.append("train_text_encoder_ti")
    elif train_text_encoder:
        commands.append("train_text_encoder")
        commands.append(f"train_text_encoder_frac={train_text_encoder_frac}")
    if enable_xformers_memory_efficient_attention: 
        commands.append("enable_xformers_memory_efficient_attention")
    if use_snr_gamma: 
        commands.append(f"snr_gamma={snr_gamma}")
    if scale_lr:
        commands.append("scale_lr")
    if with_prior_preservation:
        commands.append("with_prior_preservation")
        commands.append(f"class_prompt={class_prompt}")
        commands.append(f"num_class_images={int(num_class_images)}")
        if class_images:
            class_folder = str(uuid.uuid4())
            zip_path = os.path.join(spacerunner_folder, class_folder, "class_images.zip")
        
            if not os.path.exists(os.path.join(spacerunner_folder, class_folder)):
                os.makedirs(os.path.join(spacerunner_folder, class_folder))
        
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for image in class_images:
                    zipf.write(image, os.path.basename(image))
            
            commands.append(f"class_data_dir={class_folder}")
    if use_prodigy_beta3:
        commands.append(f"prodigy_beta3={prodigy_beta3}")
    if use_adam_weight_decay_text_encoder:
        commands.append(f"adam_weight_decay_text_encoder={adam_weight_decay_text_encoder}")
    print(commands)
    # Joining the commands with ';' separator for spacerunner format
    spacerunner_args = ';'.join(commands)
    if not os.path.exists(spacerunner_folder):
        os.makedirs(spacerunner_folder)
    shutil.copy("train_dreambooth_lora_sdxl_advanced.py", f"{spacerunner_folder}/trainer.py")
    shutil.copy("script.py", f"{spacerunner_folder}/script.py")
    shutil.copytree(dataset_folder, f"{spacerunner_folder}/{dataset_folder}")
    requirements='''peft==0.7.1
-huggingface_hub
torch
git+https://github.com/huggingface/diffusers@ba28006f8b2a0f7ec3b6784695790422b4f80a97
transformers==4.36.2
accelerate==0.25.0
safetensors==0.4.1
prodigyopt==1.0
hf-transfer==0.1.4
huggingface_hub==0.20.3
git+https://github.com/huggingface/datasets.git@3f149204a2a5948287adcade5e90707aa5207a92'''
    file_path = f'{spacerunner_folder}/requirements.txt'
    with open(file_path, 'w') as file:
        file.write(requirements)
    # The subprocess call for autotrain spacerunner
    api = HfApi(token=token)
    username = api.whoami()["name"]
    subprocess_command = ["autotrain", "spacerunner", "--project-name", slugged_lora_name, "--script-path", spacerunner_folder, "--username", username, "--token", token, "--backend", "spaces-a10g-small", "--env",f"HF_TOKEN={token};HF_HUB_ENABLE_HF_TRANSFER=1", "--args", spacerunner_args]
    outcome = subprocess.run(subprocess_command)
    if(outcome.returncode == 0):
        return f"""# Your training has started. 
## - Training Status: <a href='https://huggingface.co/spaces/{username}/autotrain-{slugged_lora_name}?logs=container'>{username}/autotrain-{slugged_lora_name}</a> <small>(in the logs tab)</small>
## - Model page: <a href='https://huggingface.co/{username}/{slugged_lora_name}'>{username}/{slugged_lora_name}</a> <small>(will be available when training finishes)</small>"""
    else:
        print("Error: ", outcome.stderr)
        raise gr.Error("Something went wrong. Make sure the name of your LoRA is unique and try again")

def calculate_price(iterations, with_prior_preservation):
    if(with_prior_preservation):
        seconds_per_iteration = 3.50
    else:
        seconds_per_iteration = 2.00
    total_seconds = (iterations * seconds_per_iteration) + 210
    cost_per_second = 1.05/60/60
    cost = round(cost_per_second * total_seconds, 2)
    return f'''To train this LoRA, we will duplicate the space and hook an A10G GPU under the hood.
## Estimated to cost <b>< US$ {str(cost)}</b> for {round(int(total_seconds)/60, 2)} minutes with your current train settings <small>({int(iterations)} iterations at {seconds_per_iteration}s/it)</small>
#### ↓ to continue, grab you <b>write</b> token [here](https://huggingface.co/settings/tokens) and enter it below ↓'''

def start_training_og(
    lora_name,
    training_option,
    concept_sentence,
    optimizer,
    use_snr_gamma,
    snr_gamma,
    mixed_precision,
    learning_rate,
    train_batch_size,
    max_train_steps,
    lora_rank,
    repeats,
    with_prior_preservation,
    class_prompt,
    class_images,
    num_class_images,
    train_text_encoder_ti,
    train_text_encoder_ti_frac,
    num_new_tokens_per_abstraction,
    train_text_encoder,
    train_text_encoder_frac,
    text_encoder_learning_rate,
    seed,
    resolution,
    num_train_epochs,
    checkpointing_steps,
    prior_loss_weight,
    gradient_accumulation_steps,
    gradient_checkpointing,
    enable_xformers_memory_efficient_attention,
    adam_beta1,
    adam_beta2,
    use_prodigy_beta3,
    prodigy_beta3,
    prodigy_decouple,
    adam_weight_decay,
    use_adam_weight_decay_text_encoder,
    adam_weight_decay_text_encoder,
    adam_epsilon,
    prodigy_use_bias_correction,
    prodigy_safeguard_warmup,
    max_grad_norm,
    scale_lr,
    lr_num_cycles,
    lr_scheduler,
    lr_power,
    lr_warmup_steps,
    dataloader_num_workers,
    local_rank,
    dataset_folder,
    token,
    #progress = gr.Progress(track_tqdm=True)
):
    if not lora_name:
        raise gr.Error("You forgot to insert your LoRA name!")
    slugged_lora_name = slugify(lora_name)
    commands = [
            "--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0",
            "--pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix",
            f"--instance_prompt={concept_sentence}",
            f"--dataset_name=./{dataset_folder}",
            "--caption_column=prompt",
            f"--output_dir={slugged_lora_name}",
            f"--mixed_precision={mixed_precision}",
            f"--resolution={int(resolution)}",
            f"--train_batch_size={int(train_batch_size)}",
            f"--repeats={int(repeats)}",
            f"--gradient_accumulation_steps={int(gradient_accumulation_steps)}",
            f"--learning_rate={learning_rate}",
            f"--text_encoder_lr={text_encoder_learning_rate}",
            f"--adam_beta1={adam_beta1}",
            f"--adam_beta2={adam_beta2}",
            f"--optimizer={'adamW' if optimizer == '8bitadam' else optimizer}",
            f"--train_text_encoder_ti_frac={train_text_encoder_ti_frac}",
            f"--lr_scheduler={lr_scheduler}",
            f"--lr_warmup_steps={int(lr_warmup_steps)}",
            f"--rank={int(lora_rank)}",
            f"--max_train_steps={int(max_train_steps)}",
            f"--checkpointing_steps={int(checkpointing_steps)}",
            f"--seed={int(seed)}",
            f"--prior_loss_weight={prior_loss_weight}",
            f"--num_new_tokens_per_abstraction={int(num_new_tokens_per_abstraction)}",
            f"--num_train_epochs={int(num_train_epochs)}",
            f"--adam_weight_decay={adam_weight_decay}",
            f"--adam_epsilon={adam_epsilon}",
            f"--prodigy_decouple={prodigy_decouple}",
            f"--prodigy_use_bias_correction={prodigy_use_bias_correction}",
            f"--prodigy_safeguard_warmup={prodigy_safeguard_warmup}",
            f"--max_grad_norm={max_grad_norm}",
            f"--lr_num_cycles={int(lr_num_cycles)}",
            f"--lr_power={lr_power}",
            f"--dataloader_num_workers={int(dataloader_num_workers)}",
            f"--local_rank={int(local_rank)}",
            "--cache_latents"
            ]
    if optimizer == "8bitadam":
        commands.append("--use_8bit_adam")
    if gradient_checkpointing:
        commands.append("--gradient_checkpointing")
    
    if train_text_encoder_ti:
        commands.append("--train_text_encoder_ti")
    elif train_text_encoder:
        commands.append("--train_text_encoder")
        commands.append(f"--train_text_encoder_frac={train_text_encoder_frac}")
    if enable_xformers_memory_efficient_attention: 
        commands.append("--enable_xformers_memory_efficient_attention")
    if use_snr_gamma: 
        commands.append(f"--snr_gamma={snr_gamma}")
    if scale_lr:
        commands.append("--scale_lr")
    if with_prior_preservation:
        commands.append(f"--with_prior_preservation")
        commands.append(f"--class_prompt={class_prompt}")
        commands.append(f"--num_class_images={int(num_class_images)}")
        if(class_images):
            class_folder = str(uuid.uuid4())
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
            for image in class_images:
                shutil.copy(image, class_folder)
            commands.append(f"--class_data_dir={class_folder}")
    if use_prodigy_beta3:
        commands.append(f"--prodigy_beta3={prodigy_beta3}")
    if use_adam_weight_decay_text_encoder:
        commands.append(f"--adam_weight_decay_text_encoder={adam_weight_decay_text_encoder}")
    from train_dreambooth_lora_sdxl_advanced import main as train_main, parse_args as parse_train_args
    args = parse_train_args(commands)
    
    train_main(args)
    
    return f"Your model has finished training and has been saved to the `{slugged_lora_name}` folder"

@spaces.GPU(enable_queue=True)
def run_captioning(*inputs):
    model.to("cuda")
    images = inputs[0]
    training_option = inputs[-1]
    final_captions = [""] * MAX_IMAGES
    for index, image in enumerate(images):
        original_caption = inputs[index + 1]
        pil_image = Image.open(image)  
        blip_inputs = processor(images=pil_image, return_tensors="pt").to(device, torch.float16)
        generated_ids = model.generate(**blip_inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        if training_option == "style":
            final_caption = generated_text + " " + original_caption
        else:
            final_caption = original_caption + " " + generated_text
        final_captions[index] = final_caption
        yield final_captions

def check_token(token):
    try:
        api = HfApi(token=token)
        user_data = api.whoami()
    except Exception as e:
        gr.Warning("Invalid user token. Make sure to get your Hugging Face token from the settings page")
        return gr.update(visible=False), gr.update(visible=False)
    else:
        if (user_data['auth']['accessToken']['role'] != "write"):
            gr.Warning("Ops, you've uploaded a Read token. You need to use a Write token!")
        else:
            if user_data['canPay']:
                return gr.update(visible=False), gr.update(visible=True)    
            else:
                return gr.update(visible=True), gr.update(visible=False)
                
        return gr.update(visible=False), gr.update(visible=False)

def check_if_tok(sentence, textual_inversion):
    if "TOK" not in sentence and textual_inversion:
        gr.Warning("⚠️ You've removed the special token TOK from your concept sentence. This will degrade performance as this special token is needed for textual inversion. Use TOK to describe what you are training.")
        
css = '''.gr-group{background-color: transparent;box-shadow: var(--block-shadow)}
.gr-group .hide-container{padding: 1em; background: var(--block-background-fill) !important}
.gr-group img{object-fit: cover}
#main_title{text-align:center}
#main_title h1 {font-size: 2.25rem}
#main_title h3, #main_title p{margin-top: 0;font-size: 1.25em}
#training_cost h2{margin-top: 10px;padding: 0.5em;border: 1px solid var(--block-border-color);font-size: 1.25em}
#training_cost h4{margin-top: 1.25em;margin-bottom: 0}
#training_cost small{font-weight: normal}
.accordion {color: var(--body-text-color)}
.main_unlogged{opacity: 0.5;pointer-events: none}
.login_logout{width: 100% !important}
#login {font-size: 0px;width: 100% !important;margin: 0 auto}
#login:after {content: 'Authorize this app to train your model';visibility: visible;display: block;font-size: var(--button-large-text-size)}
#component-3, component-697{border: 0}
'''

theme = gr.themes.Monochrome(
    text_size=gr.themes.Size(lg="18px", md="15px", sm="13px", xl="22px", xs="12px", xxl="24px", xxs="9px"),
    font=[gr.themes.GoogleFont('Source Sans Pro'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
)

with gr.Blocks(css=css, theme=theme) as demo:
    dataset_folder = gr.State()
    gr.Markdown('''# LoRA Ease 🧞‍♂️
### Train a high quality SDXL LoRA in a breeze ༄ with state-of-the-art techniques and for cheap 
<small>Dreambooth with Pivotal Tuning, Prodigy and more! Use the trained LoRAs with diffusers, AUTO1111, Comfy. [blog about the training script](https://huggingface.co/blog/sdxl_lora_advanced_script), [Colab Pro](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/SDXL_Dreambooth_LoRA_advanced_example.ipynb), [run locally or in a cloud](https://github.com/huggingface/diffusers/blob/main/examples/advanced_diffusion_training/train_dreambooth_lora_sdxl_advanced.py)</small>.''', elem_id="main_title")
    #gr.LoginButton(elem_classes=["login_logout"])
    with gr.Tab("Train on Spaces"):
        with gr.Column(elem_classes=["main_logged"]) as main_ui:
            lora_name = gr.Textbox(label="The name of your LoRA", info="This has to be a unique name", placeholder="e.g.: Persian Miniature Painting style, Cat Toy")
            training_option = gr.Radio(
                label="What are you training?", choices=["object", "style", "character", "face", "custom"]
            )
            concept_sentence = gr.Textbox(
                label="Concept sentence",
                info="Sentence to be used in all images for captioning. TOK is a special mandatory token, used to teach the model your concept.",
                placeholder="e.g.: A photo of TOK, in the style of TOK",
                visible=False,
                interactive=True,
            )
            with gr.Group(visible=False) as image_upload:
                with gr.Row():
                    images = gr.File(
                        file_types=["image"],
                        label="Upload your images",
                        file_count="multiple",
                        interactive=True,
                        visible=True,
                        scale=1,
                    )
                    with gr.Column(scale=3, visible=False) as captioning_area:
                        with gr.Column():
                            gr.Markdown(
                                """# Custom captioning
To improve the quality of your outputs, you can add a custom caption for each image, describing exactly what is taking place in each of them. Including TOK is mandatory. You can leave things as is if you don't want to include captioning.
"""
                            )
                            do_captioning = gr.Button("Add AI captions with BLIP-2")
                            output_components = [captioning_area]
                            caption_list = []
                            for i in range(1, MAX_IMAGES + 1):
                                locals()[f"captioning_row_{i}"] = gr.Row(visible=False)
                                with locals()[f"captioning_row_{i}"]:
                                    locals()[f"image_{i}"] = gr.Image(
                                        width=111,
                                        height=111,
                                        min_width=111,
                                        interactive=False,
                                        scale=2,
                                        show_label=False,
                                        show_share_button=False,
                                        show_download_button=False
                                    )
                                    locals()[f"caption_{i}"] = gr.Textbox(
                                        label=f"Caption {i}", scale=15, interactive=True
                                    )
        
                                output_components.append(locals()[f"captioning_row_{i}"])
                                output_components.append(locals()[f"image_{i}"])
                                output_components.append(locals()[f"caption_{i}"])
                                caption_list.append(locals()[f"caption_{i}"])
            with gr.Accordion(open=False, label="Advanced options", visible=False, elem_classes=['accordion']) as advanced:
                with gr.Row():
                    with gr.Column():
                        optimizer = gr.Dropdown(
                            label="Optimizer",
                            info="Prodigy is an auto-optimizer and works good by default. If you prefer to set your own learning rates, change it to AdamW. If you don't have enough VRAM to train with AdamW, pick 8-bit Adam.",
                            choices=[
                                ("Prodigy", "prodigy"),
                                ("AdamW", "adamW"),
                                ("8-bit Adam", "8bitadam"),
                            ],
                            value="prodigy",
                            interactive=True,
                        )
                        use_snr_gamma = gr.Checkbox(label="Use SNR Gamma")
                        snr_gamma = gr.Number(
                            label="snr_gamma",
                            info="SNR weighting gamma to re-balance the loss",
                            value=5.000,
                            step=0.1,
                            visible=False,
                        )
                        mixed_precision = gr.Dropdown(
                            label="Mixed Precision",
                            choices=["no", "fp16", "bf16"],
                            value="bf16",
                        )
                        learning_rate = gr.Number(
                            label="UNet Learning rate",
                            minimum=0.0,
                            maximum=10.0,
                            step=0.0000001,
                            value=1.0,  # For prodigy you start high and it will optimize down
                        )
                        max_train_steps = gr.Number(
                            label="Max train steps", minimum=1, maximum=50000, value=1000
                        )
                        lora_rank = gr.Number(
                            label="LoRA Rank",
                            info="Rank for the Low Rank Adaptation (LoRA), a higher rank produces a larger LoRA",
                            value=8,
                            step=2,
                            minimum=2,
                            maximum=1024,
                        )
                        repeats = gr.Number(
                            label="Repeats",
                            info="How many times to repeat the training data.",
                            value=1,
                            minimum=1,
                            maximum=200,
                        )
                    with gr.Column():
                        with_prior_preservation = gr.Checkbox(
                            label="Prior preservation loss",
                            info="Prior preservation helps to ground the model to things that are similar to your concept. Good for faces.",
                            value=False,
                        )
                        with gr.Column(visible=False) as prior_preservation_params:
                            with gr.Tab("prompt"):
                                class_prompt = gr.Textbox(
                                    label="Class Prompt",
                                    info="The prompt that will be used to generate your class images",
                                )
        
                            with gr.Tab("images"):
                                class_images = gr.File(
                                    file_types=["image"],
                                    label="Upload your images",
                                    file_count="multiple",
                                )
                            num_class_images = gr.Number(
                                label="Number of class images, if there are less images uploaded then the number you put here, additional images will be sampled with Class Prompt",
                                value=20,
                            )
                        train_text_encoder_ti = gr.Checkbox(
                            label="Do textual inversion",
                            value=True,
                            info="Will train a textual inversion embedding together with the LoRA. Increases quality significantly. If untoggled, you can remove the special TOK token from the prompts.",
                        )
                        with gr.Group(visible=True) as pivotal_tuning_params:
                            train_text_encoder_ti_frac = gr.Number(
                                label="Pivot Textual Inversion",
                                info="% of epochs to train textual inversion for",
                                value=0.5,
                                step=0.1,
                            )
                            num_new_tokens_per_abstraction = gr.Number(
                                label="Tokens to train",
                                info="Number of tokens to train in the textual inversion",
                                value=2,
                                minimum=1,
                                maximum=1024,
                                interactive=True,
                            )
                        with gr.Group(visible=False) as text_encoder_train_params:
                            train_text_encoder = gr.Checkbox(
                                label="Train Text Encoder", value=True
                            )
                            train_text_encoder_frac = gr.Number(
                                label="Pivot Text Encoder",
                                info="% of epochs to train the text encoder for",
                                value=0.8,
                                step=0.1,
                            )
                        text_encoder_learning_rate = gr.Number(
                            label="Text encoder learning rate",
                            minimum=0.0,
                            maximum=10.0,
                            step=0.0000001,
                            value=1.0,
                        )
                        seed = gr.Number(label="Seed", value=42)
                        resolution = gr.Number(
                            label="Resolution",
                            info="Only square sizes are supported for now, the value will be width and height",
                            value=1024,
                        )
        
                with gr.Accordion(open=False, label="Even more advanced options", elem_classes=['accordion']):
                    with gr.Row():
                        with gr.Column():
                            gradient_accumulation_steps = gr.Number(
                                info="If you change this setting, the pricing calculation will be wrong",
                                label="gradient_accumulation_steps", 
                                value=1
                            )
                            train_batch_size = gr.Number(
                                info="If you change this setting, the pricing calculation will be wrong",
                                label="Train batch size",
                                value=2
                            )
                            num_train_epochs = gr.Number(
                                info="If you change this setting, the pricing calculation will be wrong",
                                label="num_train_epochs",
                                value=1
                            )
                            checkpointing_steps = gr.Number(
                                info="How many steps to save intermediate checkpoints",
                                label="checkpointing_steps",
                                value=100000,
                                visible=False #hack to not let users break this for now
                            )
                            prior_loss_weight = gr.Number(
                                label="prior_loss_weight",
                                value=1
                            )
                            gradient_checkpointing = gr.Checkbox(
                                label="gradient_checkpointing",
                                info="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass",
                                value=True,
                            )
                            adam_beta1 = gr.Number(
                                label="adam_beta1",
                                value=0.9,
                                minimum=0,
                                maximum=1,
                                step=0.01
                            )
                            adam_beta2 = gr.Number(
                                label="adam_beta2",
                                minimum=0,
                                maximum=1,
                                step=0.01,
                                value=0.999
                            )
                            use_prodigy_beta3 = gr.Checkbox(
                                label="Use Prodigy Beta 3?"
                            )
                            prodigy_beta3 = gr.Number(
                                label="Prodigy Beta 3",
                                value=None,
                                step=0.01,
                                minimum=0,
                                maximum=1,
                            )
                            prodigy_decouple = gr.Checkbox(
                                label="Prodigy Decouple",
                                value=True
                            )
                            adam_weight_decay = gr.Number(
                                label="Adam Weight Decay",
                                value=1e-04,
                                step=0.00001,
                                minimum=0,
                                maximum=1,
                            )
                            use_adam_weight_decay_text_encoder = gr.Checkbox(
                                label="Use Adam Weight Decay Text Encoder"
                            )
                            adam_weight_decay_text_encoder = gr.Number(
                                label="Adam Weight Decay Text Encoder",
                                value=None,
                                step=0.00001,
                                minimum=0,
                                maximum=1,
                            )
                            adam_epsilon = gr.Number(
                                label="Adam Epsilon",
                                value=1e-08,
                                step=0.00000001,
                                minimum=0,
                                maximum=1,
                            )
                            prodigy_use_bias_correction = gr.Checkbox(
                                label="Prodigy Use Bias Correction",
                                value=True
                            )
                            prodigy_safeguard_warmup = gr.Checkbox(
                                label="Prodigy Safeguard Warmup",
                                value=True
                            )
                            max_grad_norm = gr.Number(
                                label="Max Grad Norm",
                                value=1.0,
                                minimum=0.1,
                                maximum=10,
                                step=0.1,
                            )
                            enable_xformers_memory_efficient_attention = gr.Checkbox(
                                label="enable_xformers_memory_efficient_attention"
                            )
                        with gr.Column():
                            scale_lr = gr.Checkbox(
                                label="Scale learning rate",
                                info="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size",
                            )
                            lr_num_cycles = gr.Number(
                                label="lr_num_cycles",
                                value=1
                            )
                            lr_scheduler = gr.Dropdown(
                                label="lr_scheduler",
                                choices=[
                                    "linear",
                                    "cosine",
                                    "cosine_with_restarts",
                                    "polynomial",
                                    "constant",
                                    "constant_with_warmup",
                                ],
                                value="constant",
                            )
                            lr_power = gr.Number(
                                label="lr_power",
                                value=1.0,
                                minimum=0.1,
                                maximum=10
                            )
                            lr_warmup_steps = gr.Number(
                                label="lr_warmup_steps",
                                value=0
                            )
                            dataloader_num_workers = gr.Number(
                                label="Dataloader num workers", value=0, minimum=0, maximum=64
                            )
                            local_rank = gr.Number(
                                label="local_rank",
                                value=-1
                            )
            with gr.Column(visible=False) as cost_estimation:
                with gr.Group(elem_id="cost_box"):
                    training_cost_estimate = gr.Markdown(elem_id="training_cost")
                    token = gr.Textbox(label="Your Hugging Face write token", info="A Hugging Face write token you can obtain on the settings page", type="password", placeholder="hf_OhHiThIsIsNoTaReALToKeNGOoDTry")
            with gr.Group(visible=False) as no_payment_method:
                with gr.Row():
                    gr.HTML("<h3 style='margin: 0'>Your Hugging Face account doesn't have a payment method set up. Set one up <a href='https://huggingface.co/settings/billing/payment' target='_blank'>here</a> and come back here to train your LoRA</h3>")
                    payment_setup = gr.Button("I have set up a payment method")
            
            start = gr.Button("Start training", visible=False, interactive=True)
            progress_area = gr.Markdown("")
    with gr.Tab("Train locally"):
        gr.Markdown(f'''To use LoRA Ease locally with a UI, you can clone this repository (yes, HF Spaces are git repos!)
```bash
git clone https://huggingface.co/spaces/multimodalart/lora-ease
```

Install the dependencies in the `requirements_local.txt` with

```bash
pip install -r requirements_local.txt
```
(if you prefer, do it in a venv environment)

Now you can run LoRA Ease locally by doing a simple 
```py
python app.py
```

If you prefer command line, you can run our [training script]({training_script_url}) yourself.
''')
    #gr.LogoutButton(elem_classes=["login_logout"])
    output_components.insert(1, advanced)
    output_components.insert(1, cost_estimation)
    gr.on(
        triggers=[
            token.change,
            payment_setup.click
        ],
        fn=check_token,
        inputs=token,
        outputs=[no_payment_method, start],
        concurrency_limit=50,
    )
    concept_sentence.change(
        check_if_tok,
        inputs=[concept_sentence, train_text_encoder_ti],
        concurrency_limit=50,
    )
    use_snr_gamma.change(
        lambda x: gr.update(visible=x),
        inputs=use_snr_gamma,
        outputs=snr_gamma,
        queue=False,
    )
    with_prior_preservation.change(
        lambda x: gr.update(visible=x),
        inputs=with_prior_preservation,
        outputs=prior_preservation_params,
        queue=False,
    )
    train_text_encoder_ti.change(
        lambda x: gr.update(visible=x),
        inputs=train_text_encoder_ti,
        outputs=pivotal_tuning_params,
        queue=False,
    ).then(
        lambda x: gr.update(visible=(not x)),
        inputs=train_text_encoder_ti,
        outputs=text_encoder_train_params,
        queue=False,
    ).then(
        lambda x: gr.Warning("As you have disabled Pivotal Tuning, you can remove TOK from your prompts and try to find a unique token for them") if not x else None,
        inputs=train_text_encoder_ti,
        concurrency_limit=50,
    )
    train_text_encoder.change(
        lambda x: [gr.update(visible=x), gr.update(visible=x)],
        inputs=train_text_encoder,
        outputs=[train_text_encoder_frac, text_encoder_learning_rate],
        queue=False,
    )
    class_images.change(
        lambda x: gr.update(value=len(x)),
        inputs=class_images,
        outputs=num_class_images,
        queue=False
    )
    images.upload(
        load_captioning,
        inputs=[images, concept_sentence],
        outputs=output_components,
        queue=False
    ).success(
        change_defaults,
        inputs=[training_option, images],
        outputs=[max_train_steps, repeats, lr_scheduler, lora_rank, with_prior_preservation, class_prompt, class_images],
        queue=False
    )
    images.change(
        check_removed_and_restart,
        inputs=[images],
        outputs=[captioning_area, advanced, cost_estimation, start],
        queue=False
    )
    training_option.change(
        make_options_visible,
        inputs=training_option,
        outputs=[concept_sentence, image_upload],
        queue=False
    )
    max_train_steps.change(
        calculate_price,
        inputs=[max_train_steps, with_prior_preservation],
        outputs=[training_cost_estimate],
        queue=False
    )
    start.click(
        fn=create_dataset,
        inputs=[images] + caption_list,
        outputs=dataset_folder,
        queue=False
    ).then(
        fn=start_training if is_spaces else start_training_og,
        inputs=[
            lora_name,
            training_option,
            concept_sentence,
            optimizer,
            use_snr_gamma,
            snr_gamma,
            mixed_precision,
            learning_rate,
            train_batch_size,
            max_train_steps,
            lora_rank,
            repeats,
            with_prior_preservation,
            class_prompt,
            class_images,
            num_class_images,
            train_text_encoder_ti,
            train_text_encoder_ti_frac,
            num_new_tokens_per_abstraction,
            train_text_encoder,
            train_text_encoder_frac,
            text_encoder_learning_rate,
            seed,
            resolution,
            num_train_epochs,
            checkpointing_steps,
            prior_loss_weight,
            gradient_accumulation_steps,
            gradient_checkpointing,
            enable_xformers_memory_efficient_attention,
            adam_beta1,
            adam_beta2,
            use_prodigy_beta3,
            prodigy_beta3,
            prodigy_decouple,
            adam_weight_decay,
            use_adam_weight_decay_text_encoder,
            adam_weight_decay_text_encoder,
            adam_epsilon,
            prodigy_use_bias_correction,
            prodigy_safeguard_warmup,
            max_grad_norm,
            scale_lr,
            lr_num_cycles,
            lr_scheduler,
            lr_power,
            lr_warmup_steps,
            dataloader_num_workers,
            local_rank,
            dataset_folder,
            token
        ],
        outputs = progress_area,
        queue=False
    )

    do_captioning.click(
        fn=run_captioning, inputs=[images] + caption_list + [training_option], outputs=caption_list
    )
    #demo.load(fn=swap_opacity, outputs=[main_ui], queue=False, concurrency_limit=50)
if __name__ == "__main__":
    demo.queue()
    demo.launch(share=True)