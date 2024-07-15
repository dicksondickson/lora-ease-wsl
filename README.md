---
license: mit
---

# lora-ease-wsl

This is a modified Python script of LoRA Ease from multimodalart to run on locally on Windows 11 via WSL. Thanks to multimodalart for this wonderful script.


Github version of this repo:  
[https://github.com/dicksondickson/lora-ease-wsl](https://github.com/dicksondickson/lora-ease-wsl)  


Hugging Face version of this repo:   
[https://huggingface.co/dicksondickson/lora-ease-wsl](https://huggingface.co/dicksondickson/lora-ease-wsl)   



Original repo:  
[https://huggingface.co/spaces/multimodalart/lora-ease](https://huggingface.co/spaces/multimodalart/lora-ease)  


Thier blog post on the script:  
[https://huggingface.co/blog/sdxl_lora_advanced_script](https://huggingface.co/blog/sdxl_lora_advanced_script)  


araminta_k's training tutorial:  
[https://huggingface.co/blog/alvdansen/training-lora-m3lt](https://huggingface.co/blog/alvdansen/training-lora-m3lt)  



# Install  

Install WSL with Ubuntu (Ubuntu 22.04)  

`wsl --install Ubuntu`  

You'll be asked to provide a username and password for your Ubuntu install.  
You should now be logged into your WSL Ubuntu distro.

Update the system packages:   

`sudo apt update`   
`sudo apt upgrade`   

Install Python related stuff:   

`sudo apt install python-is-python3`   
`sudo apt install python3-pip`   

Check if your Nvidia video card is recognized:   

`nvidia-smi`   

Make sure you are currently in your home directory:   

`cd $home`   
`pwd`   

Make a new directory to clone lora ease repo:   

`mkdir train`   

Go into dir:   

`cd train`   

Clone this repo:   

`git clone https://huggingface.co/dicksondickson/lora-ease-wsl`

Go into the dir:   

`cd lora-ease-wsl`   

Install lora ease required dependencies:   

`pip install -r requirements_local.txt`   

Run the LoRA Ease main python script. This will automatically install other stuff and download required models.   

`python app.py`   

After that is done, you should be presented with a link to load up the GUI in your web browser:   

`Running on local URL:  http://127.0.0.1:7860`   

Have fun!   


