from dataclasses import dataclass
from datetime import datetime
from distutils.util import strtobool
import logging
import os
import re
from typing import Optional
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import is_liger_kernel_available
from trl import SFTTrainer, TrlParser, ModelConfig, SFTConfig, get_peft_config
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM

if is_liger_kernel_available():
    from liger_kernel.transformers import AutoLigerKernelForCausalLM
  
########################
# Custom dataclasses
########################
@dataclass
class ScriptArguments:
    dataset_id_or_path: str
    num_samples: int = None
    dataset_splits: str = "train"
    tokenizer_name_or_path: str = None
    spectrum_config_path: Optional[str] = None
    save_merged_weights: bool = True  # Default to save merged weights


########################
# Setup logging
########################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

########################
# Helper functions
########################

def get_checkpoint(training_args: SFTConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def setup_model_for_spectrum(model, spectrum_config_path):
    unfrozen_parameters = []
    with open(spectrum_config_path, "r") as fin:
        yaml_parameters = fin.read()

    # get the unfrozen parameters from the yaml file
    for line in yaml_parameters.splitlines():
        if line.startswith("- "):
            unfrozen_parameters.append(line.split("- ")[1])

    # freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    # unfreeze Spectrum parameters
    for name, param in model.named_parameters():
        if any(re.match(unfrozen_param, name) for unfrozen_param in unfrozen_parameters):
            param.requires_grad = True
    
    # COMMENT IN: for sanity check print the trainable parameters
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"Trainable parameter: {name}")      
            
    return model


def save_merged_model(trainer, output_dir):
    """Helper function to save the merged model."""
    logger.info("Saving merged model...")
    
    # Create a "merged" subfolder to store the merged model
    merged_dir = os.path.join(output_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)
    
    # Get the merged model
    merged_model = trainer.model.merge_and_unload()
    
    # Save the merged model
    merged_model.save_pretrained(merged_dir)
    logger.info(f"Merged model saved to {merged_dir}")
    
    # Also save the tokenizer with the merged model
    trainer.tokenizer.save_pretrained(merged_dir)
    logger.info(f"Tokenizer saved to {merged_dir}")
    
    return merged_dir

def train_function(model_args: ModelConfig, script_args: ScriptArguments, training_args: SFTConfig):
    """Main training function."""
    #########################
    # Log parameters
    #########################
    logger.info(f'Model parameters {model_args}')
    logger.info(f'Script parameters {script_args}')
    logger.info(f'Training/evaluation parameters {training_args}')

    ###############
    # Load datasets
    ###############
    if script_args.dataset_id_or_path.endswith('.json'):
        train_dataset = load_dataset('json', data_files=script_args.dataset_id_or_path, split='train')
    else:
        train_dataset = load_dataset(script_args.dataset_id_or_path, split=script_args.dataset_splits)
    
    # Select a sample of the required size.
    if script_args.num_samples is not None:
        train_dataset = train_dataset.select(range(script_args.num_samples))
    
    logger.info(f'Loaded dataset with {len(train_dataset)} samples and the following features: {train_dataset.features}')

   
    ################
    # Load tokenizer
    ################

    ########################
    # Function adeed by Bob:
    # IMPORTANT for the DeepSeek-R1 models: the chat template strips out the CoT before training, which is bad!
    # So we modify the Jinja2 template to not strip out the CoT.
    ########################
    def get_tokenizer_with_new_chat_template(tokenizer):
        to_delete = "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}"
        new_template = tokenizer.get_chat_template().replace(to_delete, "")
        return AutoTokenizer.from_pretrained(
            script_args.tokenizer_name_or_path if script_args.tokenizer_name_or_path else model_args.model_name_or_path,
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
            # This line is key!
            chat_template=new_template,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.tokenizer_name_or_path if script_args.tokenizer_name_or_path else model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Bob's edits:
    tokenizer = get_tokenizer_with_new_chat_template(tokenizer)
    tokenizer.padding_side = "right"

    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
   
    
    #######################
    # Load pretrained model
    #######################
    
    # define model kwargs
    model_kwargs = dict(
        revision=model_args.model_revision, # What revision from Huggingface to use, defaults to main
        trust_remote_code=model_args.trust_remote_code, # Whether to trust the remote code, this also you to fine-tune custom architectures
        attn_implementation=model_args.attn_implementation, # What attention implementation to use, defaults to flash_attention_2
        torch_dtype=model_args.torch_dtype if model_args.torch_dtype in ['auto', None] else getattr(torch, model_args.torch_dtype), # What torch dtype to use, defaults to auto
        use_cache=False if training_args.gradient_checkpointing else True, # Whether
        low_cpu_mem_usage=True if not strtobool(os.environ.get("ACCELERATE_USE_DEEPSPEED", "false")) else None,  # Reduces memory usage on CPU for loading the model
    )
    
    # Check which training method to use and if 4-bit quantization is needed
    if model_args.load_in_4bit: 
        model_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=model_kwargs['torch_dtype'],
            bnb_4bit_quant_storage=model_kwargs['torch_dtype'],
        )
    
    # We're using LoRA, so get the PEFT config
    peft_config = get_peft_config(model_args)
    
    # load the model with our kwargs
    if training_args.use_liger_kernel:
        model = AutoLigerKernelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load


    if script_args.spectrum_config_path:
        model = setup_model_for_spectrum(model, script_args.spectrum_config_path)

    ########################
    # Initialize the Trainer
    ########################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )
    if trainer.accelerator.is_main_process and peft_config:
        trainer.model.print_trainable_parameters()

    ###############
    # Training loop
    ###############
    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f'Checkpoint detected, resuming training at {last_checkpoint}.')

    logger.info(f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***')
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    # log metrics
    metrics = train_result.metrics
    metrics['train_samples'] = len(train_dataset)
    trainer.log_metrics('train', metrics)
    trainer.save_metrics('train', metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    
    logger.info('*** Save model ***')
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type('FULL_STATE_DICT')
    # Restore k,v cache for fast inference
    trainer.model.config.use_cache = True
    
    # First save the adapter only as a backup
    trainer.save_model(training_args.output_dir)
    logger.info(f"LoRA adapter saved to {training_args.output_dir}")
    
    # Then save the merged model if requested
    if script_args.save_merged_weights:
        merged_dir = save_merged_model(trainer, training_args.output_dir)
        # Update output_dir to merged model directory for model card creation
        merged_output_dir = merged_dir
    else:
        merged_output_dir = training_args.output_dir
    
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        tags = ['sft', 'lora']
        if script_args.save_merged_weights:
            tags.append('merged')
        trainer.create_model_card({'tags': tags})
    
    # push to hub if needed
    if training_args.push_to_hub is True:
        logger.info('Pushing to hub...')
        trainer.push_to_hub()

    logger.info('*** Training complete! ***')


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, SFTConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()
    # logger.info(f"Script arguments: {script_args}")
    # logger.info(f"Model arguments: {model_args}")
    # logger.info(f"Training arguments: {training_args}")
    # with open(script_args.output_dir + "/script_args.json", "w") as f:
    #     f.write(script_args.to_json_string())
        
    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Run the main training loop
    train_function(model_args, script_args, training_args)


if __name__ == '__main__':
    main()
