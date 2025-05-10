"""
screen -S myexperiment
accelerate launch --num_processes 1 sft.py --config sft.yaml
accelerate launch --config_file fsdp_config.yaml sft.py --config sft.yaml
"""

from dataclasses import dataclass
from datetime import datetime
from distutils.util import strtobool
import logging
import os
import re
import yaml
from typing import Optional
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import is_liger_kernel_available
from trl import SFTTrainer, TrlParser, ModelConfig, SFTConfig
from datasets import load_dataset

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

# def train_function(model_args: ModelConfig, script_args: ScriptArguments, training_args: SFTConfig):
#     """Main training function."""
#     #########################
#     # Log parameters
#     #########################
#     logger.info(f'Model parameters {model_args}')
#     logger.info(f'Script parameters {script_args}')
#     logger.info(f'Training/evaluation parameters {training_args}')

#     ###############
#     # Load datasets
#     ###############
#     if script_args.dataset_id_or_path.endswith('.json'):
#         train_dataset = load_dataset('json', data_files=script_args.dataset_id_or_path, split='train', streaming=True)
#     else:
#         train_dataset = load_dataset(script_args.dataset_id_or_path, split=script_args.dataset_splits, streaming=True)
    
#     # Select a sample of the required size.
#     if script_args.num_samples is not None:
#         train_dataset = train_dataset.take(script_args.num_samples)
    
#     # Convert to IterableDataset format
#     train_dataset = train_dataset.with_format("torch")
    
#     # We can't directly get dataset length with IterableDataset, so we log available features
#     logger.info(f'Loaded streaming dataset with the following features: {train_dataset.features}')

   
#     ################
#     # Load tokenizer
#     ################

#     ########################
#     # DeepSeek-R1 models: the chat template strips out the CoT before training, which is bad!
#     # So we modify the Jinja2 template to not strip out the CoT.
#     ########################
#     def get_tokenizer_with_new_chat_template(tokenizer):
#         to_delete = "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}"
#         new_template = tokenizer.get_chat_template().replace(to_delete, "")
#         return AutoTokenizer.from_pretrained(
#             script_args.tokenizer_name_or_path if script_args.tokenizer_name_or_path else model_args.model_name_or_path,
#             revision=model_args.model_revision,
#             trust_remote_code=model_args.trust_remote_code,
#             # This line is key!
#             chat_template=new_template,
#         )

#     tokenizer = AutoTokenizer.from_pretrained(
#         script_args.tokenizer_name_or_path if script_args.tokenizer_name_or_path else model_args.model_name_or_path,
#         revision=model_args.model_revision,
#         trust_remote_code=model_args.trust_remote_code,
#     )

#     tokenizer = get_tokenizer_with_new_chat_template(tokenizer)
#     tokenizer.padding_side = "right"

#     if tokenizer.pad_token is None: 
#         tokenizer.pad_token = tokenizer.eos_token
   
    
#     #######################
#     # Load pretrained model
#     #######################
    
#     # define model kwargs
#     model_kwargs = dict(
#         revision=model_args.model_revision, 
#         trust_remote_code=model_args.trust_remote_code,
#         attn_implementation=model_args.attn_implementation,
#         torch_dtype=model_args.torch_dtype if model_args.torch_dtype in ['auto', None] else getattr(torch, model_args.torch_dtype),
#         use_cache=False if training_args.gradient_checkpointing else True,
#         low_cpu_mem_usage=True if not strtobool(os.environ.get("ACCELERATE_USE_DEEPSPEED", "false")) else None,
#     )
    
#     # load the model with our kwargs
#     if training_args.use_liger_kernel:
#         model = AutoLigerKernelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
#     else:
#         model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
#     training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

#     # If using Spectrum, set up the model accordingly
#     if script_args.spectrum_config_path:
#         model = setup_model_for_spectrum(model, script_args.spectrum_config_path)

#     ########################
#     # Initialize the Trainer
#     ########################
#     with open("/disk/u/harshraj/CotIF/src/train/fsdp_config.yaml", "r") as file:
#         distributed_config = yaml.safe_load(file)
#     training_args.fsdp = "full_shard auto_wrap"
#     training_args.fsdp_config = distributed_config['fsdp_config']
#     trainer = SFTTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         tokenizer=tokenizer,
#         peft_config=None,  # No PEFT config for full fine-tuning
#         dataset_text_field="messages",
#     )

#     # For full fine-tuning, it's helpful to see the proportion of trainable parameters
#     if trainer.accelerator.is_main_process:
#         # Count trainable parameters
#         trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         total_params = sum(p.numel() for p in model.parameters())
#         logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")

#     ###############
#     # Training loop
#     ###############
#     # Check for last checkpoint
#     last_checkpoint = get_checkpoint(training_args)
#     if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
#         logger.info(f'Checkpoint detected, resuming training at {last_checkpoint}.')

#     logger.info(f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***')
#     train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
#     # log metrics
#     metrics = train_result.metrics
#     # For IterableDataset we can't directly access length, so we use approx count if available or skip
#     if hasattr(train_dataset, '_approx_sample_count') and train_dataset._approx_sample_count is not None:
#         metrics['train_samples'] = train_dataset._approx_sample_count
#     trainer.log_metrics('train', metrics)
#     trainer.save_metrics('train', metrics)
#     trainer.save_state()

#     ##################################
#     # Save model and create model card
#     ##################################
    
#     logger.info('*** Save model ***')
#     if trainer.is_fsdp_enabled:
#         trainer.accelerator.state.fsdp_plugin.set_state_dict_type('FULL_STATE_DICT')
#     # Restore k,v cache for fast inference
#     trainer.model.config.use_cache = True
#     trainer.save_model(training_args.output_dir)
#     logger.info(f'Model saved to {training_args.output_dir}')
#     training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

#     tokenizer.save_pretrained(training_args.output_dir)
#     logger.info(f'Tokenizer saved to {training_args.output_dir}')

#     # Save everything else on main process
#     if trainer.accelerator.is_main_process:
#         trainer.create_model_card({'tags': ['sft', 'full-finetuning']})
#     # push to hub if needed
#     if training_args.push_to_hub is True:
#         logger.info('Pushing to hub...')
#         trainer.push_to_hub()

#     logger.info('*** Training complete! ***')
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
    # DeepSeek-R1 models: the chat template strips out the CoT before training, which is bad!
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

    tokenizer = get_tokenizer_with_new_chat_template(tokenizer)
    tokenizer.padding_side = "right"

    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
   
    
    #######################
    # Load pretrained model
    #######################
    
    # define model kwargs
    model_kwargs = dict(
        revision=model_args.model_revision, 
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype if model_args.torch_dtype in ['auto', None] else getattr(torch, model_args.torch_dtype),
        use_cache=False if training_args.gradient_checkpointing else True,
        low_cpu_mem_usage=True if not strtobool(os.environ.get("ACCELERATE_USE_DEEPSPEED", "false")) else None,
    )
    
    # load the model with our kwargs
    if training_args.use_liger_kernel:
        model = AutoLigerKernelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    # If using Spectrum, set up the model accordingly
    if script_args.spectrum_config_path:
        model = setup_model_for_spectrum(model, script_args.spectrum_config_path)

    ########################
    # Initialize the Trainer
    ########################
    with open("/disk/u/harshraj/CotIF/src/train/fsdp_config.yaml", "r") as file:
        distributed_config = yaml.safe_load(file)
    training_args.fsdp = "full_shard auto_wrap"
    training_args.fsdp_config = distributed_config['fsdp_config']
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=None,  # No PEFT config for full fine-tuning
    )

    # For full fine-tuning, it's helpful to see the proportion of trainable parameters
    if trainer.accelerator.is_main_process:
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")

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
    trainer.save_model(training_args.output_dir)
    logger.info(f'Model saved to {training_args.output_dir}')
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f'Tokenizer saved to {training_args.output_dir}')

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({'tags': ['sft', 'full-finetuning']})
    # push to hub if needed
    if training_args.push_to_hub is True:
        logger.info('Pushing to hub...')
        trainer.push_to_hub()

    logger.info('*** Training complete! ***')


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, SFTConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Run the main training loop
    train_function(model_args, script_args, training_args)


if __name__ == '__main__':
    main()