import gradio as gr
import torch
import time
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import Dataset

MODEL_OPTIONS = [
    "google/gemma-3-1b-it", "google/gemma-3-1b-pt",
    "google/gemma-3-4b-it", "google/gemma-3-4b-pt",
    "google/gemma-3-12b-it", "google/gemma-3-12b-pt",
    "google/gemma-3-27b-it", "google/gemma-3-27b-pt"
]

def load_model_fn(model_name, load_4bit, hf_token):
    logs = []
    model = None
    tokenizer = None
    if not hf_token.strip():
        logs.append("Error: Please enter Hugging Face token.")
        return "\n".join(logs), None, None

    logs.append(f"Starting to load model: {model_name}...")
    time.sleep(0.5)
    try:
        if load_4bit:
            logs.append("4-bit option selected. Setting up BitsAndBytes config...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_storage=torch.float16
            )
            logs.append("Loading model with 4-bit quantization. This may take a while...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                attn_implementation='eager',
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map='auto',
                token=hf_token
            )
        else:
            logs.append("Loading model without 4-bit quantization...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                attn_implementation='eager',
                torch_dtype=torch.float16,
                device_map='auto',
                token=hf_token
            )

        for percent in range(10, 101, 30):
            time.sleep(0.5)
            logs.append(f"Model loading progress: {percent}%...")

        logs.append("Model loaded successfully.")
        logs.append("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        logs.append("Tokenizer loaded successfully.")
        logs.append("Gemma model and tokenizer are ready for use.")
        return "\n".join(logs), model, tokenizer
    except Exception as e:
        logs.append(f"Error loading model: {str(e)}")
        return "\n".join(logs), None, None

def display_dataset(file):
    if file is None:
        return None
    try:
        # Handle file object or path string
        file_path = file["name"] if isinstance(file, dict) and "name" in file else file
        df = pd.read_csv(file_path)
        if 'user' not in df.columns or 'assistant' not in df.columns:
            return pd.DataFrame({"Error": ["CSV must have 'user' and 'assistant' columns."]})
        return df
    except Exception as e:
        return pd.DataFrame({"Error": [f"Error reading CSV: {str(e)}"]})

def process_dataset(file, system_message):
    if file is None:
        return {"error": "No file uploaded."}, None
    try:
        file_path = file["name"] if isinstance(file, dict) and "name" in file else file
        df = pd.read_csv(file_path)
        if 'user' not in df.columns or 'assistant' not in df.columns:
            return {"error": "CSV must have 'user' and 'assistant' columns."}, None

        ds = Dataset.from_pandas(df)

        def format_example(example):
            return {
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": example['user']},
                    {"role": "assistant", "content": example['assistant']}
                ]
            }

        ds = ds.map(format_example, remove_columns=ds.column_names, batched=False)
        sample = ds[0]
        return sample, ds
    except Exception as e:
        return {"error": f"Error processing dataset: {str(e)}"}, None

def fine_tune_fn(model, tokenizer, dataset, use_lora, rank, alpha, dropout, epochs, learning_rate, max_seq_length, optim, push_to_hub, repo_id, hf_token):
    logs = []
    if model is None:
        return "Error: Model not loaded.", None
    if tokenizer is None:
        return "Error: Tokenizer not loaded.", None
    if dataset is None:
        return "Error: Processed dataset not loaded.", None

    logs.append("Starting fine tuning process...")
    if use_lora:
        from peft import LoraConfig
        peft_config = LoraConfig(
            lora_alpha=alpha,
            lora_dropout=dropout,
            r=rank,
            bias="none",
            target_modules=['q_proj', 'k_proj', 'v_proj'],
            task_type="CAUSAL_LM",
        )
        logs.append("LoRA configuration set.")
    else:
        peft_config = None
        logs.append("LoRA not used. Proceeding without PEFT configuration.")

    # Build training configuration (with fixed parameters for this prototype)
    from trl import SFTConfig
    training_args = SFTConfig(
        output_dir="./gemma-finetune",
        max_seq_length=max_seq_length,
        packing=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim=optim,
        logging_steps=2,
        save_strategy='epoch',
        learning_rate=learning_rate,
        fp16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type='constant',
        report_to='none',
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": True
        },
        push_to_hub=push_to_hub,
        hub_model_id=repo_id,
        hub_token=hf_token
    )
    logs.append("Training configuration set. Starting trainer...")

    from trl import SFTTrainer
    try:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            peft_config=peft_config,
            processing_class=tokenizer,
            train_dataset=dataset
        )
        logs.append("Training started...")
        trainer.train()
        logs.append("Training complete.")
        return "\n".join(logs), trainer
    except Exception as e:
        return f"Error during training: {str(e)}", None

def push_to_hub_fn(trainer, repo_id):
    if trainer is None:
        return "Error: Model not trained yet."
    if not trainer.push_to_hub:
        return "Error: Push to Hub wasn't enabled!"
    try:
        trainer.push_to_hub()
        return f"Model successfully pushed to the Hugging Face Hub at {repo_id}!"
    except Exception as e:
        return f"Error pushing model to hub: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# Gemma Model Loader, Dataset Processor, Fine Tuner & Hub Pusher")
    gr.Markdown("Use the sidebar for model configuration and system message. Then upload and process a CSV dataset, fine tune the model, and finally push it to the Hugging Face Hub.")

    model_state = gr.State()
    tokenizer_state = gr.State()
    dataset_state = gr.State()
    trained_model_state = gr.State()

    with gr.Sidebar():
        gr.Markdown("## Model Configuration")
        model_dropdown = gr.Dropdown(choices=MODEL_OPTIONS, label="Select Model", value=MODEL_OPTIONS[0])
        load_4bit_checkbox = gr.Checkbox(label="Load model in 4 bit? (Saves ton of memory)", value=False)
        hf_token_input = gr.Textbox(label="Hugging Face Token", placeholder="Enter your Hugging Face token", type="password")
        gr.Markdown("## System Message")
        system_message_input = gr.Textbox(label="System Message", placeholder="Enter system message here")
        load_model_button = gr.Button("Load Model")

    with gr.Column():
        gr.Markdown("### Model Loading Log")
        model_log = gr.Textbox(label="Loading Log", lines=15)
        load_model_button.click(load_model_fn,
                                inputs=[model_dropdown, load_4bit_checkbox, hf_token_input],
                                outputs=[model_log, model_state, tokenizer_state])

        gr.Markdown("---")
        gr.Markdown("## Dataset Loader")
        csv_file = gr.File(label="Upload CSV", file_types=['.csv'])
        dataset_preview = gr.Dataframe(label="Dataset Preview")
        process_button = gr.Button("Process Dataset")
        processed_output = gr.JSON(label="Sample Processed Output")

        csv_file.change(fn=display_dataset, inputs=csv_file, outputs=dataset_preview)

        process_button.click(process_dataset,
                             inputs=[csv_file, system_message_input],
                             outputs=[processed_output, dataset_state])

        gr.Markdown("---")
        gr.Markdown("## Fine Tuning Configuration")
        with gr.Row():
            use_lora_checkbox = gr.Checkbox(label="Use LoRA", value=False)
            rank_input = gr.Number(label="LoRA Rank (r)", value=8)
            alpha_input = gr.Number(label="LoRA Alpha", value=8)
            dropout_input = gr.Number(label="LoRA Dropout", value=0.05)
        with gr.Row():
            epochs_input = gr.Number(label="Epochs", value=3)
            lr_input = gr.Number(label="Learning Rate", value=2e-4)
        with gr.Row():
            max_seq_length_input = gr.Number(label="Max Seq Length", value=512)
            optim_dropdown = gr.Dropdown(label="Optimizer", choices=["adamw_torch_fused", "adamw"], value="adamw_torch_fused")
        with gr.Row():
            push_to_hub_checkbox = gr.Checkbox(label="Push to Hub", value=False)
            repo_name_input = gr.Textbox(label="Repository Name", placeholder="Enter repository name where the model should be pushed")

        training_log = gr.Textbox(label="Training Log", lines=15)
        start_training_button = gr.Button("Start Training")

        start_training_button.click(fine_tune_fn,
                                    inputs=[model_state, tokenizer_state, dataset_state,
                                            use_lora_checkbox, rank_input, alpha_input, dropout_input,
                                            epochs_input, lr_input, max_seq_length_input, optim_dropdown, push_to_hub_checkbox, repo_name_input, hf_token_input],
                                    outputs=[training_log, trained_model_state])

        gr.Markdown("---")
        gr.Markdown("## Push Fine-Tuned Model to Hugging Face Hub")
        push_to_hub_button = gr.Button("Push to Hub")
        push_log = gr.Textbox(label="Push Log", lines=8)
        push_to_hub_button.click(push_to_hub_fn,
                                 inputs=[trained_model_state, repo_name_input],
                                 outputs=push_log)

demo.launch()

