import os
import gradio as gr
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

HF_TOKEN = os.getenv("HF_TOKEN")


def merge_model(model_id, model_is_merged):
    """
    If the model is not merged, this function loads the PEFT model,
    merges it with its base model, and saves the merged model to './merged_model'.
    If the model is already merged, it skips merging.
    """
    try:
        if model_is_merged:
            return "Merge skipped. Using existing merged model folder: './merged_model'"
        else:
            peft_config = PeftConfig.from_pretrained(model_id)
            base_model_id = peft_config.base_model_name_or_path

            tokenizer = AutoTokenizer.from_pretrained(base_model_id)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                device_map="auto",
                torch_dtype=torch.float16,
                attn_implementation="eager",
            )

            model = PeftModel.from_pretrained(base_model, model_id)

            merged_model = model.merge_and_unload()

            os.makedirs("./merged_model", exist_ok=True)
            merged_model.save_pretrained("./merged_model")
            tokenizer.save_pretrained("./merged_model")
            return "Merging completed and saved to './merged_model'"
    except Exception as e:
        return f"Error during merging: {e}"


def convert_to_gguf():
    """
    This function clones the llama.cpp repository (if needed) and runs the conversion
    from the merged Hugging Face model (in './merged_model') to GGUF format.
    The output GGUF file is saved in './merged_model/merged_model.gguf' and its path is returned.
    """
    try:
        if not os.path.exists("llama.cpp"):
            os.system("git clone https://github.com/ggerganov/llama.cpp.git")

        orig_dir = os.getcwd()
        os.chdir("llama.cpp")

        merged_model_path = os.path.join("..", "merged_model")
        outfile = os.path.join("..", "merged_model", "merged_model.gguf")

        conversion_command = (
            f"python convert_hf_to_gguf.py {merged_model_path} --outfile {outfile}"
        )
        os.system(conversion_command)

        os.chdir(orig_dir)

        if os.path.exists("./merged_model/merged_model.gguf"):
            return "./merged_model/merged_model.gguf"
        else:
            return "Conversion finished but the output file was not found."
    except Exception as e:
        return f"Error during conversion: {e}"


with gr.Blocks() as demo:
    gr.Markdown("# Hugging Face Model Merger and GGUF Converter")

    with gr.Column():
        gr.Markdown("## Step 1: Merge Model")
        model_id = gr.Textbox(
            label="Hugging Face Model ID",
            placeholder="e.g., inclinedadarsh/gemma-3-1b-nl-to-regex",
        )
        model_is_merged = gr.Checkbox(label="Model is already merged", value=False)
        merge_button = gr.Button("Merge Model")
        merge_status = gr.Textbox(label="Merge Status", interactive=False)

        merge_button.click(
            fn=merge_model, inputs=[model_id, model_is_merged], outputs=merge_status
        )

    with gr.Column():
        gr.Markdown("## Step 2: Convert to GGUF")
        convert_button = gr.Button("Convert to GGUF")

        gguf_file = gr.File(label="GGUF File (click to download)")

        convert_button.click(fn=convert_to_gguf, inputs=[], outputs=gguf_file)

    gr.Markdown(
        "**Note:** Ensure that the merged model folder (`./merged_model`) exists before converting."
    )


demo.launch()
