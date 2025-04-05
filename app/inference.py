import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

HF_TOKEN = os.getenv("HF_TOKEN")

# A global container to store the loaded model and tokenizer
model_container = {"model": None, "tokenizer": None}


def load_model(selected_model, custom_model):
    """
    Load model and tokenizer from Hugging Face.
    If a custom model name is provided (non-empty), it takes precedence.
    """
    model_name = custom_model.strip() if custom_model.strip() != "" else selected_model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN)
        # Set the model to evaluation mode and, if available, to the appropriate device
        model.eval()
        if torch.cuda.is_available():
            model.to("cuda")
        return model, tokenizer, f"Successfully loaded model: {model_name}"
    except Exception as e:
        return None, None, f"Error loading model {model_name}: {str(e)}"


def load_and_set_model(selected_model, custom_model):
    """
    Wrapper function to load the model and update the global container.
    """
    model, tokenizer, status = load_model(selected_model, custom_model)
    model_container["model"] = model
    model_container["tokenizer"] = tokenizer
    return status


def generate_response(message, history):
    """
    Generate a response using the loaded model.
    For simplicity, we are only encoding the latest message.
    You can extend this to incorporate conversation history.
    """
    if model_container["model"] is None:
        return "Model is not loaded yet!"
    tokenizer = model_container["tokenizer"]
    model = model_container["model"]

    # Encode the message and generate a response
    inputs = tokenizer.encode(message, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    outputs = model.generate(inputs, max_length=100, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def chat(message, history):
    """
    Chat function to handle each incoming message.
    """
    if model_container["model"] is None:
        history.append(("System", "Please load a model first."))
        return "", history
    response = generate_response(message, history)
    history.append((message, response))
    return "", history


# Replace these with your actual gemma model names
gemma_models = [
    "google/gemma-3-1b-it",
    "google/gemma-3-1b-pt",
    "google/gemma-3-4b-it",
    "google/gemma-3-4b-pt",
    "google/gemma-3-27b-it",
    "google/gemma-3-27b-pt",
    "google/gemma-3-12b-it",
    "google/gemma-3-12b-pt",
]

with gr.Blocks() as demo:
    gr.Markdown("## Hugging Face Model Chat App")
    with gr.Row():
        # Dropdown for gemma models and textbox for custom model name, side by side
        dropdown = gr.Dropdown(
            label="Select a Gemma Model", choices=gemma_models, value=gemma_models[0]
        )
        custom_input = gr.Textbox(
            label="Or enter a custom model name", placeholder="username/model-name"
        )

    load_button = gr.Button("Load Model")
    load_status = gr.Textbox(label="Status", interactive=False)

    # Chat UI
    chatbot = gr.Chatbot(label="Chat with Model")
    message_input = gr.Textbox(
        label="Your Message", placeholder="Type your message here..."
    )

    # Button click to load the model using the provided input
    load_button.click(
        fn=load_and_set_model, inputs=[dropdown, custom_input], outputs=load_status
    )

    # Submit message to chat, updating the chatbot interface
    message_input.submit(
        fn=chat, inputs=[message_input, chatbot], outputs=[message_input, chatbot]
    )

demo.launch()
