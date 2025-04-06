# Gemma Finetune UI

## Overview
I have made this repo has an exploration for the project [Gemma Model Fine-tuning UI](https://gist.github.com/dynamicwebpaige/92f7739ad69d2863ac7e2032fe52fbad#11-gemma-model-projects-). It includes a _working prototype_ of a gradio interface that can be used to fine-tune Gemma 3 models. Other than that, it also includes all the notebooks and code that I have used to experiment and explore the fine-tuning process.

To learn how you can use it to fine tune your own Gemma 3 models, please refer to the [Usage](#usage) section.

## Demo

YouTube video: https://youtu.be/NKV788bQEgE

## Repo Structure

![tree-structure](https://github.com/user-attachments/assets/65cbffc8-3770-45df-abf8-40527a729638)

## Usage

YouTube video: https://youtu.be/KKHP4wLV_I8

There are two ways to use this app. 

1. Run the app locally
2. Using the app on Google Colab

### Run the app locally

This is not recommended because it requires you to have a GPU and a lot of memory. However, if you still want to try it, here's what you can do:

1. Clone the repo

    ```bash
    git clone https://github.com/inclinedadarsh/gemma-finetune-ui.git
    ```

2. Create a virtual environment

    You can use either conda or venv to create a virtual environment.

    ```bash
    conda create -n gemma-finetune-ui python=3.11
    ```

    ```bash
    conda activate gemma-finetune-ui
    ```

3. Install the dependencies

    First, install PyTorch from the [official website](https://pytorch.org/get-started/locally/).

    Then, install the other dependencies.

    ```bash
    pip install -U gradio transformers peft datasets trl bitsandbytes
    ```

4. Run the app

    ```bash
    gradio app/app.py
    ```

    If you want to run the conversion app, you can do so by running:

    ```bash
    gradio app/conversion.py
    ```

    If you want to run the inference app, you can do so by running:

    ```bash
    gradio app/inference.py
    ```


### Using the app on Google Colab

This is the recommended way to use the app.

1. Open the [gemma-finetune-gradio](./notebooks/app/gemma-finetune-gradio.ipynb) notebook, and click on the `Open in Colab` button.

2. Change the runtime type to T4 GPU.

3. Run all the cells.

4. Visit the link from the output of the last cell.












