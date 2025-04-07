# Gemma Finetune UI

## Overview

**Demo video:**: https://youtu.be/NKV788bQEgE

I have made this repo has an exploration for the project [Gemma Model Fine-tuning UI](https://gist.github.com/dynamicwebpaige/92f7739ad69d2863ac7e2032fe52fbad#11-gemma-model-projects-). It includes a _working prototype_ of a gradio interface that can be used to fine-tune Gemma 3 models. Other than that, it also includes all the notebooks and code that I have used to experiment and explore the fine-tuning process.

To learn how you can use it to fine tune your own Gemma 3 models, please refer to the [Usage](#usage) section.

## Repo Structure

![tree-structure](https://github.com/user-attachments/assets/65cbffc8-3770-45df-abf8-40527a729638)

Here's a more detailed view of the repo structure:

- `app/`: Contains the gradio app.
  - `app.py`: The main app.
  - `conversion.py`: A tool to convert a huggingface dataset to a format that can be used by the app.
  - `inference.py`: A tool to use the fine-tuned model for inference.
- `notebooks/`: Contains the notebooks used to explore the fine-tuning process.
  - `app/`: Contains the notebook for the gradio app. You can open it in Google Colab by clicking on the `Open in Colab` button.
    - [`gemma-finetune-gradio.ipynb`](./notebooks/app/gemma-finetune-gradio.ipynb): Notebook to start the main app in Google Colab.
    - [`merging_model_gradio.ipynb`](./notebooks/app/merging_model_gradio.ipynb): Notebook to start the conversion app in Google Colab.
  - `failed-attempts/`: Contains the notebooks of my failed attempts at fine-tuning the model.
  - `fine_tuning/`: Contains the notebooks of my successful attempts at fine-tuning the model.
    - [`gemma_keras.ipynb`](./notebooks/fine_tuning/gemma_keras.ipynb): A notebook that uses the Keras API to fine-tune the model.
    - [`gemma_transformers.ipynb`](./notebooks/fine_tuning/gemma_transformers.ipynb): A notebook that uses the Transformers API to fine-tune the model.
    - [`gemma-finetune-nl-to-regex.ipynb`](./notebooks/fine_tuning/gemma-finetune-nl-to-regex.ipynb): A notebook that uses the Transformers API to fine-tune the model for the task of converting natural language to regex. You can check out the model at https://huggingface.co/inclinedadarsh/gemma-3-1b-nl-to-regex
  - `other_notebooks/`: Contains the notebooks of other experiments.
    - [`dataset-preparation.ipynb`](./notebooks/other_notebooks/dataset-preparation.ipynb): A notebook that prepares an example dataset for fine-tuning the model.
    - [`merging_model.ipynb`](./notebooks/other_notebooks/merging_model.ipynb): A notebook that merges the weights of two models.
- `dataset/`: Contains a sample dataset used for fine-tuning the model.

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












