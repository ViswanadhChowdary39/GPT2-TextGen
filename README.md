Text Generation with Fine-Tuned GPT-2
Overview
This project implements a text generation system by fine-tuning the GPT-2 model from HuggingFace Transformers on the Tiny Shakespeare dataset, a public-domain text corpus. The fine-tuned model generates coherent narrative text, achieving a perplexity score of approximately 20. A FastAPI-based REST API enables real-time text generation, demonstrating practical applications in generative AI. Developed as a personal project to explore Large Language Models (LLMs), this work enhances skills in natural language processing (NLP) and AI-driven automation, aligning with research interests in LLMs and text processing.
Features

Model Fine-Tuning: Fine-tuned gpt2 model on Tiny Shakespeare for narrative text generation.
Text Preprocessing: Applied tokenization and cleaning using HuggingFace tokenizers and Python.
API Deployment: Deployed a FastAPI endpoint (/generate) for real-time text generation from user prompts.
Evaluation: Measured model performance with a perplexity score of ~20, indicating effective language modeling.
Visualization: Included training loss curves in a Jupyter Notebook, visualized with Matplotlib.

Technologies

Programming: Python
Libraries/Frameworks: HuggingFace Transformers, PyTorch, FastAPI, Pandas, Matplotlib
Tools: Git, VS Code, Anaconda, Jupyter Notebook
Dataset: Tiny Shakespeare (public domain)

Installation

Clone the Repository:git clone https://github.com/yourusername/gpt2-text-gen.git
cd gpt2-text-gen


Install Dependencies:pip install -r requirements.txt

Dependencies include transformers, torch, fastapi, uvicorn, pandas, and matplotlib.
Download Dataset:
Place Tiny Shakespeare in the data/ directory.


Optional: Download pre-trained model weights (if not fine-tuning):
Run python download_model.py to fetch gpt2 weights.



Usage

Fine-Tuning:
Run the training script:python train_gpt2.py


Or use notebooks/train.ipynb for interactive training.
Configure hyperparameters (e.g., epochs, batch size) in config.yaml.


Text Generation:
Start the FastAPI server:uvicorn app.main:app --host 0.0.0.0 --port 8000


Generate text via API:curl -X GET "http://localhost:8000/generate?prompt=Once%20upon%20a%20time"




Evaluation:
Run evaluate.py to compute perplexity and visualize loss:python evaluate.py





Results

Perplexity: Approximately 20 on the validation set, reflecting strong model performance.
Sample Output:Prompt: "Once upon a time"
Output: "Once upon a time, in a land of ancient tales, the bard wove stories of knights and dreams..."


Visualization: Training loss curves are available in notebooks/train.ipynb, plotted with Matplotlib.

Project Structure
gpt2-text-gen/
├── app/                  # FastAPI application
│   └── main.py           # API endpoint for text generation
├── data/                 # Dataset (e.g., shakespeare.txt)
├── notebooks/            # Jupyter Notebooks
│   └── train.ipynb       # Training and evaluation
├── train_gpt2.py         # Training script
├── evaluate.py           # Evaluation script
├── download_model.py     # Script to download model weights
├── config.yaml           # Hyperparameters
├── requirements.txt      # Dependencies
└── README.md             # Documentation

Setup Requirements

Hardware: Laptop with 8GB RAM (GPU optional; use Google Colab for faster training).
OS: Windows/Linux/MacOS
Python: Version 3.8+
Storage: ~2GB for dataset and model weights

Future Improvements

Experiment with larger datasets or models (e.g., GPT-Neo) for enhanced performance.
Develop a React-based front-end for interactive text generation.
Implement multi-prompt generation and style-specific outputs.

Acknowledgments

Built as a personal project to advance skills in LLMs and NLP.
Utilized resources from HuggingFace documentation and open-source NLP tutorials.

Contact
For feedback or inquiries, create an issue on GitHub.
