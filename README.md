Text Generation with Fine-Tuned GPT-2
Overview
This project demonstrates a text generation system built by fine-tuning the GPT-2 model (HuggingFace Transformers) on a public-domain text dataset. The model generates coherent narrative text, achieving a perplexity score of approximately 20 after training. A FastAPI-based REST API enables real-time text generation, showcasing generative AI capabilities for applications like automated content creation. This personal project was developed to explore Large Language Models (LLMs) and aligns with research interests in natural language processing and AI-driven automation.
Features

Fine-Tuned GPT-2: Trained gpt2 model on the Tiny Shakespeare dataset for narrative text generation.
Text Preprocessing: Implemented tokenization and data cleaning using HuggingFace tokenizers and Python.
API Deployment: Built a FastAPI endpoint (/generate) for real-time text generation from user prompts.
Evaluation: Achieved a perplexity score of ~20, indicating strong language modeling performance.
Documentation: Includes Jupyter Notebook for training and evaluation, with results visualized using Matplotlib.

Technologies

Programming: Python
Libraries/Frameworks: HuggingFace Transformers, PyTorch, FastAPI, Pandas, Matplotlib
Tools: Git, VS Code, Anaconda
Dataset: Tiny Shakespeare (public domain)

Setup Instructions

Clone the Repository:git clone https://github.com/yourusername/gpt2-text-gen.git
cd gpt2-text-gen


Install Dependencies:pip install -r requirements.txt

Requirements include: transformers, torch, fastapi, uvicorn, pandas, matplotlib.
Download Dataset:
Download the Tiny Shakespeare dataset and place it in the data/ folder.


Fine-Tune GPT-2:
Run the training script:python train_gpt2.py


Alternatively, use the Jupyter Notebook (notebooks/train.ipynb) for step-by-step training.


Run the API:
Start the FastAPI server:uvicorn app.main:app --host 0.0.0.0 --port 8000


Test the API:curl -X GET "http://localhost:8000/generate?prompt=Once%20upon%20a%20time"





Usage

Training: Use train_gpt2.py or notebooks/train.ipynb to fine-tune GPT-2 on your dataset. Adjust hyperparameters in config.yaml (e.g., epochs, batch size).
Text Generation: Access the API at http://localhost:8000/generate with a prompt query parameter to generate text.
Evaluation: Run evaluate.py to compute perplexity and visualize training loss with Matplotlib.

Results

Perplexity: Achieved ~20 on the validation set, indicating effective fine-tuning.
Sample Output:Prompt: "Once upon a time"
Output: "Once upon a time, in a kingdom far away, the people gathered to hear tales of valor and love, as the bard spoke of heroes past..."


Visualization: Training loss curves are available in notebooks/train.ipynb.

Project Structure
gpt2-text-gen/
├── app/                  # FastAPI application
│   └── main.py           # API endpoint for text generation
├── data/                 # Dataset folder (e.g., shakespeare.txt)
├── notebooks/            # Jupyter Notebooks for training/evaluation
│   └── train.ipynb
├── train_gpt2.py        # Training script
├── evaluate.py           # Evaluation script
├── config.yaml           # Hyperparameters
├── requirements.txt      # Dependencies
└── README.md             # Project documentation

Future Improvements

Enhance model performance by experimenting with larger datasets or advanced models (e.g., GPT-Neo).
Integrate a React front-end for interactive text generation.
Add support for multi-prompt generation and style transfer.

Acknowledgments

Built as a personal project to explore LLMs and generative AI.
Inspired by HuggingFace tutorials and open-source NLP resources.

Contact
For questions or collaboration, reach out via GitHub Issues.
