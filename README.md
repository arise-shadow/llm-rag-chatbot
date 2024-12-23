# Test Performance of RNGD with various LLM tasks

This repository contains a chatbot implementation that combines Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) techniques to provide accurate and contextually relevant responses. On the other hand, we also support translation between English and Korean based on LLM translator. 

## Project Structure

├── LICENSE             # Open-source license information
├── README.md           # Project overview and instructions
├── data/
│   ├── db/             # Database made for warboy and rngd sdk 
│   ├── translate/      # Database made for translation (Kor <-> Eng)
│   ├── chatbot/        # QA dataset for warboy and rngd sdk
│   └── handmade-faq/   # QA obtained from furiosa customer portal
├── docs/               # Documentation for the project
├── notebooks/          # Jupyter notebooks for exploration and analysis
├── references/         # Data dictionaries, manuals, and other reference materials
├── requirements.txt    # List of required Python packages
└── src/                # Source code for the chatbot


## Getting Started

To set up the project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/arise-shadow/llm-rag-chatbot.git

	2.	Navigate to the project directory:

cd llm-rag-chatbot


	3.	Create and activate a virtual environment (optional but recommended):

python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate


	4.	Install the required packages:

pip install -r requirements.txt



Usage

Detailed instructions on how to use the chatbot, including data preparation, model training, and running the application, can be found in the docs directory.

Contributing

Contributions are welcome! Please refer to the CONTRIBUTING.md file for guidelines on how to contribute to this project.

License

This project is licensed under the terms of the MIT License.

This `README.md` provides a comprehensive overview of your project, including its structure, setup instructions, usage guidelines, and acknowledgements. Feel free to modify any sections to better fit your project's specifics. 