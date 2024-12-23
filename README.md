# Test Performance of RNGD with Various LLM Tasks

This repository contains a chatbot implementation that combines Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) techniques to provide accurate and contextually relevant responses. Additionally, it supports translation between English and Korean based on an LLM-based translator.

## Project Structure

The project is organized as follows:

```
├── LICENSE              # Open-source license information
├── README.md            # Project overview and instructions
├── data/
│   ├── db/              # Database for Warboy and RNGD SDK
│   ├── translate/       # Translation database (Kor ↔ Eng)
│   ├── chatbot/         # QA dataset for Warboy and RNGD SDK
│   └── handmade-faq/    # QA data from Furiosa customer portal
├── docs/                # Documentation for the project
├── notebooks/           # Jupyter notebooks for exploration and analysis
├── references/          # Data dictionaries, manuals, and other reference materials
├── requirements.txt     # List of required Python packages
└── src/                 # Source code for the chatbot
```

## Getting Started

Follow these steps to set up the project locally:

### 1. Clone the repository

```bash
git clone https://github.com/arise-shadow/llm-rag-chatbot.git
```

### 2. Navigate to the project directory

```bash
cd llm-rag-chatbot
```

### 3. Create and activate a virtual environment (optional but recommended)

```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### 4. Install the required packages

```bash
pip install -r requirements.txt
```

## Usage

Detailed instructions on how to use the chatbot, including data preparation, model training, and running the application, can be found in the `docs` directory.

## Contributing

Contributions are welcome! Please refer to the `CONTRIBUTING.md` file for guidelines on how to contribute to this project.

## License

This project is licensed under the terms of the MIT License.

---

This `README.md` provides a comprehensive overview of the project, including its structure, setup instructions, usage guidelines, and acknowledgements. Feel free to modify any sections to better fit your project's specifics.