# Benchmarking FuriosaAI RNGD Across Diverse LLM Tasks

This project benchmarks the performance of FuriosaAI RNGD for a range of LLM tasks, including:
	•	RAG (Retrieval-Augmented Generation) Chatbot: Use databases and QA datasets to test retrieval accuracy and generative capabilities.
	•	Translation: Evaluate multi-language translation accuracy and performance.
	•	Summarization: Test the summarization of large-scale documents.
	•	Multi-Modal Agent: Combine vision models and LLMs for multi-modal understanding and reasoning.

The project enables easy comparisons between GPU and NPU environments using unified APIs and configurable pipelines.

## Project Structure

The project is organized as follows:

```
├── LICENSE              # Open-source license information
├── README.md            # Project overview and instructions
├── configuration/       # Collection of available configurations for tasks
├── data/
│   ├── db/              # (For RAG use) Database for Warboy and RNGD SDK
│   ├── translate/       # Translation benchmarking dataset
│   ├── chatbot/         # (For RAG use) QA dataset for Warboy and RNGD SDK
│   ├── summary/         # Dataset for document summarization
│   ├── multimodal/      # Example datasets for vision + LLM pairing
│   └── handmade-faq/    # (For RAG use) QA data from Furiosa customer portal
├── docs/                # Documentation for the project
├── notebooks/           # Jupyter notebooks for exploration and analysis
├── references/          # Data dictionaries, manuals, and reference materials
├── requirements.txt     # List of required Python packages
└── src/                 # Source code for various tasks
    ├── common/          # Environment detection, configuration loader
    ├── metrics/         # Metric definitions for LLM tasks
    ├── translation/     # GPU/NPU modules for LLM translation
    ├── summarization/   # Summarization task modules
    └── multimodal/      # Vision + LLM service examples (TBU)
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