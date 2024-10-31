# rev4re


## Setup

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
2. **Configure environment variables** by creating a `.env` file with your API keys.

## Usage
Run the main script to start temporal relation extraction in the demo contract:
```sh
python run.py
```
**Note:** The contract data cannot be shared publicly, but we provide a sanitized sample that allows you to start from the clause extraction step. The cleaned data is available at `retrievedFrom_sampleContract`.


## Customization
To **customize the language model** used for contract analysis, modify the `llm_choice` parameter in the script. Available options include:
- `gpt4o` for openai GPT-4o.
- `llama70b` for Meta-Llama-3.1-70B-Instruct.
- `wizard` for WizardLM-2-8x22B.
- `qwen` for Qwen2.5-72B-Instruct

To **customize the extraction and validation methods** used for contract analysis, modify the `extractor_choice` parameter in the script. Available options include:
- `standard` for standard extraction.
- `cot` for chain-of-thought extraction.
- `rev` for retrieve-enhance-verify extraction.
