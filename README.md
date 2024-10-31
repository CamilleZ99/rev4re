# rev4re


## Setup

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
2. **Configure environment variables** by creating a `.env` file with your API keys.

## Usage
Run the main script to start contract analysis:

**Note:** The contract data cannot be shared publicly, but we provide a sanitized sample that allows you to start from the clause extraction step. The cleaned data is available at `retrievedFrom_sampleContract`.
```sh
python main.py
```

## Customization
To **customize the language model** used for contract analysis, modify the `llm_choice` parameter in the script. Available options include:
- `gpt4o` for GPT-4.
- `llama70b` for Llama 70B.
- `wizard` for WizardLM.

To **customize the extraction and validation methods** used for contract analysis, modify the `extractor_choice` parameter in the script. Available options include:
- `standard` for standard extraction.
- `rev` for reverse extraction.
- `cot` for chain-of-thought extraction.
