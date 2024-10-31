import pandas as pd
from openai import OpenAI
from huggingface_hub import login
from dotenv import load_dotenv
import os
from extractor import ContractExtractor
from validator import Validator
import csv
from concurrent.futures import ThreadPoolExecutor

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Define the event list
entity_pair_list = [
    ('Employer receives Milestone Payment Application', 'Employer notifies Contractor about the satisfaction of the Milestone'),
    ('Employer receives Milestone Payment Application', 'Employer issues Milestone Completion Certificate'),
    ('Employer receives Milestone Payment Application', 'Employer makes Milestone Payment'),
    ('Employer issues Milestone Completion Certificate', 'Employer makes Milestone Payment')
]

# for gpt series model, we call the openai api
def gpt_response(prompt, model="gpt-4o", temp=0, max_tokens=100):
    instruction = "You are a construction contract review assistant."

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt}
        ],
        temperature=temp,  # Set the temperature=0 to ensure generation stalability
        max_tokens=max_tokens 
    )

    return completion.choices[0].message.content

# for open-source model, we call api from the deepinfra platform
def deepinfra_response(prompt, model, temp=0):
    try:

        api_key = os.getenv("DEEPINFRA_API_KEY")
        base_url = "https://api.deepinfra.com/v1/openai"
        openai_client = OpenAI(api_key=api_key, base_url=base_url)
        chat_completion = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp  
        )
        
        response = chat_completion.choices[0].message.content
        return response
    except Exception as e:
        return f"Error generating response: {e}"


# select LLM based on input
def get_llm_function(choice):
    if choice == "gpt4o":
        return lambda prompt: gpt_response(prompt, model="gpt-4o", temp=0, max_tokens=100)
    elif choice == "llama70b":
        return lambda prompt: deepinfra_response(prompt, model="meta-llama/Meta-Llama-3.1-70B-Instruct", temperature=0)
    elif choice == "wizard":
        return lambda prompt: deepinfra_response(prompt, model="microsoft/WizardLM-2-8x22B", temperature=0)
    elif choice == "qwen":
        return lambda prompt: deepinfra_response(prompt, model="Qwen/Qwen2.5-72B-Instruct", temperature=0)
    else:
        raise ValueError(f"Unknown LLM choice: {choice}")

def process_file(extractor, project_path, llm_function, extractor_choice):
    
    df = pd.read_csv(project_path, encoding='utf-8')
    matched_row = df[df['entity_pair'] == 'e']
    joined_clause = matched_row['retrieved_chunk'].iloc[0]

    if extractor_choice == "standard":
        output = extractor.standard_extractor(joined_clause, llm_function)
        return [(output,)]  # Ensure the output is wrapped in a tuple to avoid token splitting
    else:
        results = []
        for head_entity, tail_entity in entity_pair_list:
            clause = extractor.get_all_clauses(project_path, head_entity, tail_entity)
            if extractor_choice == "rev":
                output = extractor.rev_extractor(clause, head_entity, tail_entity, llm_function)
            elif extractor_choice == "cot":
                output = extractor.cot_extractor(clause, head_entity, tail_entity, llm_function)
            else:
                raise ValueError(f"Unknown extractor choice: {extractor_choice}")
            results.append((head_entity, tail_entity, output))
        return results

# extract results and validate results
def main(project_path, llm_choice="gpt4o", extractor_choice="rev"):
    extractor = ContractExtractor(sadb_path='external_database/small_sampleDB.csv', term_dic_path='external_database/term_dictionary.csv')

    # output file name
    output_path = f'data/{llm_choice}_{extractor_choice}.csv'
    
    llm_function = get_llm_function(llm_choice)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_file, extractor, project_path, llm_function, extractor_choice)]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if extractor_choice == "standard":
            writer.writerow(["output"])
            for future in futures:
                try:
                    results = future.result()
                    for result in results:
                        writer.writerow(result)  # Write the entire output as a single cell
                except Exception as e:
                    print(f"Error processing file: {e}")
        else:
            writer.writerow(["head_entity", "tail_entity", "output"])
            for future in futures:
                try:
                    results = future.result()
                    writer.writerows(results)
                except Exception as e:
                    print(f"Error processing file: {e}")
    if extractor_choice == "rev":
        validator = Validator(llm_choice='gpt4o', threshold=0.8, num_samples=3, max_attempts=2)
        validator.validate_project('data/retrievedFrom_sampleContract.csv', output_path, entity_pair_list, extractor.rev_extractor)

if __name__ == '__main__':
    # You can choose the models and baselines
    project_path = "data/retrievedFrom_sampleContract.csv"
    llm_choice = "gpt4o"        # Options: "gpt4o", "llama70b", "mistral7b8","wizard"
    extractor_choice = "rev"  # Options: "standard", "rev", "cot"
    main(project_path, llm_choice=llm_choice, extractor_choice=extractor_choice)
