import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)


def gpt_response(prompt, model="gpt-4o", temp=0, max_tokens=100):
    instruction = "You are a construction contract review assistant."
    
    # Use the new API call method
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt}
        ],
        temperature=temp,
        max_tokens=max_tokens
    )
    
    return completion.choices[0].message.content

def deepinfra_response(prompt, model, temperature=0):
    try:
        api_key = os.getenv("DEEPINFRA_API_KEY")
        base_url = "https://api.deepinfra.com/v1/openai"
        openai_client = OpenAI(api_key=api_key, base_url=base_url)
        
        chat_completion = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}"

def get_llm_function(choice):
    if choice == "gpt4o":
        return lambda prompt: gpt_response(prompt, model="gpt-4o", temp=0, max_tokens=100)
    elif choice == "qwen":
        return lambda prompt: deepinfra_response(prompt, model="Qwen/Qwen2.5-72B-Instruct", temperature=0)
    elif choice == "llama70b":
        return lambda prompt: deepinfra_response(prompt, model="meta-llama/Meta-Llama-3.1-70B-Instruct", temperature=0)
    elif choice == "llama405b":
        return lambda prompt: deepinfra_response(prompt, model="meta-llama/Meta-Llama-3.1-405B-Instruct", temperature=0)
    elif choice == "wizard":
        return lambda prompt: deepinfra_response(prompt, model="microsoft/WizardLM-2-8x22B", temperature=0)
    else:
        raise ValueError(f"Unknown LLM choice: {choice}")

class Validator:
    def __init__(self, llm_choice, threshold=0.6, num_samples=3, max_attempts=2):
        self.llm_response_func = get_llm_function(llm_choice)
        self.threshold = threshold
        self.num_samples = num_samples
        self.max_attempts = max_attempts

    def get_model_probability(self, prompt):
        yes_count = 0
        no_count = 0

        for _ in range(self.num_samples):
            response = self.llm_response_func(prompt)
            if "yes" in response.strip().lower():
                yes_count += 1
            elif "no" in response.strip().lower():
                no_count += 1

        total = yes_count + no_count
        if total == 0:
            # If no valid response is given, return equal probability
            return 0.5, 0.5

        yes_ratio = yes_count / total
        no_ratio = no_count / total

        return yes_ratio, no_ratio

    def load_data(self, path):
        return pd.read_csv(path, encoding='ISO-8859-1')

    def save_output(self, output_df, path='ISO-8859-1'):
        output_df.to_csv(path, index=False)

    def get_all_clauses(self, project_path, head_entity, tail_entity):
        df = pd.read_csv(project_path, encoding='utf-8')
        
        matched_row = df[(df['head_entity'] == head_entity) & (df['tail_entity'] == tail_entity)]
        
        # return retrieved clause for entity pair
        if not matched_row.empty:
            return matched_row['retrieved_chunk'].iloc[0]
        
        return None

    def first_validation(self, clause, head_entity, tail_entity, output, extractor_func):
        prompt1 = f"""These are contract events in a contract payment process: Employer receives Milestone Payment Application (E1), Employer notifies Contractor about the satisfaction of the Milestone (E2), Employer issues Milestone Completion Certificate (E3), Employer makes Milestone Payment(E4). In the contract clause: {clause}, the temporal relation between the entity {head_entity} and {tail_entity} in a triple format is {output}. Verify if the relation is correct according to the clause.
        Only answer "Yes" for correct or "No" for incorrect."""
        
        for attempt in range(self.max_attempts):
            yes_prob, no_prob = self.get_model_probability(prompt1)
            V1 = yes_prob / (yes_prob + no_prob)

            if V1 >= self.threshold:
                return V1, output, "pass validation"
            
            new_output = extractor_func(clause, head_entity, tail_entity, self.llm_response_func)  # Re-extract relation
            if new_output == output:
                return V1, output, "pass validation"

        return V1, output, "failed validation"

    def second_validation(self, clause, output):
        joined_result = ", ".join(output)
        prompt2 = f"""The interim payment contains the four event: Employer receives Milestone Payment Application (E1), Employer notifies Contractor about the satisfaction of the Milestone (E2), Employer issues Milestone Completion Certificate (E3), Employer makes Milestone Payment(E4). In the contract context {clause}, the four extracted temporal relations between two of the events are written in the triple format: {joined_result}. Check if these relations indicate the following conflict. 
        a. The event entity in E = (Employer receives Milestone Payment Application (E1), Employer notifies Contractor about the satisfaction of the Milestone (E2), Employer issues Milestone Completion Certificate (E3), Employer makes Milestone Payment(E4)) happens following the time order in the list. If the extraction result imply a different order, there is a conflict.
        b. There is more than one same temporal relations between two pair of events. If there is two triple have same days extracted, one of them is likely wrong, change the wrong one to "Not mentioned".
        If the current four answer indicate an conflict according to a or b, return a new output in the following format without other words:(E1, exact days/Not mentioned, E2), (E1, exact days/Not mentioned, E3), (E1, exact days/Not mentioned, E4), (E3, exact days/Not mentioned, E4).\\
        If there is no above conflict in the extracted triples, return "pass validation" and no more other words.
         """
        response = self.llm_response_func(prompt2)
        return response

    def validate_project(self, project_path, output_path, entity_pair_list, extractor_func):
        output_df = self.load_data(output_path)

        if 'output' not in output_df.columns:
            raise ValueError("The output DataFrame does not contain the 'output' column.")

        output_df['first_validation'] = ""
        output_df['update_answer'] = ""
        output_df['second_validation'] = ""

        all_outputs = []
        for head_entity, tail_entity in entity_pair_list:
            clauses = self.get_all_clauses(project_path, head_entity, tail_entity)
            matched_rows = output_df[(output_df['head_entity'] == head_entity) & (output_df['tail_entity'] == tail_entity)]
            
            for idx, row in matched_rows.iterrows():
                output = row['output']
                V1, updated_output, validation_status = self.first_validation(clauses, head_entity, tail_entity, output, extractor_func)

                output_df.at[idx, 'output'] = updated_output
                output_df.at[idx, 'first_validation'] = validation_status
                output_df.at[idx, 'update_answer'] = updated_output if validation_status == "pass validation" else output
                all_outputs.append(updated_output)

        # Perform second validation once for all outputs
        if all_outputs:
            second_validation_answer = self.second_validation(clauses, all_outputs)
            if second_validation_answer == "pass validation":
                output_df['second_validation'] = "pass validation"
            else:
                output_df['second_validation'] = ""
                output_df.at[output_df.index[-1], 'second_validation'] = second_validation_answer

        self.save_output(output_df, output_path)

        return output_df


# sadb_path = 'external_database/small_sampleDB.csv'
# term_dic_path = 'external_database/term_dictionary.csv'
# extractor = ContractExtractor(sadb_path, term_dic_path)
# validator = Validator(llm_choice='llama405b', threshold=0.8, num_samples=3, max_attempts=2)
# entity_pair_list = [
#     ('Employer receives Milestone Payment Application', 'Employer notifies Contractor about the satisfaction of the Milestone'),
#     ('Employer receives Milestone Payment Application', 'Employer issues Milestone Completion Certificate'),
#     ('Employer receives Milestone Payment Application', 'Employer makes Milestone Payment'),
#     ('Employer issues Milestone Completion Certificate', 'Employer makes Milestone Payment')
# ]

# project_path = 'data/project_data.csv'
# output_path = 'gpt4o_rev.csv'
# # 运行验证过程
# validator.validate_project(project_path, output_path, entity_pair_list, extractor.rev_extractor)
