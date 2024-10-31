from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class ContractExtractor:
    def __init__(self, sadb_path, term_dic_path):
        """
        :param sadb_path
        :param model_name
        """
        self.kb_df, self.td_df = self.load_external_knowledge(sadb_path, term_dic_path)
  
    def load_external_knowledge(self,sadb_path, term_dic_path):
        kb_df = pd.read_csv(sadb_path, encoding='ISO-8859-1')
        kb_texts = kb_df['sample_clause'].tolist()
        td_df = pd.read_csv(term_dic_path, encoding='ISO-8859-1')
        return kb_df, td_df
    

    def get_all_clauses(self, project_path, head_entity, tail_entity):
        df = pd.read_csv(project_path, encoding='utf-8')
        
        matched_row = df[(df['head_entity'] == head_entity) & (df['tail_entity'] == tail_entity)]
        
        # return retrieved clause for entity pair
        if not matched_row.empty:
            return matched_row['retrieved_chunk'].iloc[0]
        
        return None


    def find_demonstration(self, head_entity, tail_entity):
        """
        :param head_entity
        :param tail_entity
        """
        # match head entity&tail entity
        matching_row = self.kb_df[
            (self.kb_df['head_entity'] == head_entity) & 
            (self.kb_df['tail_entity'] == tail_entity)
        ].iloc[0]  

        # extract small sample labeled data
        sample_clause = matching_row['sample_clause']
        entHead_mention = matching_row['entHead_mention']
        entTail_mention = matching_row['entTail_mention']
        relation = matching_row['relation']

        return sample_clause, entHead_mention, entTail_mention, relation
    

    def find_dictionary(self, entity):

        # match term definition and disambiguation
        matching_rows = self.td_df[self.td_df['term'] == entity]

        if not matching_rows.empty:
            definition = matching_rows.iloc[0]['definition']
            disambiguation = matching_rows.iloc[0]['disambiguation']
            return definition, disambiguation
        else:
            return "No definition found", "No disambiguation found"
        
    def standard_extractor(self, clause, llm_response_func):

        standard_prompt = f"""These are contract events in a contract payment process placed in timely order: Employer receives Milestone Payment Application (E1), Employer notifies Contractor about the satisfaction of the Milestone (E2), Employer issues Milestone Completion Certificate (E3), Employer makes Milestone Payment(E4).\
            Your task is to extract certain relation between given entity pairs: <E1,E2>, <E1,E3>, <E1,E4>, <E3,E4> from the given contract clause.
        Contract clauses: {clause}\
        Answer directly in the following format AND no more other words: 
        (E1, exact days/Not mentioned, E2), (E1, exact days/Not mentioned, E3), (E1, exact days/Not mentioned, E4), (E3, exact days/Not mentioned, E4).
        You should answer in this format and no more other words. Do not give me your resoning process. Leave the event number as it is, do not replace with event name.
        """
        
        standard_response = llm_response_func(standard_prompt)
        return standard_response

    def cot_extractor(self, clause, head_entity, tail_entity, llm_response_func):
        
        cot_prompt = f"""These are contract events in a contract payment process placed in timely order: Employer receives Milestone Payment Application (E1), Employer notifies Contractor about the satisfaction of the Milestone (E2), Employer issues Milestone Completion Certificate (E3), Employer makes Milestone Payment(E4).\
            Your task is to extract certain relation between given entity pairs {head_entity} and {tail_entity} if it is mentioned in the given contract clause. You will be provided with contract text in a construction contract, the input entity pair and the instructions.
            <Contract clauses>: 
                        ```{clause}```\\
        For extracting relation between Ea and Eb, please follow the instructions:
            1.	check if the contract clause contains the event Ea and Eb;
            2.	If there is, check if the clause regulates their temporal relation;
            3.	If there is temporal relation regulated, return the relation between the entity pair in the following format AND no more other words: 
                <Ea, exact days, Eb>;
            4.	If there is no correct entity pair or correct temporal relation exist, you should genuinely answer:
            <Ea, not mentioned, Eb>;\
            Answer directly in the following format AND no more other words: 
            <Ea, the specific day/Not mentioned, Eb>
            (replace the Ea, Eb with actual event number, the specific daya should be 'within * days', * is a number)
            5.	A demonstration is given for you:
                a)	In the clause /The Employer will within 28 Days after receiving the Statements and all supporting documents including the Progress Report in accordance with Clause 4.23 [Progress Reports], give to the Contractor Notice of any items in the Statement with which the Employer disagrees, with supporting particulars./, the mention of /Employer receives Milestone Payment Application/ is /receiving the Statements and all supporting documents/, the mention of /Employer notifies Contractor about the satisfaction of the Milestone/ is /give to the Contractor Notice of any items in the Statement with which the Employer disagrees/. The relation between them is within 28 Days.
                b)	Therefore, you should answer (E1, within 28 Days, E2)\\
        """
        cot_response = llm_response_func(cot_prompt)
        return cot_response

    def rev_extractor(self, clause, head_entity, tail_entity, llm_response_func):
        """
        extract 
        :param clause
        :param head_entity
        :param tail_entity
        :param llm_response_func
        """
        # match the demonstration in SADB for entity pairs
        sample_clause, entHead_mention, entTail_mention, relation = self.find_demonstration(head_entity, tail_entity)

        # match the term definition and disambiguation knowledge in the dictionary
        head_entity_definition, head_entity_disambiguation = self.find_dictionary(head_entity)
        tail_entity_definition, tail_entity_disambiguation = self.find_dictionary(tail_entity)

        REV_prompt = f"""" These are contract events in a contract payment process placed in timely order: Employer receives Milestone Payment Application (E1), Employer notifies Contractor about the satisfaction of the Milestone (E2), Employer issues Milestone Completion Certificate (E3), Employer makes Milestone Payment(E4).\
            Your task is to extract certain relation between given entity pairs {head_entity} and {tail_entity} if it is mentioned in the given contract clause. You will be provided with contract text in a construction contract, the input entity pair, the knowledge about the entities in contract, and a demonstration. 
            <Contract clauses>: 
                        ```{clause}```\\
            There is some background knowledge provided about the entities. 
            For {head_entity},{head_entity_definition}.To recognize this entity from contract context you should note that: {head_entity_disambiguation}.
            For {tail_entity},{tail_entity_definition}.To recognize this entity from contract context you should note that: {tail_entity_disambiguation}.
            Please follow the instructions:
            1.	check if the contract clause contains the event{head_entity} and {tail_entity};
            2.	If there is, check if the clause regulates their temporal relation;
            3.	If there is temporal relation regulated, return the relation between the entity pair in the following format AND no more other words: 
                within * days/ Not mentioned  (replace the Ea, Eb with actual event number, where Ea happens before Eb);
            4.	If there is no correct entity pair or correct temporal relation exist, you should genuinely answer:
            <Ea, not mentioned, Eb>  (replace the Ea, Eb with actual event number, where Ea happens before Eb);
            5.	A demonstration is given for you:
                a)	In the clause {sample_clause}, the mention of {head_entity} is {entHead_mention}, the mention of {tail_entity} is {entTail_mention}. The relation between them is {relation}.
                b)	Therefore, you should answer ({head_entity}, {relation}, {tail_entity })\\
            You should answer in <Ea, exact days/not mentioned, Eb> format and no more other words. Do not give me your resoning process. Replace the Ea, Eb with actual event number, namely E1, E2, E3, E4.
            """

        # call LLM
        response = llm_response_func(REV_prompt)
        return response

