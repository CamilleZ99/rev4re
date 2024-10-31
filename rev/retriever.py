import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import PyPDF2
import re

def get_clauses(entity_pair, df):
    """
    Retrieve clauses corresponding to the given entity pair.
    """
    clauses = df[df['entity_pair'] == entity_pair]['sample_clause'].tolist()
    return clauses

def extract_text_from_pdf(pdf_path):
   
    text = ''
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'
                else:
                    print(f"Warning: No text found on page {page_num + 1}.")
    except Exception as e:
        print(f"Error during PDF text extraction: {e}")
    
    return text

def create_chunks(text, max_chunk_size=2000):

    chunks = []
    current_chunk = ''
    sentences = re.split(r'(?<=[。.!?])\s+', text.replace('\n', ' '))
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            current_chunk += sentence + ' '
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ' '
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def main():
    pdf_path = 'contract_data/'
    entity_pairs = ['e1,e2', 'e1,e3', 'e1,e4', 'e3,e4']
    csv_path = 'external_database/small_sampleDB.csv'

    df = pd.read_csv(csv_path, encoding='ISO-8859-1')

    # Extract text from PDF
    print("Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print("No text extracted from the PDF.")
        return

    chunks = create_chunks(text, max_chunk_size=2000)
    if not chunks:
        print("No chunks created from the text.")
        return

    model_name = 'sentence-transformers/all-mpnet-base-v2'
    model = SentenceTransformer(model_name)

    # Compute embeddings for chunks in batches
    chunk_embeddings = []
    batch_size = 64
    for i in range(0, len(chunks), batch_size):
        chunk_batch = chunks[i:i+batch_size]
        embeddings = model.encode(chunk_batch, normalize_embeddings=True)
        chunk_embeddings.append(embeddings)
    chunk_embeddings = np.vstack(chunk_embeddings).astype('float32')

    results = []

    for entity_pair in entity_pairs:
        clauses = get_clauses(entity_pair, df)
        if len(clauses) < 1:
            print(f"No clauses found for the entity pair {entity_pair}.")
            continue

        # Compute embeddings for clauses
        clause_embeddings = model.encode(clauses, normalize_embeddings=True).astype('float32')

        # Compute similarities between clauses and chunks
        S = np.dot(clause_embeddings, chunk_embeddings.T)

        # Find the maximum similarity and corresponding indices
        max_sim_idx = np.unravel_index(np.argmax(S), S.shape)
        max_clause_idx = max_sim_idx[0]
        max_chunk_idx = max_sim_idx[1]
        max_similarity = S[max_clause_idx, max_chunk_idx]
        max_similarity_normalized = (max_similarity + 1) / 2  # Convert from [-1,1] to [0,1]

        # Get the corresponding clause and chunk
        best_clause = clauses[max_clause_idx]
        best_chunk = chunks[max_chunk_idx]


        result = {
            'Entity Pair': entity_pair,
            'Clause Index': max_clause_idx + 1, 
            'Clause Text': best_clause,
            'Similarity': max_similarity_normalized,
            'Chunk Text': best_chunk
        }

        results.append(result)

    if results:
        # Save the results to CSV
        output_df = pd.DataFrame(results)
        output_csv_path = 'output.csv'
        output_df.to_csv(output_csv_path, index=False)
        print(f"Results saved to {output_csv_path}")

        # Print the results
        for res in results:
            print(f"Entity Pair: {res['Entity Pair']}")
            print(f"Clause Index: {res['Clause Index']}")
            print(f"Clause Text: {res['Clause Text']}")
            print(f"Similarity: {res['Similarity']}")
            print("Chunk Text:")
            print(res['Chunk Text'])
            print('-' * 50)
    else:
        print("No results to save.")

if __name__ == '__main__':
    main()
