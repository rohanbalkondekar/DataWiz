from transformers import pipeline
from InstructorEmbedding import INSTRUCTOR
import ast
import os
import faiss
import openai
import pickle
from sentence_transformers import SentenceTransformer

# -------  Enter Your OpenAI API Key
openai.api_key = "sk-"
os.environ["OPENAI_API_KEY"] = "sk-"

# ------- Or Run a model locally

# pip install -q transformers

## checkpoint = "{model_name}"
# checkpoint = "{MBZUAI/LaMini-Cerebras-111M}"

# model = pipeline('text-generation', model=checkpoint)

# instruction = 'Please let me know your thoughts on the given place and why you think it deserves to be visited: \n"Barcelona, Spain"'

# input_prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"

# generated_text = model(input_prompt, max_length=512, do_sample=True)[
#     0]['generated_text']

# print("Response", generated_text)



# ------- Retriving the Dataframe

directory = "./ingested_data/dataframe"
file_path = os.path.join(directory, "data.pkl")

# Retrieve the DataFrame from the pickle file
with open(file_path, "rb") as file:
    df = pickle.load(file)
# ----------------------------------------------

directory = "./ingested_data/list"
file_path = os.path.join(directory, "list.pkl")

# Retrieve the stored list
if os.path.exists(file_path):
    with open(file_path, "rb") as file:
        stored_list = pickle.load(file)
        # print(stored_list)
else:
    print("List file does not exist.")

data = stored_list

model = SentenceTransformer('all-MiniLM-L6-v2')
# model = INSTRUCTOR('hkunlp/instructor-xl')
index = faiss.read_index('./ingested_data/index/index')

##--------------------------
query =  input("Ask a Question: ")
##--------------------------

def search(query):
   query_vector = model.encode([query])
   k = 5
   top_k = index.search(query_vector, k)
   return [data[_id] for _id in top_k[1].tolist()[0]]

results = search(query)

for result in results:
   print('\n\n -----#-----', result)

# -- OpenAI

result_strings = [str(result) for result in results]
result_string = "\n".join(result_strings)
prompt = f"Answer the following question based on the context, if you don't know the answer then just politely say that `No, I did not find answer in the given context`. The contest is:\n\n Question: {query}\n\n context:\n{result_string}\n\n Your Answer:"

# create a chat completion

chat_completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])

answer = chat_completion.choices[0].message.content
print(f"\n\n Answer: {answer} \n\n")

# -----------------------------------------------------------------------------------------------------------


print('\n\n ------- Getting Data Nearby to the Key Search Term -------\n\n')
# Search for the string 'Terms of Cover' throughout the DataFrame

content = f"Your task: From the given question identify keywords to search in the database. The format will be as follows. Input queston (in angle brackets): < A string >, Your Output as array of keywords: ['Keyword1', 'Keyword2', ... ]. Now this is the Actual Input question: < {query} > Now Your Answer: [Array of keywords]"

chat_completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", messages=[{"role": "user", "content": content}])

search_term = chat_completion.choices[0].message.content

print(f"search term -> {search_term}")

search_terms = ast.literal_eval(search_term)

##-----------------------

def search_keyword_in_file(file_path, keyword):
    # Read the contents of the file
    with open(file_path, "r", encoding="utf-8") as file:
        file_contents = file.read()

    # Find the keyword occurrences in the file
    keyword_positions = []
    start_index = 0
    while True:
        position = file_contents.lower().find(keyword.lower(), start_index)
        if position == -1:
            break
        keyword_positions.append(position)
        start_index = position + 1

    # Retrieve the surrounding text for each keyword occurrence
    search_results = []
    for position in keyword_positions:
        start_index = max(0, position - 70)
        end_index = min(len(file_contents), position + len(keyword) + 70)
        surrounding_text = file_contents[start_index:end_index]
        search_results.append(surrounding_text)

    return search_results


# Usage example
file_path = "./ingested_data/text/data.txt"

for search_term in search_terms:
    result = search_keyword_in_file(file_path, search_term)
    results.extend(result)

# results = search_keyword_in_file(file_path, search_term)

print("\n\n\n Search Results from the text file \n\n\n")

# for result in results:
#     print(result)

if len(results) > 15:
    first_few_results = results[:5]
    middle_few_results = results[(
        len(results) // 2 - 2):(len(results) // 2 + 3)]
    last_few_results = results[-5:]
    results_string = "\n".join(
        first_few_results + middle_few_results + last_few_results)
else:
    results_string = "\n".join(results)

print(results_string)

# Convert results to a single string

print("\n\n\n Search Resutls from the text file \n\n\n")

prompt = f"Answer the following question based on the context, if you don't know the answer then just politely say that `No, I did not find answer in the given context`. Question: {query}\n\n context:\n{results_string}\n\n Your Answer:"

# create a chat completion
chat_completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])


print("\n Answer: \n")
# print the chat completion
print(chat_completion.choices[0].message.content)
print("\n Answer End \n")


##-----------------------

print("\n\n\n Search Resutls from the dataframe \n\n\n")

try:
    # Find the row index containing the search term
    row_index = df[df.apply(lambda row: row.astype(str).str.contains(
        search_term, case=False).any(), axis=1)].index[0]

    # Define the range of rows to retrieve (1 row before and 1 row after the row containing the search term)
    start_index = max(row_index - 2, 0)
    end_index = min(row_index + 2, len(df) - 2)

    # Retrieve the subset of rows
    result_df = df.iloc[start_index:end_index + 2]
    print(result_df)
except IndexError:
    result_df = "Keyword not found in the DataFrame."
    print(result_df)


prompt = f"Answer the following question based on the context, if you don't know the answer then just politely say that `No, I did not find answer in the given context`. Question: {query}\n\n context:\n{result_df}\n\n Your Answer:"
# create a chat completion
chat_completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
print("\n Answer: \n")
# print the chat completion
print(chat_completion.choices[0].message.content)
print("\n Answer End \n")
