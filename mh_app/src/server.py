from flask import Flask, request, jsonify
from flask_cors import CORS
import os, re
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from dotenv import load_dotenv
from keywords import filter_data_by_label

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

app = Flask(__name__)
CORS(app)

# clean text of unnecessary tokens
def clean_text(text):
    cleaned_text = text.strip()
    
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
    
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    return cleaned_text

def process_query(query):
    data_path = "C:/Emotional AI/Data/CompleteData/StringData.json"

    filtered_data = filter_data_by_label(query, data_path)

    data_string = ""

    for app in filtered_data:
        # remove unnecessary tokens and convert data to a string before sending to model
        name = clean_text(app.get("name", ""))
        developer = clean_text(app.get("developer", ""))
        rating = app.get("rating", "")
        labels = clean_text(" ".join(app.get("labels", [])))
        reviews = clean_text(app.get("reviews", ""))
        sentiment = clean_text(str(app.get("sentiment", "")))

        data_string += f"name: {name}\n"
        data_string += f"developer: {developer}\n"
        data_string += f"rating: {rating}\n"
        data_string += f"labels: {labels}\n"
        data_string += f"reviews: {reviews}\n"
        data_string += f"sentiment: {sentiment}\n\n"

    text_path = "C:/Emotional AI/Data/CompleteData/output.txt"

    with open(text_path, 'w', encoding='utf-8') as file:
        file.write(data_string)

    # chat with GPT via LangChain
    if data_string:
        loader = TextLoader(file_path=text_path, encoding='utf-8')
        data = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        data = text_splitter.split_documents(data)

        embeddings = OpenAIEmbeddings()

        vectorstore = FAISS.from_documents(data, embedding=embeddings)

        llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")

        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            memory=memory
        )

        result = conversation_chain({"question": query})
        # use a fallback llm if a response to the query can't be generated
        if "I don't know" in result["answer"]:
            fallback_llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini")
            fallback_response = fallback_llm.invoke(query)
            print("gpt-4o-mini")
            return fallback_response.content
        else:
            print("gpt-3.5-turbo")
            return result["answer"]
    # use this llm if the query is unrelated (has no condition); reduces the input tokens with the API request
    else:
        fallback_llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini")
        fallback_response = fallback_llm.invoke(query)
        print("gpt-4o-mini")
        return fallback_response.content

@app.route('/ask', methods=['POST'])
def ask_question():
    # pull the user input from the frontend
    query = request.json.get('message')
    if not query:
        return jsonify({'error': 'No message provided'}), 400

    answer = process_query(query)
    # return llm response to be displayed on the frontend
    return jsonify({'choices': [{'message': {'role': 'assistant', 'content': answer}}]})

# run app
if __name__ == '__main__':
    app.run(debug=True)
