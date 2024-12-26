from langchain_ollama import ChatOllama, OllamaEmbeddings
import re
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from keywords import filter_data_by_label

model_name = "gemma2:latest"
llm = ChatOllama(model=model_name, temperature=0.7)
text_path = "C:/Emotional AI/Data/CompleteData/output.txt"

def clean_text(text):
	cleaned_text = text.strip()
	cleaned_text = re.sub(r"[^\w\s]", "", cleaned_text)
	cleaned_text = re.sub(r"\s+", " ", cleaned_text)
	return cleaned_text

def prepare_data(condition, data_path):
	filtered_data = filter_data_by_label(condition, data_path)
	if not filtered_data:
		return None

	data_string = ""
	for app in filtered_data:
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

	with open(text_path, 'w', encoding='utf-8') as file:
			file.write(data_string)

	return data_string

def process_query_with_condition(condition, data_path):
	data_string = prepare_data(condition, data_path)
	if not data_string:
		return "No apps found for the given condition."

	try:
		loader = TextLoader(file_path=text_path, encoding="utf-8")

		data = loader.load()

		text_splitter = RecursiveCharacterTextSplitter(separators=['.', '\n'], chunk_size=1000, chunk_overlap=200)
		split_data = text_splitter.split_documents(data)

		embeddings = OllamaEmbeddings(model=model_name)
		vectorstore = FAISS.from_documents(split_data, embedding=embeddings)

		memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
		conversation_chain = ConversationalRetrievalChain.from_llm(
			llm=llm,
			chain_type="stuff",
			retriever=vectorstore.as_retriever(),
			memory=memory,
		)

		query = (f"""
		  Analyze the dataset and recommend up to 5 different apps for managing {condition}.
			You should check the entire list of "labels" and see if the condition is there.
			The dataset contains multiple apps - ensure you look through all of them to select the 5 with positive sentiment and high rating.
			
			For each app, provide:
			App name as a header with a number
			The name of the app's developer
			The app's rating
			Brief sentiment summary
			Brief explanation of why it's helpful
			
			Format each app consistently like this:
			[number]. [App Name]
			Developer: [Developer]
			Rating: [Rating]
			Sentiment: [Summary]
			[Explaination/Summary]
			""")

		result = conversation_chain({"question": query})
		return result.get("answer", "No response generated.")
	except Exception as e:
		return f"Error processing query: {e}"

def chat_with_model():
	print("Welcome to the Emotional AI Recommendation Tool!")
	while True:
		condition = input("Enter the condition you'd like app recommendations for (e.g., depression) or 'exit' to quit: ").strip()

		if condition.lower() == "exit":
			print("Goodbye!")
			break

		if not condition:
			print("Condition is required. Try again.")
			continue

		data_path = "C:/Emotional AI/Data/CompleteData/StringData.json"
		response = process_query_with_condition(condition, data_path)
		print("\nRecommendations:\n")
		print(response)

if __name__ == "__main__":
	chat_with_model()
