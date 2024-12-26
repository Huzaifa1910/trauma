from flask import Flask, request, jsonify, render_template
import dotenv
from flask_cors import CORS
import os
import requests
import re
# from openai import OpenAI
from langchain.memory import ConversationSummaryBufferMemory
# from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
# from langchain_openai import ChatOpenAI
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
import os.path
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.embeddings import AzureOpenAIEmbeddings, OpenAIEmbeddings
import base64
from io import BytesIO
from PIL import Image
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage
import sys

sys.modules['sqlite3'] = __import__('pysqlite3')
from langchain_community.vectorstores import Chroma

dotenv.load_dotenv()
app = Flask(__name__)
CORS(app)
openai_api_key = os.getenv("OPENAI_API")
os.environ["OPENAI_API_KEY"] = openai_api_key

class ConversationManager:
    _instance = None

    @staticmethod
    def get_instance():
        if ConversationManager._instance is None:
            ConversationManager._instance = ConversationManager()
        return ConversationManager._instance

    def __init__(self):
        if ConversationManager._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            self.chain = None
            self.memory = None

    def set_conversation(self, chain, memory=None):
        self.chain = chain
        if memory is not None:
            self.memory = memory


    def get_conversation(self):
        if self.chain is None:
            raise Exception("Conversation object has not been initialized.")
        return {
            "chain": self.chain,
            "memory": self.memory
        }


def set_model(vectordb,prev_memory=None):
    retriever = vectordb.as_retriever(search_kwargs={"k": 10})
    # llm = gpt
    llm = ChatOpenAI(
        api_key=openai_api_key,
        model="gpt-4o",
        temperature=0.7,
        max_tokens=500,
        timeout=None,
        max_retries=2,
        )
    # llm = AzureChatOpenAI(
    #         azure_deployment=deployment,  # or your deployment
    #         api_version="2024-05-01-preview",  # or your api version
    #         temperature=0.7,
    #         azure_endpoint=endpoint,
    #         max_tokens=None,
    #         timeout=None,
    #         max_retries=2,
    #     )
    if prev_memory is not None:
        memory = prev_memory
    else:
        memory = ConversationSummaryBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer', llm=llm)
    


    template = """You are a dental trauma chatbot designed to assist health care workers, dentists, and first responders (including teachers) in handling dental trauma cases. Your role is to provide precise, step-by-step guidance, ensuring users receive accurate and tailored responses based on their professional role and the specific situation of the trauma. Your responses must remain short and concise. At the start of each conversation, ask the user to identify their role—whether they are a health care worker, dentist, or first responder (e.g., teacher). This allows you to customize your responses to match their expertise level. If the user requests to print any type of information, apologize and politely ask for another question instead.

                If the user asks an urgent question, promptly provide a clear answer and follow up by asking them for another question to continue assisting. To understand the trauma situation better, gather key details by asking questions such as the patient's age to determine if the affected teeth are permanent or deciduous (baby teeth), the type of tooth involved (permanent or deciduous), whether the patient has received a Tetanus shot (especially important for open wounds), and the intensity of the trauma, including whether the patient has lost consciousness or sustained other serious injuries that may require immediate medical attention.

                Based on the user's answers, provide step-by-step instructions to address the situation effectively. Reference the provided context and pull relevant details from the vector database to ensure your guidance is accurate. Keep conversations concise yet informative, elaborating only when necessary or if the situation escalates. Always prioritize user safety by recommending immediate medical attention when required.

                As you interact, adapt to the user's needs with empathy and patience. If they seem uncertain, offer clarification or additional questions to guide them through the process. Ensure your responses are accessible, especially for first responders like teachers who may not have medical training. Think carefully and deeply before formulating a response. If asking a question could lead to a more accurate or insightful answer, do not hesitate to seek clarification or refine the information provided.

                Maintain your focus by not breaking character or answering irrelevant questions. Avoid summarizing or altering the user’s question, and if the user wants to provide an image, accept it and integrate it into your assistance process. Your ultimate goal is to ensure each interaction is smooth, intuitive, and context-driven, providing the best possible support for handling dental trauma cases.

                Context: {context}
                History: {chat_history}
                Question: {question}
        # """
    prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template=template
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False,
        rephrase_question=False,
        combine_docs_chain_kwargs={'prompt': prompt}
    )
    
    return qa_chain


def get_file_data(memory=None):    
    # dumpData()
    persist_directory = 'openaidbnew'

    ## here we are using OpenAI embeddings but in future we will swap out to local embeddings
    embedding = OpenAIEmbeddings()
    # embedding = AzureOpenAIEmbeddings(openai_api_base=endpoint, openai_api_version="2024-05-01-preview", chunk_size=1536, validate_base_url=True, deployment='gpt40EmbeddingSmall')
    
    vectordb = Chroma(persist_directory=persist_directory, 
                    embedding_function=embedding)
    qa_chain = set_model(vectordb,memory)
    return qa_chain
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("ENDPOINT_URL")
deployment = "gpt40"

qa_chain = get_file_data()
ConversationManager.get_instance().set_conversation(qa_chain,None)
def transform_messages(messages):
    transformed_messages = []
    for i in range(len(messages)):
        if type(messages[i]) == HumanMessage:
            transformed_message = (
                'user',
                messages[i].content
            )
            transformed_messages.append(transformed_message)
        else:
            transformed_message = (
                'assistant',
                messages[i].content
            )
            transformed_messages.append(transformed_message)
    return transformed_messages

def retrieve_history_from_json(message_list):
    convo_hist = []
    for i in range(len(message_list)):
        if message_list[i][0] == 'user':
            message = HumanMessage(message_list[i][1])
        else:
            message = AIMessage(message_list[i][1])
        convo_hist.append(message)
    chat_history = InMemoryChatMessageHistory(messages=convo_hist)
    return ConversationSummaryBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer', chat_memory=chat_history)


def make_payload(payload, user_query, user_query_text=None):
    qa_chain = ConversationManager.get_instance().get_conversation()["chain"]
    convo_dict = qa_chain.__dict__
    y = convo_dict['memory'].__dict__
    x = y['chat_memory'].__dict__
    data_dict_convo = {
        "memory": transform_messages(x['messages']),
    }
    payload['messages'] = data_dict_convo['memory']
    payload['messages'].append(user_query)
    payload['messages'].insert(0, (
            "system",
            [
                {
                "type": "text",
                "text": """You are a dental trauma chatbot designed to assist health care workers, dentists, and first responders (including teachers) in handling dental trauma cases. In the provided image, your task is to analyze and assess the visible dental trauma or related injuries. Use the context of the image to identify damages, such as displaced teeth, fractures, swelling, or other signs of trauma, and provide appropriate, step-by-step guidance to address the situation. Tailor your analysis and recommendations based on the user's professional role—whether they are a health care worker, dentist, or first responder—ensuring the instructions align with their level of expertise and ability to act. When analyzing the image, consider the patient’s age and whether the affected teeth are permanent or deciduous. If the image contains open wounds, inquire whether the patient has received a Tetanus shot. Assess the severity of the trauma, including potential loss of consciousness or systemic injuries that might require urgent medical attention. Always prioritize safety, recommending immediate medical care when necessary, and ensure your responses are empathetic and accessible, especially for non-medical users like teachers. Think carefully and deeply before formulating a response. If asking a question could lead to a more accurate or insightful answer, feel free to ask first to clarify or refine the information. Your response must remain accurate, concise, and actionable, relying on the provided image and any additional context shared by the user. Begin every response with, "In the provided image," and deliver clear, step-by-step insights to guide the user effectively."""
                }
            ]
    ))
    if user_query_text is not None:
        payload['messages'].append(user_query_text)
    # system_prompt = {
    #     "role": "system",
    #     "content": "You are a dental trauma chatbot designed to assist health care workers, dentists, and first responders (including teachers) in handling dental trauma cases. In the provided image, your task is to analyze and assess the visible dental trauma or related injuries. Use the context of the image to identify damages, such as displaced teeth, fractures, swelling, or other signs of trauma, and provide appropriate, step-by-step guidance to address the situation. Tailor your analysis and recommendations based on the user's professional role—whether they are a health care worker, dentist, or first responder—ensuring the instructions align with their level of expertise and ability to act. If the image contains injuries to a tooth or surrounding structures, consider the patient’s age and whether the affected teeth are permanent or deciduous. Inquire if the patient has received a Tetanus shot in cases of open wounds and assess the severity of the trauma, including any potential loss of consciousness or other systemic injuries requiring urgent care. Your response must prioritize safety, recommend immediate medical attention if necessary, and remain empathetic and accessible, especially for non-medical users like teachers. Always aim to provide accurate, concise, and actionable insights while relying on the image and any additional context provided by the user, and you can analyze the image do not say you can not see the image and always start response from in the provided image."
    # }
    # payload['messages'].insert(0, system_prompt)
    return payload['messages']

def process_llm_response(llm_response):
    return llm_response['result']
    # print('\n\nSources:')
    # for source in llm_response["source_documents"]:
    #     print(source.metadata['source'])

def query_db(query):
    query = query
    global qa_chain
    llm_response = qa_chain(query)
    response = process_llm_response(llm_response)
    return response

def reduce_base64_image_size(base64_image, output_format='PNG', quality=70, width_scale=0.5):
    # Step 1: Decode base64 string to image
    image_data = base64.b64decode(base64_image.split(",")[1])  # Split the header and decode the actual base64 data
    img = Image.open(BytesIO(image_data))

    # Step 2: Resize the image (reduce size by the scale factor)
    new_width = int(img.width * width_scale)
    new_height = int(img.height * width_scale)
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)

    # Step 3: Compress the image and save to BytesIO buffer
    buffer = BytesIO()
    resized_img.save(buffer, format=output_format, quality=quality)  # Adjust quality to control compression
    buffer.seek(0)

    # Step 4: Encode the resized and compressed image back to base64
    new_base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    new_base64_image = f"data:image/{output_format.lower()};base64,{new_base64_image}"

    return new_base64_image

# write the function to change all iamges in png
def change_image_format(image):
    image = re.sub(r'data:image/[^;]+;base64,', '', image)
    image = base64.b64decode(image)
    image = Image.open(BytesIO(image))
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    # convert the image to base64
    image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return image

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


# Configuration



@app.route('/get_image_description', methods=['POST'])
def get_image_description():
    headers = {
        "Content-Type": "application/json",
        "api-key": os.getenv("AZURE_OPENAI_API_KEY"),
    }
    # Encode the uploaded image in base64
    data = request.json
    image = data['image']
    prompt_question = data['question']
    png_image = change_image_format(image)
    print(png_image[:100])
    png_image = f"data:image/png;base64,{png_image}"
    image_red = reduce_base64_image_size(png_image, output_format='PNG', quality=100, width_scale=0.9)
    user_query = (
            "user",
            [
                {
                "type": "image_url",
                "image_url": {
                    "url": f"{image_red}"
                }
                }
            ]
    )
    user_queryText = (
            "user",
            [
                {
                "type": "text",
                "text": f"{prompt_question}"
                }
            ]
    )
    
    # Payload for the request
    payload = {
        "messages": [],
        }
    resp_payload = make_payload(payload,user_query, user_query_text=user_queryText)
    llm = ChatOpenAI(
        api_key=openai_api_key,
        model="gpt-4o",
        temperature=0.7,
        max_tokens=500,
        timeout=None,
        max_retries=2,
        )
    # llm = AzureChatOpenAI(
    #         azure_deployment=deployment,  # or your deployment
    #         api_version="2024-05-01-preview",  # or your api version
    #         temperature=0.7,
    #         azure_endpoint=endpoint,
    #         max_tokens=1000,
    #         timeout=None,
    #         max_retries=2,
    #     )
    prompt = ChatPromptTemplate.from_messages(resp_payload)
    chain = prompt | llm
    response = chain.invoke({"image_red": image_red})
    print(response)
    
    return jsonify({
        'response': response.content,
        "sender": {
            "name": "Monika Figi",
            "avatar": "https://th.bing.com/th/id/R.78399594cd4ce07c0246b0413c95f7bf?rik=Nwo0AAuaJO%2fPEQ&pid=ImgRaw&r=0"
        }
    })

@app.route('/get_response', methods=['POST', 'GET'])
def get_response():
    data = request.json
    question = data.get('question', None)
    image = data.get('image', None)
    
    # Process image if it exists
    output_base64 = None
    if image:
        output_base64 = reduce_base64_image_size(data['image'], output_format='PNG', quality=70, width_scale=0.5)
        if output_base64.startswith("data:image/png;base64"):
            print("Image processed.")
    print(ConversationManager.get_instance().get_conversation()['memory'])
    # Get the conversation chain instance
    memory_list = ConversationManager.get_instance().get_conversation()['memory']
    if memory_list is not None:
        memory_list = retrieve_history_from_json(memory_list)
        qa_chain = get_file_data(memory_list)
        ConversationManager.get_instance().set_conversation(qa_chain, None)
    else:
        qa_chain = ConversationManager.get_instance().get_conversation()['chain']
    try:
        response = qa_chain({"question":question})
        print(response)
        resp = {
        'source': response['source_documents'][0].__dict__['metadata']['source'],
        'response': response['answer'],
        "sender": {
            "name": "Monika Figi",
            "avatar": "https://th.bing.com/th/id/R.78399594cd4ce07c0246b0413c95f7bf?rik=Nwo0AAuaJO%2fPEQ&pid=ImgRaw&r=0"
        }
    }
    except Exception as e:
        print(e)
        response = {"answer": str(e), "source_documents": []}
        resp = {
            "response": response['answer'],
            "sender": {
                "name": "Monika Figi",
                "avatar": "https://th.bing.com/th/id/R.78399594cd4ce07c0246b0413c95f7bf?rik=Nwo0AAuaJO%2fPEQ&pid=ImgRaw&r=0"
            }
        }
    # Prepare the input for the chain
    # print(response['source_documents'][0].__dict__['metadata']['source'])

    return jsonify(resp)
if __name__ == '__main__':
    app.run(debug=True)
