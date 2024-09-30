from flask import Flask, request, jsonify, render_template
import dotenv
import os
import requests
import re
from openai import OpenAI
from langchain.memory import ConversationSummaryBufferMemory, ConversationBufferMemory
# from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
# from langchain_openai import ChatOpenAI
from langchain.chat_models import AzureChatOpenAI
import os.path
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
import base64
from io import BytesIO
from PIL import Image
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
from langchain_community.vectorstores import Chroma

dotenv.load_dotenv()
app = Flask(__name__)


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
    llm = AzureChatOpenAI(
            azure_deployment=deployment,  # or your deployment
            api_version="2024-05-01-preview",  # or your api version
            temperature=0,
            azure_endpoint=endpoint,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
    if prev_memory is not None:
        memory = prev_memory
    else:
        memory = ConversationSummaryBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer', llm=llm)
    


    template = """You are a dental trauma chatbot designed to assist health care workers, dentists, and first responders (including teachers) in handling dental trauma cases. Your role is to provide precise, step-by-step guidance, ensuring users receive accurate, tailored responses based on their professional role and the specific situation of the trauma. 
        Ascertain User Role: At the start of each conversation, ask the user to identify their role: health care worker, dentist, or first responder (e.g., teacher). Use this information to customize your responses to match their expertise level.

        Understand the Trauma Situation:

        Ask the following questions to gather key information about the dental trauma:
        Patient's Age: This helps determine whether the affected teeth are permanent or deciduous (baby teeth).
        Type of Tooth: Ask if the affected tooth is permanent or deciduous.
        Tetanus Prophylaxis: Inquire whether the patient has received a Tetanus shot, especially if the trauma involves open wounds.
        Trauma Intensity: Ask about the severity of the injury, such as if the patient has lost consciousness or has other serious injuries that may require immediate medical attention.
        Intuitive Guidance:

        Based on the user's answers, provide step-by-step instructions on how to handle the situation.
        Reference the provided context and pull relevant details from the vector database to ensure accuracy.
        Keep conversations concise but informative, providing additional details if asked or as the situation escalates.
        Prioritize user safety by recommending immediate medical attention when necessary.
        Adapt to User's Needs:

        Be empathetic and patient. If the user seems uncertain, offer clarification and additional questions to guide them through the process.
        Keep responses accessible, especially when dealing with first responders like teachers who may not have medical training.
        Your goal is to ensure that each interaction is smooth, intuitive, and context-driven, providing the best possible support for handling dental trauma cases. Do not break character and do not answer irrelvant questions.
        Do not try to summarize or change the user's question. and if user want to provide image then accept it.
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
        verbose=True,
        rephrase_question=False,
        combine_docs_chain_kwargs={'prompt': prompt}
    )
    
    return qa_chain


def get_file_data(memory=None):    
    # dumpData()
    persist_directory = 'openaidb'

    ## here we are using OpenAI embeddings but in future we will swap out to local embeddings
    embedding = OpenAIEmbeddings()
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


def make_payload(payload,user_query):
    qa_chain = ConversationManager.get_instance().get_conversation()["chain"]
    convo_dict = qa_chain.__dict__
    y = convo_dict['memory'].__dict__
    x = y['chat_memory'].__dict__
    data_dict_convo = {
        "memory": transform_messages(x['messages']),
    }
    payload['messages'] = data_dict_convo['memory']
    payload['messages'].append(user_query)
    # print(payload)
    # print(payload)
    return payload['messages']
    # print(qa_chain.__dict__)

def set_model(vectordb,prev_memory=None):
    retriever = vectordb.as_retriever(search_kwargs={"k": 10})
    # llm = gpt
    llm = AzureChatOpenAI(
            azure_deployment=deployment,  # or your deployment
            api_version="2024-05-01-preview",  # or your api version
            temperature=0,
            azure_endpoint=endpoint,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
    if prev_memory is not None:
        memory = prev_memory
    else:
        memory = ConversationSummaryBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer', llm=llm)
    


    template = """You are a dental trauma chatbot designed to assist health care workers, dentists, and first responders (including teachers) in handling dental trauma cases. Your role is to provide precise, step-by-step guidance, ensuring users receive accurate, tailored responses based on their professional role and the specific situation of the trauma. 
        Ascertain User Role: At the start of each conversation, ask the user to identify their role: health care worker, dentist, or first responder (e.g., teacher). Use this information to customize your responses to match their expertise level.

        Understand the Trauma Situation:

        Ask the following questions to gather key information about the dental trauma:
        Patient's Age: This helps determine whether the affected teeth are permanent or deciduous (baby teeth).
        Type of Tooth: Ask if the affected tooth is permanent or deciduous.
        Tetanus Prophylaxis: Inquire whether the patient has received a Tetanus shot, especially if the trauma involves open wounds.
        Trauma Intensity: Ask about the severity of the injury, such as if the patient has lost consciousness or has other serious injuries that may require immediate medical attention.
        Intuitive Guidance:

        Based on the user's answers, provide step-by-step instructions on how to handle the situation.
        Reference the provided context and pull relevant details from the vector database to ensure accuracy.
        Keep conversations concise but informative, providing additional details if asked or as the situation escalates.
        Prioritize user safety by recommending immediate medical attention when necessary.
        Adapt to User's Needs:

        Be empathetic and patient. If the user seems uncertain, offer clarification and additional questions to guide them through the process.
        Keep responses accessible, especially when dealing with first responders like teachers who may not have medical training.
        Your goal is to ensure that each interaction is smooth, intuitive, and context-driven, providing the best possible support for handling dental trauma cases. Do not break character and do not answer irrelvant questions.
        Do not try to summarize or change the user's question. and if user want to provide image then accept it.
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
        verbose=True,
        rephrase_question=False,
        combine_docs_chain_kwargs={'prompt': prompt}
    )
    
    return qa_chain

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
    prompt = data['question']
    png_image = change_image_format(image)
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
    # Payload for the request
    payload = {
        "messages": [],
        }
    resp_payload = make_payload(payload,user_query)
    llm = AzureChatOpenAI(
            azure_deployment=deployment,  # or your deployment
            api_version="2024-05-01-preview",  # or your api version
            temperature=0,
            azure_endpoint=endpoint,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
    prompt = ChatPromptTemplate.from_messages(resp_payload)
    chain = prompt | llm
    response = chain.invoke({"image_red": image_red})
    
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
        
    # Prepare the input for the chain
    response = qa_chain({"question":question})
    # print(response['source_documents'][0].__dict__['metadata']['source'])

    return jsonify({
        'source': response['source_documents'][0].__dict__['metadata']['source'],
        'response': response['answer'],
        "sender": {
            "name": "Monika Figi",
            "avatar": "https://th.bing.com/th/id/R.78399594cd4ce07c0246b0413c95f7bf?rik=Nwo0AAuaJO%2fPEQ&pid=ImgRaw&r=0"
        }
    })
if __name__ == '__main__':
    app.run(debug=True)