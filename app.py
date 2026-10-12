'''
this file contains real time job
a rest api that accepts question from postman
note: question should be json format

step1: read question
step2: convert question to vector using embeddding model
step3: load faiss db -> tcs_doc_index
step4: ask question to faiss db get relevant chunk
step5: make prompt to with question and context
step6: pass prompt to any free llms(openai,gemini,huggingface,etc..)
step7: once we get answer from llm ,return answer to POSTMAN
'''
from flask import Flask
from flask import jsonify
from flask import request
from flask import render_template
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from langchain_community .vectorstores import FAISS
import os
#from openai import OpenAI
from dotenv import load_dotenv
from google import genai
# from groq import Groq

app = Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/tcs",methods=['POST'])
def tcs_chatbot_api():
    #step1: read question coming fro POSTMAN
    # data  = request.get_json()
    # question = data.get("tcs_question")
    # print(question)
    question = request.form.get("tcs_question", "").strip()
    print(question)

# step2: convert question to vector using embeddding model
    embeddings  = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")
    question_vector = embeddings.embed_query(question)

#step3: load faiss db -> tcs_doc_index
    new_vector_store = FAISS.load_local("tcs_doc_index", embeddings,allow_dangerous_deserialization=True)

#step4: ask question to faiss db get relevant chunk
    context = new_vector_store.similarity_search(question,k=10)
    


#step : convert context into string format
    context_text = ""
    for document in context:
        context_text = context_text + document.page_content + "\n"

#step5: make prompt to with question and context
    prompt = f'''
     i am going to ask a question, please answer based on the given context.
     if you don't get answer from the context then don't answer.

    Question: {question}
    Context: {context_text}
    answer:
    '''
    # os.environ["OPENAI_API_KEY"] = 
    # client = OpenAI()
    # response  = client.responses.create(
    #     model = "gpt-5.6-luna",
    #     input  = prompt
    # )
    
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    client = genai.Client(
    api_key=api_key
    )

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents = prompt
    )
    
    # return jsonify({
    #             "response": response.text

    return render_template(
    "index.html",
    question=question,
    answer=response.text,
    error=None
)
                
        #  })

app.run()


    # return jsonify( {"gpt response": str(response.output_text) } )





    # client = Groq(
    #     api_key=os.environ["GROQ_API_KEY"]
    # )

    # response = client.chat.completions.create(
    #     model="openai/gpt-oss-120b",
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": prompt
    #         }
    #     ]
    # )

    # answer = response.choices[0].message.content

    # return jsonify({
    #     "question": question,
    #     "answer": answer
    # })






