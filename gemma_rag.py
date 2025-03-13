import gradio as gr
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Dict, Any
import time

# Define the prompt template
template = """
You are an intelligent assistant designed to provide accurate and helpful answers based on the context provided. Follow these guidelines:
1. Use only the information from the context to answer the question.
2. If the context does not contain enough information to answer the question, say "I don't know" and do not make up an answer.
3. Be concise and specific in your response.
4. Always end your answer with "Thanks for asking!" to maintain a friendly tone.

Context: {context}

Question: {question}

Answer:
"""
custom_rag_prompt = PromptTemplate.from_template(template)

# Initialize models and vector store
model = ChatOllama(model="gemma3:4b", temperature=0.5)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma(embedding_function=embeddings)

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)

class State:
    def __init__(self, question: str):
        self.question = question
        self.context: List[Document] = []
        self.answer: str = ""

def retrieve(state: State):
    state.context = vector_store.similarity_search(state.question)
    return state

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state.context)
    messages = custom_rag_prompt.invoke({"question": state.question, "context": docs_content})
    response = model.invoke(messages)
    state.answer = response
    return state

def workflow(state_input: Dict[str, Any]) -> Dict[str, Any]:
    state = State(state_input["question"])
    state = retrieve(state)
    state = generate(state)
    return {"context": state.context, "answer": state.answer}

def process_pdfs(files):
    status_messages = []
    for file in files:
        loader = PyPDFLoader(file.name)
        pages = loader.load()
        all_splits = text_splitter.split_documents(pages)
        document_ids = vector_store.add_documents(documents=all_splits)
        status_messages.append(f"Processed {file.name} and added to database!")
    return "\n".join(status_messages)

def ask_question(question):
    result = workflow({"question": question})
    context = "\n\n".join(doc.page_content for doc in result["context"])
    answer = result["answer"].content
    return context, answer

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# PDF Query Interface")
    with gr.Tab("Upload PDFs"):
        pdf_input = gr.File(label="Upload PDFs", file_count="multiple")
        upload_status = gr.Textbox(label="Status")
        upload_button = gr.Button("Upload and Process")
    with gr.Tab("Ask Question"):
        question_input = gr.Textbox(label="Your Question")
        context_output = gr.Textbox(label="Context", lines=10)
        answer_output = gr.Textbox(label="Answer", lines=5)
        ask_button = gr.Button("Ask")

    upload_button.click(process_pdfs, inputs=pdf_input, outputs=upload_status)
    ask_button.click(ask_question, inputs=question_input, outputs=[context_output, answer_output])

demo.launch()