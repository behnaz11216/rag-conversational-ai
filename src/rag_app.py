import os
import sys
import json
import logging
from typing import List, Dict, Optional, Any

import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
# Hide the HTTP request logs of the Ollama client
logging.getLogger("httpx").setLevel(logging.WARNING)

# Constants
PDF_PATH = "data/source.pdf"
INDEX_PATH = "data/faiss_index"
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
NOT_DECIDED = "Not decided yet."

GREETING = (
    "Hi there! 👋 I'm here to help you plan the perfect website for your business. "
    "I'll guide you step by step and keep things simple. "
    "First, what is the name of your project or business? 😊"
)

# The topics the assistant asks about, in this order.
# The description is used in the prompts and as the search query for the PDF.
TOPICS = {
    "product_name": "the name of the project or business",
    "project_type": "the type of website, for example an online shop, a portfolio or a blog",
    "purpose": "the main goal of the website",
    "target_audience": "the people who will use the website",
    "design_style": "the visual style of the website",
    "core_features": "the main features the website needs",
    "content_needs": "the content that will be on the website",
    "technical_requirements": "technical needs such as languages, payments or hosting",
}

EXTRACT_PROMPT = """The user was asked about {topic}.
User's answer: "{answer}"

Return JSON with two keys:
"value": the part of the answer that is about {topic}, in the user's own words
"is_clear": true if the user gave an answer, false if the user is not sure, asks for help or did not answer
"""

QUESTION_PROMPT = """You are a friendly assistant who helps small business owners plan their website.
You are talking directly to the user, so use "you".

What we know so far:
{known}

The user's last answer: "{last_answer}"

Background knowledge (use it only for ideas, it is not what the user wants):
{context}

Now {task}
Write one or two short sentences and end with the question.
"""

# -------------------------------
# Custom Embeddings Class
# -------------------------------

class CustomEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(text, convert_to_tensor=False, show_progress_bar=False)
        return embedding.tolist()


# -------------------------------
# Assistant Class: asks the questions and keeps the answers
# -------------------------------

class Assistant:
    def __init__(self, extract_llm: OllamaLLM, chat_llm: OllamaLLM, retriever: Any):
        self.extract_llm = extract_llm
        self.chat_llm = chat_llm
        self.retriever = retriever
        self.project_details = {topic: "" for topic in TOPICS}
        self.project_details["special_requests"] = ""

    def get_context(self, topic: str) -> str:
        """
        Search the PDF for knowledge about the topic (this is the RAG part).
        """
        query = f"{TOPICS[topic]} {self.project_details['project_type']}"
        docs = self.retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in docs)

    def known_details(self) -> str:
        """
        The answers we have so far, one per line.
        """
        lines = [
            f"- {topic.replace('_', ' ')}: {value}"
            for topic, value in self.project_details.items() if value
        ]
        return "\n".join(lines) or "Nothing yet."

    def extract_answer(self, topic: str, answer: str) -> Dict[str, Any]:
        """
        Ask the model which part of the answer is about the topic,
        and if the user really answered or is not sure.
        """
        prompt = EXTRACT_PROMPT.format(topic=TOPICS[topic], answer=answer)
        try:
            result = json.loads(self.extract_llm.invoke(prompt))
        except Exception as e:
            logging.error(f"Error extracting answer: {e}")
            return {"value": answer, "is_clear": True}

        value = str(result.get("value", "")).strip()
        is_clear = str(result.get("is_clear", "true")).lower() == "true"

        # If the user answers with a question, they need help first.
        if answer.endswith("?"):
            is_clear = False

        # Small models sometimes change or add words.
        # If the value is not really in the user's answer, keep the full answer.
        if not value or value.lower() not in answer.lower():
            value = answer
        return {"value": value, "is_clear": is_clear}

    def ask_question(self, topic: str, last_answer: str, with_examples: bool = False) -> str:
        """
        Generate the next question with the chat model.
        If the output is not one short question, use a simple fixed question instead.
        """
        if with_examples:
            task = (
                f"the user is not sure about {TOPICS[topic]}. Help them choose: ask one question that "
                "names two or three options from the background knowledge in one sentence, "
                'for example "Would you prefer X, Y or Z?"'
            )
        else:
            task = f"ask the user about {TOPICS[topic]}."

        prompt = QUESTION_PROMPT.format(
            known=self.known_details(),
            last_answer=last_answer,
            context=self.get_context(topic),
            task=task,
        )
        try:
            reply = self.chat_llm.invoke(prompt).strip()
        except Exception as e:
            logging.error(f"Error generating question: {e}")
            reply = ""

        # Keep everything up to the first question mark, so only one question is asked.
        if "?" in reply:
            reply = reply[:reply.find("?") + 1].strip('" ')
        else:
            reply = ""

        if not reply or len(reply.split()) > 60:
            reply = f"Could you tell me more about {TOPICS[topic]}? 😊"
        return reply

    def make_summary(self) -> str:
        summary = "\n✨ Project Summary ✨\n\n"
        for key, value in self.project_details.items():
            if value.strip():
                summary += f"📌 {key.replace('_', ' ').title()}: {value}\n"
        summary += "\nThank you for sharing your vision with me! I'm excited to see this project come to life! 🚀"
        return summary


# -------------------------------
# Helper Functions: Document Loading, Splitting, Vectorstore, etc.
# -------------------------------

def check_device() -> str:
    if torch.cuda.is_available():
        device = "cuda"
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logging.info("No GPU found, using CPU instead.")
    return device

def load_pdf(pdf_path: str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)

def create_vectorstore(docs, embedding_model: Embeddings) -> FAISS:
    texts = [doc.page_content for doc in docs]
    return FAISS.from_texts(texts, embedding_model)

def load_or_create_vectorstore(pdf_path: str, index_path: str, embedding_model: Embeddings) -> FAISS:
    """
    Load the FAISS index from disk if it exists.
    Otherwise build it from the PDF and save it, so next time it loads faster.
    """
    if os.path.exists(index_path):
        logging.info(f"Loading existing vectorstore from {index_path}")
        return FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)

    documents = load_pdf(pdf_path)
    logging.info(f"PDF loaded successfully: {len(documents)} pages")
    docs = split_documents(documents)
    logging.info(f"Document split into {len(docs)} chunks")

    vectorstore = create_vectorstore(docs, embedding_model)
    vectorstore.save_local(index_path)
    logging.info(f"Saved new vectorstore to {index_path}")
    return vectorstore

def create_retriever(vectorstore, k: int = 3):
    return vectorstore.as_retriever(search_kwargs={"k": k})

def test_llm(llm: OllamaLLM) -> bool:
    try:
        response = llm.invoke("Generate a short, friendly greeting.")
        logging.info(f"LLM Test Response: {response}")
        return True
    except Exception as e:
        logging.error(f"Error testing LLM: {e}. Is Ollama running?")
        return False


# -------------------------------
# Conversation
# -------------------------------

def get_user_input() -> Optional[str]:
    """
    Read the user's answer. Returns None when the user wants to stop.
    """
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("\nAssistant: It was wonderful helping you! Goodbye! 👋")
            return None
        if user_input:
            return user_input

def interactive_questionnaire(assistant: Assistant) -> Dict[str, str]:
    """
    Ask about each topic one by one.
    If the user is not sure, ask once more with examples from the PDF.
    """
    print("\nAssistant: " + GREETING)
    topics = list(TOPICS)

    for i, topic in enumerate(topics):
        answer = get_user_input()
        if answer is None:
            return assistant.project_details

        result = assistant.extract_answer(topic, answer)
        if not result["is_clear"]:
            print("\nAssistant: " + assistant.ask_question(topic, answer, with_examples=True))
            answer = get_user_input()
            if answer is None:
                return assistant.project_details
            result = assistant.extract_answer(topic, answer)

        assistant.project_details[topic] = result["value"] if result["is_clear"] else NOT_DECIDED
        logging.info(f"Saved {topic}: {assistant.project_details[topic]}")

        if i + 1 < len(topics):
            print("\nAssistant: " + assistant.ask_question(topics[i + 1], answer))

    print("\nAssistant: Is there anything else you would like to add? (You can say 'no'.)")
    answer = get_user_input()
    if answer and answer.lower().strip(" .!") not in ["no", "nothing", "none"]:
        assistant.project_details["special_requests"] = answer
    return assistant.project_details

def save_responses(responses: Dict[str, str], output_file: str = "project_requirements.txt") -> None:
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("🌟 Website/App Project Requirements 🌟\n\n")
            for key, value in responses.items():
                f.write(f"📌 {key.replace('_', ' ').title()}:\n{value}\n\n")
        logging.info(f"Responses saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving responses: {e}")

def main() -> None:
    logging.info("Starting application...")

    device = check_device()
    embedding_model = CustomEmbeddings(device=device)
    try:
        vectorstore = load_or_create_vectorstore(PDF_PATH, INDEX_PATH, embedding_model)
    except FileNotFoundError as e:
        logging.error(e)
        sys.exit(1)
    retriever = create_retriever(vectorstore)

    # Each task gets its own settings:
    # - Extracting the answer has one correct result, so temperature is 0 and the output is JSON.
    # - Writing a question should sound natural, so a little randomness is fine.
    extract_llm = OllamaLLM(model=MODEL_NAME, temperature=0, format="json")
    chat_llm = OllamaLLM(model=MODEL_NAME, temperature=0.4, top_p=0.9, top_k=40, repeat_penalty=1.1)

    if not test_llm(chat_llm):
        sys.exit(1)

    assistant = Assistant(extract_llm, chat_llm, retriever)
    responses = interactive_questionnaire(assistant)

    print(assistant.make_summary())
    save_responses(responses)
    logging.info("Application completed successfully")

if __name__ == "__main__":
    main()
