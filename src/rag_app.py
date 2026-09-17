import logging
import os
from typing import Any, Dict, List, Optional

import torch
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class SentenceTransformerEmbeddings(Embeddings):
    """LangChain-compatible wrapper around SentenceTransformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(texts, convert_to_tensor=False)
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        vector = self.model.encode(text, convert_to_tensor=False)
        return vector.tolist()


class ConversationalRequirementsAssistant:
    """Track requirements and generate adaptive follow-up questions."""

    REQUIRED_TOPICS = [
        "project_type",
        "purpose",
        "target_audience",
        "design_style",
        "core_features",
        "content_needs",
        "technical_requirements",
    ]

    def __init__(self, llm: OllamaLLM, retriever: Optional[Any] = None) -> None:
        self.llm = llm
        self.retriever = retriever
        self.conversation_history: List[tuple[str, str]] = []
        self.current_topic = "product_name"
        self.project_details: Dict[str, str] = {
            "product_name": "",
            "project_type": "",
            "purpose": "",
            "target_audience": "",
            "design_style": "",
            "core_features": "",
            "content_needs": "",
            "technical_requirements": "",
            "special_requests": "",
        }

    def process_response(self, user_input: str) -> None:
        self.conversation_history.append(("user", user_input))
        if self.current_topic in self.project_details:
            self.project_details[self.current_topic] = user_input.strip()
        self._advance_topic()

    def _advance_topic(self) -> None:
        sequence = ["product_name"] + self.REQUIRED_TOPICS + ["special_requests"]
        try:
            index = sequence.index(self.current_topic)
            if index + 1 < len(sequence):
                self.current_topic = sequence[index + 1]
        except ValueError:
            self.current_topic = "special_requests"

    def generate_follow_up(self) -> str:
        recent_history = self.conversation_history[-3:]
        prompt = f"""
You are a conversational assistant that helps a user define a website or application project.

Current topic: {self.current_topic}

Project details:
{self.project_details}

Recent conversation:
{recent_history}

Instructions:
- Ask exactly one focused question.
- Adapt the question to the user's previous answers.
- Keep the language simple and friendly.
- Stay focused on collecting project requirements.
- Do not invent requirements or facts.
- If suggestions are useful, keep them brief.
"""
        try:
            response = self.llm.invoke(prompt).strip()
        except Exception as exc:
            logging.error("Follow-up generation failed: %s", exc)
            response = f"Could you tell me more about {self.current_topic.replace('_', ' ')}?"
        self.conversation_history.append(("assistant", response))
        return response

    def has_sufficient_information(self) -> bool:
        return all(self.project_details[topic].strip() for topic in self.REQUIRED_TOPICS)

    def generate_project_summary(self) -> str:
        prompt = f"""
Create a concise, structured summary of the project requirements below.

Project details:
{self.project_details}

Preserve the user's intent. Do not add requirements that were not provided.
"""
        try:
            summary = self.llm.invoke(prompt).strip()
        except Exception as exc:
            logging.error("Summary generation failed: %s", exc)
            summary = self._fallback_summary()
        self.conversation_history.append(("assistant", summary))
        return summary

    def _fallback_summary(self) -> str:
        lines = ["Project Requirements", ""]
        for key, value in self.project_details.items():
            if value.strip():
                lines.append(f"- {key.replace('_', ' ').title()}: {value}")
        return "\n".join(lines)


def load_documents(pdf_path: str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    return PyPDFLoader(pdf_path).load()


def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def create_vectorstore(documents, embeddings: Embeddings) -> FAISS:
    texts = [doc.page_content for doc in documents]
    return FAISS.from_texts(texts, embeddings)


def build_llm() -> OllamaLLM:
    return OllamaLLM(
        model="llama3.1:8b",
        temperature=0.4,
        top_p=0.8,
        repeat_penalty=1.1,
    )


def run_interactive_assistant(llm: OllamaLLM, retriever: Any) -> Dict[str, str]:
    assistant = ConversationalRequirementsAssistant(llm=llm, retriever=retriever)
    print("\nRAG Conversational AI Assistant")
    print("Type 'exit' to stop.\n")

    while not assistant.has_sufficient_information():
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit", "bye"}:
            break
        if len(user_input) < 3:
            print("Assistant: Please provide a little more detail.")
            continue
        assistant.process_response(user_input)
        print(f"Assistant: {assistant.generate_follow_up()}\n")

    if assistant.has_sufficient_information():
        print("\n--- Project Summary ---")
        print(assistant.generate_project_summary())

    return assistant.project_details


def main() -> None:
    pdf_path = os.getenv("RAG_PDF_PATH", "data/source.pdf")
    index_path = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Embedding/inference environment: %s", device)

    documents = load_documents(pdf_path)
    logging.info("Loaded %d document pages.", len(documents))

    chunks = split_documents(documents)
    logging.info("Created %d document chunks.", len(chunks))

    embeddings = SentenceTransformerEmbeddings()
    vectorstore = create_vectorstore(chunks, embeddings)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    vectorstore.save_local(index_path)
    logging.info("FAISS index saved to %s", index_path)

    retriever = vectorstore.as_retriever()
    llm = build_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )
    _ = qa_chain

    run_interactive_assistant(llm, retriever)


if __name__ == "__main__":
    main()
