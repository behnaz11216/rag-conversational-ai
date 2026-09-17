import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow imports of project-level modules (e.g. website_generator.py)
# when this file is executed directly from the src/ directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ---------------- Configuration ----------------
def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        logging.warning("Invalid %s; using %s", name, default)
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        logging.warning("Invalid %s; using %s", name, default)
        return default


def env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


PDF_PATH = os.getenv("RAG_PDF_PATH", "data/source.pdf")
FAISS_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
CHUNK_SIZE = env_int("CHUNK_SIZE", 1000)
CHUNK_OVERLAP = env_int("CHUNK_OVERLAP", 100)
RETRIEVAL_K = env_int("RETRIEVAL_K", 4)
TEMPERATURE = env_float("OLLAMA_TEMPERATURE", 0.3)
TOP_P = env_float("OLLAMA_TOP_P", 0.8)
REPEAT_PENALTY = env_float("OLLAMA_REPEAT_PENALTY", 1.1)
REBUILD_INDEX = env_bool("REBUILD_FAISS_INDEX", False)
GENERATE_WEBSITE = env_bool("GENERATE_WEBSITE", True)
WEBSITE_OUTPUT_DIR = os.getenv("WEBSITE_OUTPUT_DIR", "generated_website")
EXIT_COMMANDS = {"exit", "quit", "bye"}
NOT_DECIDED = "Not decided yet."


def device_name() -> str:
    if torch.cuda.is_available():
        try:
            logging.info("Using GPU: %s", torch.cuda.get_device_name(0))
        except Exception:
            logging.info("Using CUDA GPU")
        return "cuda"
    logging.info("Using CPU")
    return "cpu"


# ---------------- RAG pipeline ----------------
class CustomEmbeddings(Embeddings):
    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)
        logging.info("Embedding model: %s (%s)", model_name, device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self.model.encode(texts, convert_to_tensor=False, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_tensor=False, show_progress_bar=False).tolist()


def load_or_create_vectorstore(
    pdf_path: str,
    embeddings: CustomEmbeddings,
    index_path: str,
    rebuild: bool = False,
) -> FAISS:
    index_file = os.path.join(index_path, "index.faiss")
    if os.path.exists(index_file) and not rebuild:
        logging.info("Loading FAISS index: %s", index_path)
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    documents = PyPDFLoader(pdf_path).load()
    if not documents:
        raise ValueError("PDF contains no readable pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    texts = [d.page_content.strip() for d in chunks if d.page_content.strip()]
    if not texts:
        raise ValueError("No text was found in the PDF.")

    logging.info("Creating FAISS index from %d chunks", len(texts))
    store = FAISS.from_texts(texts, embeddings)
    os.makedirs(index_path, exist_ok=True)
    store.save_local(index_path)
    logging.info("FAISS index saved: %s", index_path)
    return store


# ---------------- Conversation agent ----------------
class ConversationAgent:
    TOPICS = [
        "product_name", "project_type", "purpose", "target_audience",
        "design_style", "core_features", "content_needs", "technical_requirements",
    ]

    LABELS = {
        "product_name": "project name",
        "project_type": "website or application type",
        "purpose": "main purpose",
        "target_audience": "target audience",
        "design_style": "design style",
        "core_features": "main features",
        "content_needs": "content requirements",
        "technical_requirements": "technical requirements",
    }

    SUMMARY_LABELS = {
        "product_name": "Product / Project Name", "project_type": "Project Type",
        "purpose": "Purpose", "target_audience": "Target Audience",
        "design_style": "Design Style", "core_features": "Core Features",
        "content_needs": "Content Needs", "technical_requirements": "Technical Requirements",
        "special_requests": "Special Requests",
    }

    FALLBACKS = {
        "product_name": "What would you like to call the project?",
        "project_type": "How would you describe the type of website or application?",
        "purpose": "What would you like the project to help users accomplish?",
        "target_audience": "Who are you building this for?",
        "design_style": "How would you like the project to look and feel?",
        "core_features": "What would you like users to be able to do?",
        "content_needs": "What kind of content should appear on the pages?",
        "technical_requirements": "Is there anything specific the project needs technically?",
    }

    OPTION_FALLBACKS = {
        "product_name": "Would a simple working name based on the topic work for now?",
        "project_type": "Would you say it is closer to an online shop, a blog, or an information website?",
        "purpose": "Should it mainly sell something, share information, or present your work?",
        "target_audience": "Is it for general visitors, customers, or a specific group such as students?",
        "design_style": "Do you prefer something minimal, colourful, or classic?",
        "core_features": "Would search, categories, or a contact form be useful for your users?",
        "content_needs": "Are you thinking of text, images, prices, or a mix of these?",
        "technical_requirements": "Do you need anything specific, such as a language option or online payment?",
    }

    NO_SPECIAL = {"no", "nope", "none", "nothing", "nothing else", "no special requests"}

    UNDECIDED_RE = re.compile(
        r"\b(no idea|not sure|no clue|dunno|idk|i\s*(?:do not|don't|dont)\s*know|"
        r"i\s*(?:have|got)\s*no idea|whatever|no preference|not decided|undecided|"
        r"anything\s+(?:is\s+)?(?:fine|ok|okay)|up to you|you\s+(?:decide|choose))\b",
        re.I,
    )
    QUESTION_RE = re.compile(
        r"^(what|which|how|why|who|do you|can you|could you|would you|give me|"
        r"suggest|tell me|help me|any idea|any suggestions?)\b",
        re.I,
    )
    USER_VOICE_RE = re.compile(
        r"\b(i want|i need|i'd like|my website|my app|my project|our website|our app)\b",
        re.I,
    )
    FILLER = {
        "i","we","you","it","this","that","a","an","the","and","or","but","for","of","to","in","on","at","by","with","from",
        "my","our","your","me","us","is","am","are","be","was","were","do","does","want","need","like","would","should","will","can","could","have","has","had",
        "called","call","named","name","maybe","really","just","very","please","some","any","all","kind","kinds","sort","type","actually","about","more","most","other","something","anything","everything","thing","things",
    }

    def __init__(self, llm: OllamaLLM, retriever: Optional[Any] = None):
        self.llm = llm
        self.retriever = retriever
        self.current_topic = self.TOPICS[0]
        self.details = {topic: "" for topic in self.TOPICS}
        self.details["special_requests"] = ""
        self.covered = set()
        self.history: List[Tuple[str, str]] = []
        self.questions: List[str] = []
        self.undecided_count: Dict[str, int] = {}
        self.off_topic_count: Dict[str, int] = {}
        self.special_done = False

    # ---------- helpers ----------
    @staticmethod
    def normalize(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").lower().strip()).strip(".,!?;:\"'()[]{}")

    def topic_label(self, topic: str) -> str:
        return self.LABELS.get(topic, topic.replace("_", " "))

    def words(self, text: str) -> List[str]:
        return [w for w in re.findall(r"[a-z0-9]+", self.normalize(text)) if w not in self.FILLER]

    def overlap(self, a: str, b: str) -> bool:
        a, b = set(self.words(a)), set(self.words(b))
        return bool(a and b) and len(a & b) / min(len(a), len(b)) >= 0.8

    def is_undecided(self, text: str) -> bool:
        return len(text.split()) <= 8 and bool(self.UNDECIDED_RE.search(text))

    def is_user_question(self, text: str) -> bool:
        return text.strip().endswith("?") and bool(self.QUESTION_RE.match(text.strip()))

    def explicit(self, value: str, user_text: str) -> bool:
        value, user_text = self.normalize(value), self.normalize(user_text)
        return bool(value) and len(value) >= 3 and bool(re.search(r"\b" + re.escape(value) + r"\b", user_text))

    def topic_open(self, topic: str) -> bool:
        """Open when unanswered, or answered only with 'Not decided yet.'."""
        return topic not in self.covered or self.details.get(topic, "").strip() == NOT_DECIDED

    # ---------- extraction ----------
    def parse_json(self, text: str) -> Dict[str, Any]:
        text = re.sub(r"^```json\s*|^```\s*|\s*```$", "", (text or "").strip(), flags=re.I)
        match = re.search(r"\{.*\}", text, re.S)
        for candidate in (text, match.group(0) if match else ""):
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, TypeError):
                pass
        return {}

    def extract(self, user_text: str) -> Dict[str, str]:
        schema = {topic: "" for topic in self.TOPICS}
        prompt = f"""
Extract project requirements explicitly stated in USER MESSAGE.
Return JSON only using exactly these keys:
{json.dumps(schema)}

USER MESSAGE:
{user_text}

Rules:
- Copy only words that are explicitly present in the user message.
- Never infer, guess, summarize, or use outside knowledge.
- If a field is not explicitly stated, return an empty string.
- Do not put the same phrase into multiple fields unless the user clearly expressed it as separate information for both.
- Special requests are not extracted here.
"""
        try:
            data = self.parse_json(self.llm.invoke(prompt))
        except Exception as exc:
            logging.warning("Requirement extraction failed: %s", exc)
            return {}

        result = {}
        for topic in self.TOPICS:
            value = data.get(topic, "")
            if not isinstance(value, str) or not value.strip():
                continue
            if self.explicit(value, user_text):
                result[topic] = value.strip()
            else:
                logging.warning("Ignoring inferred requirement for '%s': %s", topic, value.strip())
        return result

    def choose_value(self, answer: str, extracted: str, all_values: Dict[str, str]) -> str:
        """Store the short extracted value only when nothing the user said is lost."""
        if not extracted or not self.explicit(extracted, answer):
            return answer.strip()
        answer_words = self.words(answer)
        covered = set(self.words(extracted))
        for value in all_values.values():
            covered.update(self.words(value))
        uncovered = [w for w in answer_words if w not in covered]
        if answer_words and len(uncovered) > int(len(answer_words) * 0.3):
            return answer.strip()
        return extracted.strip()

    # ---------- conversation state ----------
    def process_answer(self, text: str) -> bool:
        """
        Store the answer. Returns True when the answer was about another
        topic, so the current question is still waiting for an answer.
        """
        text = text.strip()
        self.history.append(("user", text))

        if self.current_topic not in self.TOPICS:
            return False

        extracted = self.extract(text)
        current = extracted.get(self.current_topic, "")
        elsewhere = {k: v for k, v in extracted.items() if k != self.current_topic}

        # The answer talks about another topic and says nothing about the
        # current one: store it where it belongs and ask again (once).
        answers_current = True
        if not current and elsewhere:
            if self.off_topic_count.get(self.current_topic, 0) < 1:
                self.off_topic_count[self.current_topic] = 1
                answers_current = False
                logging.info(
                    "Answer looks like it is about %s, not '%s'. Asking again.",
                    list(elsewhere), self.current_topic,
                )

        if answers_current:
            self.details[self.current_topic] = self.choose_value(text, current, extracted)
            self.covered.add(self.current_topic)

        stored = self.details.get(self.current_topic, "")
        for topic, value in elsewhere.items():
            if not self.topic_open(topic):
                continue
            if answers_current and self.overlap(value, stored):
                logging.info("Ignoring duplicated extraction for '%s': %s", topic, value)
                continue
            self.details[topic] = value
            self.covered.add(topic)
            logging.info("Detected explicit requirement for '%s': %s", topic, value)

        if answers_current:
            self.next_topic()
        return not answers_current

    def next_topic(self) -> None:
        for topic in self.TOPICS:
            if topic not in self.covered:
                self.current_topic = topic
                return
        self.current_topic = "special_requests" if not self.special_done else ""

    def handle_undecided(self, text: str) -> bool:
        """First time: offer examples. Second time: record and move on."""
        topic = self.current_topic
        self.history.append(("user", text.strip()))
        self.undecided_count[topic] = self.undecided_count.get(topic, 0) + 1

        if self.undecided_count[topic] == 1:
            print(f"\nAssistant: {self.generate_question(examples=True)}")
            return True

        self.details[topic] = NOT_DECIDED
        self.covered.add(topic)
        self.next_topic()
        print(f"\nAssistant: No problem. I noted that we have not decided the {self.topic_label(topic)} yet.")
        return False

    def special_answer(self, text: str) -> None:
        text = text.strip()
        self.history.append(("user", text))
        if self.normalize(text) in self.NO_SPECIAL or self.is_undecided(text):
            self.details["special_requests"] = "None specified."
        else:
            self.details["special_requests"] = text
        self.special_done = True
        self.current_topic = ""

    def all_topics_answered(self) -> bool:
        """Every topic was asked, including the ones left undecided."""
        return all(topic in self.covered for topic in self.TOPICS)

    def complete(self) -> bool:
        """Every topic has a real decision (used before website generation)."""
        return all(
            self.details.get(topic, "").strip() not in ("", NOT_DECIDED)
            for topic in self.TOPICS
        )

    def undecided_topics(self) -> List[str]:
        return [t for t in self.TOPICS if self.details.get(t, "").strip() == NOT_DECIDED]

    # ---------- question generation ----------
    def build_context(self) -> str:
        if not self.retriever:
            return "No knowledge-base context available."
        query = self.topic_label(self.current_topic)
        project_type = self.details.get("project_type", "")
        if project_type:
            query += " | " + project_type
        try:
            docs = self.retriever.invoke(query)
            return "\n\n---\n\n".join(
                d.page_content.strip() for d in docs if d.page_content.strip()
            ) or "No relevant context."
        except Exception as exc:
            logging.warning("Retrieval failed: %s", exc)
            return "No relevant context."

    def question_valid(self, text: str) -> bool:
        text = (text or "").strip()
        if not text.endswith("?") or text.count("?") != 1 or len(text.split()) > 45:
            return False
        if self.USER_VOICE_RE.search(text):
            return False
        name = self.details.get("product_name", "").strip()
        if name and re.match(rf"^{re.escape(name)}\s+(is|will|has)\b", text, re.I):
            return False
        return True

    def clean_question(self, text: str) -> str:
        text = re.sub(
            r"^(assistant(?:'s)?\s+next\s+question|assistant|question|next question)\s*:\s*",
            "", (text or "").strip(), flags=re.I,
        )
        text = text.strip("\"'").lstrip("-•* ").strip()
        if "?" in text:
            text = text[:text.find("?") + 1]
        return text.strip()

    def already_asked(self, question: str) -> bool:
        q = set(self.words(question))
        if not q:
            return False
        for old in self.questions:
            old_words = set(self.words(old))
            if old_words and len(q & old_words) / min(len(q), len(old_words)) >= 0.75:
                return True
        return False

    def recent_questions(self) -> str:
        return "\n".join(f"- {q}" for q in self.questions[-4:]) or "None yet."

    def last_user_message(self) -> str:
        for role, message in reversed(self.history):
            if role == "user":
                return message
        return "Nothing yet."

    def remember(self, question: str) -> str:
        self.questions.append(question)
        self.history.append(("assistant", question))
        return question

    def generate_question(self, examples: bool = False, returning: bool = False) -> str:
        if self.current_topic == "special_requests":
            return self.remember(
                "Is there anything else you would like to add as a special request or preference?"
            )

        label = self.topic_label(self.current_topic)
        fallback = (
            self.OPTION_FALLBACKS.get(self.current_topic, self.FALLBACKS[self.current_topic])
            if examples
            else self.FALLBACKS[self.current_topic]
        )
        if returning:
            fallback = f"Thanks, I noted that. Coming back to the {label}: {fallback[0].lower()}{fallback[1:]}"

        prompt = f"""
You are a conversational AI assistant gathering website/application requirements.
Write ONE short question about the CURRENT MISSING REQUIREMENT: {label}.

Confirmed requirements (already answered, never ask about them again):
{self.confirmed()}

Questions already asked (never ask them again, not even in other words):
{self.recent_questions()}

The user's last message:
{self.last_user_message()}

Knowledge-base context (reference only; never treat it as a user requirement):
{self.build_context()}

Rules:
- Ask exactly one question and return only that question.
- Do not answer your own question.
- Do not describe the project as a fact.
- Do not write in the user's voice.
- Do not invent requirements.
"""
        if examples:
            prompt += "- The user asked for help, so offer two or three optional examples in the question.\n"
        if returning:
            prompt += (
                "- The last message was about another part of the project. "
                f"Briefly accept it and bring the conversation back to {label}.\n"
            )

        try:
            candidate = self.clean_question(self.llm.invoke(prompt))
            if self.question_valid(candidate) and not self.already_asked(candidate):
                return self.remember(candidate)
            logging.info("Generated question rejected: %s", candidate[:80])
        except Exception as exc:
            logging.warning("Question generation failed: %s", exc)

        return self.remember(fallback)

    def confirmed(self) -> str:
        return "\n".join(
            f"- {self.topic_label(t)}: {self.details[t]}" for t in self.TOPICS if self.details[t]
        ) or "None yet."

    def summary(self) -> str:
        lines = ["\n" + "=" * 60, "PROJECT REQUIREMENTS SUMMARY", "=" * 60]
        for topic in self.TOPICS + ["special_requests"]:
            value = self.details.get(topic, "").strip()
            if not value:
                value = NOT_DECIDED if topic in self.TOPICS else "None specified."
            lines += [f"\n{self.SUMMARY_LABELS[topic]}:", f"  {value}"]
        return "\n".join(lines)


# ---------------- Interactive flow ----------------
def build_llm() -> OllamaLLM:
    logging.info("Using Ollama model: %s", OLLAMA_MODEL)
    return OllamaLLM(model=OLLAMA_MODEL, temperature=TEMPERATURE, top_p=TOP_P, repeat_penalty=REPEAT_PENALTY)


def read_input() -> Optional[str]:
    text = input("\nYou: ").strip()
    if text.lower() in EXIT_COMMANDS:
        print("\nAssistant: Goodbye!")
        return None
    return text


def run_conversation(llm: OllamaLLM, retriever: Any) -> Dict[str, str]:
    agent = ConversationAgent(llm, retriever)
    print("\n" + "=" * 60)
    print("RAG Conversational AI Assistant")
    print("=" * 60)
    print("Answer the questions to define your website or application project.")
    print("Type 'exit' to stop.\n")

    greeting = "Hi! I'll help you define the project step by step. What would you like to call it?"
    agent.remember(greeting)

    try:
        print(f"Assistant: {greeting}")

        # Required topics.
        while not agent.all_topics_answered():
            user = read_input()
            if user is None:
                return agent.details
            if not user:
                print("\nAssistant: Could you provide a little more information about that?")
                continue

            if agent.is_user_question(user):
                agent.history.append(("user", user))
                print(f"\nAssistant: {agent.generate_question(examples=True)}")
                continue

            if agent.is_undecided(user):
                if agent.handle_undecided(user):
                    continue
                if not agent.all_topics_answered():
                    print(f"\nAssistant: {agent.generate_question()}")
                continue

            if agent.process_answer(user):
                print(f"\nAssistant: {agent.generate_question(returning=True)}")
                continue

            if not agent.all_topics_answered():
                print(f"\nAssistant: {agent.generate_question()}")

        # Special requests (optional, asked once).
        print(f"\nAssistant: {agent.generate_question()}")
        while not agent.special_done:
            user = read_input()
            if user is None:
                return agent.details
            if not user:
                print("\nAssistant: You can say 'none' if you have no additional requests.")
                continue
            agent.special_answer(user)

        print(agent.summary())
        undecided = agent.undecided_topics()
        if undecided:
            labels = ", ".join(agent.topic_label(t) for t in undecided)
            print(f"\nAssistant: We have not decided the {labels} yet. Nothing was invented for it.")
        return agent.details

    except (KeyboardInterrupt, EOFError):
        print("\n\nAssistant: The conversation was stopped by the user.")
        return agent.details
    except Exception as exc:
        logging.error("Conversation error: %s", exc)
        print("\nAssistant: A technical error occurred.")
        return agent.details


# ---------------- Output ----------------
def save_requirements(requirements: Dict[str, str], path: str = "project_requirements.txt") -> None:
    if not any(value.strip() for value in requirements.values()):
        logging.info("No requirements collected. Nothing was saved.")
        return
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("RAG Conversational AI - Project Requirements\n" + "=" * 60 + "\n\n")
            for key, value in requirements.items():
                if value.strip():
                    f.write(f"{key.replace('_', ' ').title()}:\n{value.strip()}\n\n")
        logging.info("Project requirements saved to %s", path)
    except OSError as exc:
        logging.error("Could not save requirements: %s", exc)


def generate_website(requirements: Dict[str, str], llm: OllamaLLM, retriever: Any) -> None:
    if not GENERATE_WEBSITE:
        logging.info("Website generation disabled.")
        return

    missing = [t for t in ConversationAgent.TOPICS if not requirements.get(t, "").strip()]
    if missing:
        logging.info("Requirements are incomplete (%s). Website generation skipped.", missing)
        return

    undecided = [t for t in ConversationAgent.TOPICS if requirements[t].strip() == NOT_DECIDED]
    if undecided:
        logging.info("Undecided requirements, nothing may be invented for them: %s", undecided)

    print("\nAssistant: Generating the website files. This can take a moment...")
    try:
        from website_generator import WebsiteGenerator

        generator = WebsiteGenerator(llm=llm, retriever=retriever, output_dir=WEBSITE_OUTPUT_DIR)
        result = generator.generate(dict(requirements))
        print("\nAssistant: Website files generated successfully.")
        if isinstance(result, dict):
            for key, value in result.items():
                print(f"  {key.upper()}: {value}")
        else:
            print(f"  Output: {result}")
    except ImportError as exc:
        logging.error("Website generator is not available: %s", exc)
        print("\nAssistant: The requirements were saved, but the website generator could not be loaded.")
    except Exception as exc:
        logging.error("Website generation failed: %s", exc)
        print("\nAssistant: The requirements were saved, but website generation failed.")


def main() -> None:
    logging.info("Starting RAG Conversational AI application")
    llm = build_llm()
    try:
        llm.invoke("Reply with exactly: READY")
    except Exception as exc:
        logging.error("Ollama is not available: %s", exc)
        return

    embeddings = CustomEmbeddings(EMBEDDING_MODEL, device_name())
    try:
        store = load_or_create_vectorstore(PDF_PATH, embeddings, FAISS_PATH, REBUILD_INDEX)
    except (FileNotFoundError, ValueError) as exc:
        logging.error("Could not prepare knowledge base: %s", exc)
        return

    retriever = store.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    requirements = run_conversation(llm, retriever)
    save_requirements(requirements)
    generate_website(requirements, llm, retriever)
    logging.info("Application completed successfully.")


if __name__ == "__main__":
    main()