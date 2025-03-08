import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_neo4j import Neo4jVector
from langchain_neo4j import Neo4jGraph
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from PIL import Image
import base64
import os
from dotenv import load_dotenv
from io import BytesIO
import re

load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Connect to Neo4j
graph = Neo4jGraph(
    url=os.getenv("neo4j_URL"),
    username="neo4j",
    password=os.getenv("neo4j_Pass")
)


# Hybrid Retriever Class (unchanged from previous implementation)
class HybridRetriever:
    def __init__(self, graph, embeddings):
        self.vector_store = Neo4jVector(
            embedding=embeddings,
            url="neo4j+s://02955f7e.databases.neo4j.io",
            username="neo4j",
            password="2U3gCZIk3rYw-qnXNr2l_pmhFrY8HOS_XimfyUe_xdY",
            index_name="medical_entities",
            node_label="Entity",
            text_node_property="text"
        )
        self.graph = graph

    def retrieve(self, query):
        # Rephrase the query using Gemini
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        rephrase_prompt = REPHRASE_QUERY_PROMPT.format(query=query)
        rephrased_query = llm.invoke(rephrase_prompt).content

        print(f"Rephrased Query: '{rephrased_query}'")

        # Vector similarity search
        vector_results = self.vector_store.similarity_search(query, k=5)

        # Graph pattern search (using the rephrased query)
        try:
            graph_results = self.graph.query("""
                CALL db.index.fulltext.queryNodes("entityIndex", $query) 
                YIELD node, score
                MATCH (node)-[r]-(related)
                RETURN node.text AS node, type(r) AS relationship, related.text AS related_node
                LIMIT 100
            """, {"query": rephrased_query})
        except Exception as e:
            print(f"Error executing Neo4j query: {e}")
            return {"vector": vector_results, "graph": []}

        return {
            "vector": vector_results,
            "graph": graph_results
        }


# Prompt Templates
# Non-Concept Mode Medical Prompt
STANDARD_MEDICAL_PROMPT = PromptTemplate(
    input_variables=["question", "text_context", "graph_context", "conversation_history"],
    template="""Your name is Luna, you are a medical knowledge assistant. Use the following information to answer.
    First, use the provided text context to construct your answer.
    Then, use the provided Graph relationships to add to or enhance the answer.
    Consider the conversation history for context.

    Text Context:
    {text_context}

    Graph Relationships:
    {graph_context}

    Conversation History:
    {conversation_history}

    Question: {question}
    Answer:"""
)

# Concept Mode Medical Prompt
CONCEPT_MEDICAL_PROMPT = PromptTemplate(
    input_variables=["question", "text_context", "graph_context", "conversation_history"],
    template="""Your name is Luna, you are a medical knowledge assistant. Provide a comprehensive, in-depth explanation using simple, accessible language.

    Explanation Guidelines:
    - Break down complex medical concepts into easily understandable terms
    - Provide context and background information
    - Explain medical terminology in layman's language
    - Offer a holistic understanding of the topic
    - Consider the conversation history for relevant context

    Text Context:
    {text_context}

    Graph Relationships:
    {graph_context}

    Conversation History:
    {conversation_history}

    Question: {question}
    Comprehensive Explanation:"""
)

# Rephrase Query Prompt (unchanged)
REPHRASE_QUERY_PROMPT = PromptTemplate(
    input_variables=["query"],
    template="""Rephrase the following medical query to ensure it:
    - Does not contain special characters that could cause Lucene parsing errors.
    - Follows Lucene query guidelines.
    - Is not unnecessarily long.
    - Maintains its essence and all relevant medical information.

    Original Query: {query}

    Rephrased Query:"""
)


# Function to format conversation history
def format_conversation_history(chat_history):
    if not chat_history:
        return "No previous conversation."

    formatted_history = ""
    for message in chat_history:
        role = "User" if message["role"] == "user" else "Assistant"
        formatted_history += f"{role}: {message['content']}\n\n"

    return formatted_history


# Function to estimate token count (rough approximation)
def estimate_token_count(text):
    # A very rough approximation: ~4 characters per token
    return len(text) // 4


# Hybrid Chain
class HybridQAChain:
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        self.TOKEN_LIMIT = 950000  # Set token limit

    def format_context(self, retrieval_result):
        graph_context = "\n".join(
            [f"{row['node']} -[{row['relationship']}]-> {row['related_node']}"
             for row in retrieval_result['graph']]
        )
        text_context = " ".join(
            [doc.text for doc in retrieval_result['vector']]
        )
        return {"graph_context": graph_context, "text_context": text_context}

    def run(self, question, chat_history, image_path=None, concept_mode=True):
        retrieved = self.retriever.retrieve(question)
        formatted = self.format_context(retrieved)

        # Format conversation history
        conversation_history = format_conversation_history(chat_history)

        # Select appropriate prompt based on concept mode
        prompt_template = CONCEPT_MEDICAL_PROMPT if concept_mode else STANDARD_MEDICAL_PROMPT

        prompt = prompt_template.format(
            question=question,
            conversation_history=conversation_history,
            **formatted
        )

        # Check token count
        token_count = estimate_token_count(prompt)
        if token_count > self.TOKEN_LIMIT:
            return {
                "answer": "Error: The conversation has become too long. Please refresh the page to start a new conversation.",
                "error": "token_limit_exceeded"
            }

        messages = []
        messages.append({"type": "text", "text": prompt})

        if image_path:
            try:
                image = Image.open(image_path)
                buffered = BytesIO()
                image.save(buffered, format=image.format)
                img_b64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                messages.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image.format.lower()};base64,{img_b64_str}"
                    }
                })
            except Exception as e:
                print(f"Error processing image: {e}")
                messages[0]["text"] = messages[0]["text"] + "\nError processing the provided image."

        human_message = HumanMessage(content=messages)
        result = self.llm.invoke([human_message])

        return {
            "answer": result.content,
            "error": None
        }


def main():
    st.set_page_config(page_title="Medical QA Assistant", page_icon="🩺")

    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Initialize retriever and QA chain
    retriever = HybridRetriever(graph, embeddings)
    qa_chain = HybridQAChain(retriever)

    # Sidebar for image upload and concept mode
    st.sidebar.title("Medical QA Options")
    uploaded_image = st.sidebar.file_uploader(
        "Choose an image",
        type=['jpg', 'png'],
        help="Upload a medical-related image"
    )

    # Concept mode toggle
    concept_mode = st.sidebar.toggle("Academic mode", help="Enable detailed conceptual explanation", value=True)
    st.sidebar.markdown(
        "<div style= margin-top: 10px;>Made with 💌 from Nepal.</div>",
        unsafe_allow_html=True
    )
    # Main chat interface
    st.title("🩺 Luna")
    st.markdown("<div style= font-size: 25px;><b>Your Personal AI Medical Assistant</b></div>", unsafe_allow_html=True)

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Enter your medical query"):
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt
        })

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Show loading indicator
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your query..."):
                try:
                    # Prepare image path
                    image_path = uploaded_image.name if uploaded_image else None

                    # Run QA chain
                    result = qa_chain.run(prompt, st.session_state.chat_history, image_path, concept_mode)

                    # Check if token limit was exceeded
                    if result.get("error") == "token_limit_exceeded":
                        st.error(result["answer"])
                        # Clear the chat history
                        st.session_state.chat_history = []
                        st.rerun()
                    else:
                        # Display assistant response
                        st.markdown(result["answer"])

                        # Add assistant response to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": result["answer"]
                        })
                except Exception as e:
                    st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()