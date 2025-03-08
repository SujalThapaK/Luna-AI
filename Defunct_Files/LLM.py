from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.graphs import Neo4jGraph  # Updated import
from langchain_community.vectorstores import Neo4jVector  # Updated import
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
from dotenv import load_dotenv
load_dotenv()

NEO4J_URI = os.getenv("neo4j_URL")
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = os.getenv("neo4j_Pass")


def initialize_llm():
    """Initialize Google's Generative AI model"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.1,
    )
    return llm


def connect_to_neo4j():
    """Connect to Neo4j database without requiring schema knowledge"""
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD
    )
    return graph


def explore_graph_schema(graph):
    """Explore the graph schema to understand available nodes and relationships"""
    # Get node labels
    node_labels = graph.query("CALL db.labels()")
    print("Node labels:", node_labels)

    # Get relationship types
    rel_types = graph.query("CALL db.relationshipTypes()")
    print("Relationship types:", rel_types)

    # Get property keys
    prop_keys = graph.query("CALL db.propertyKeys()")
    print("Property keys:", prop_keys)

    # Sample of nodes for each label
    schema_info = "Database schema information:\n"
    for label in node_labels:
        label_name = label["label"]
        sample_query = f"MATCH (n:{label_name}) RETURN n LIMIT 1"
        try:
            sample = graph.query(sample_query)
            if sample:
                schema_info += f"Sample {label_name} node: {sample}\n"
        except:
            pass

    return schema_info


def setup_graph_qa_chain(graph, llm):
    """Set up a GraphCypherQAChain that can generate Cypher queries based on the schema"""
    # Using the updated Neo4j-specific GraphCypherQAChain
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True
    )
    return chain


def setup_vector_store(llm, graph):
    """Set up a Neo4j vector store for RAG capabilities"""
    # Initialize embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Initialize vector store in Neo4j
    vector_store = Neo4jVector.from_existing_graph(
        embedding=embeddings,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name="document_embeddings",
        node_label="Document",
        text_node_properties=["content"],
        embedding_node_property="embedding"
    )

    return vector_store


def ingest_documents(vector_store, documents):
    """Ingest documents into the Neo4j vector store"""
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    # Process each document
    for doc in documents:
        # Split document into chunks
        chunks = text_splitter.split_text(doc["content"])

        # Add metadata to each chunk
        texts = []
        metadatas = []
        for chunk in chunks:
            texts.append(chunk)
            metadatas.append({
                "source": doc.get("source", "unknown"),
                "title": doc.get("title", ""),
                "doc_id": doc.get("id", "")
            })

        # Add to vector store
        vector_store.add_texts(texts=texts, metadatas=metadatas)


def setup_rag_chain(llm, vector_store):
    """Set up a RAG chain that combines retrieval with graph navigation"""
    # Create retriever
    retriever = vector_store.as_retriever()

    # Create prompt template for the RAG chain
    prompt_template = """
    Answer the following question based on the provided context and knowledge graph information.

    Context: {context}

    Question: {input}

    When generating your answer, consider both the retrieved document chunks and any relevant connections 
    in the graph. If helpful, describe relationships between entities mentioned in the documents.

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "input"])

    # Create a document chain that applies the LLM to the retrieved documents
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Create the retrieval chain
    rag_chain = create_retrieval_chain(retriever, document_chain)

    return rag_chain


def graphrag_query(query, graph_qa_chain, rag_chain):
    """Execute a query using both graph QA and RAG"""
    # First try to answer using graph query
    graph_response = graph_qa_chain.invoke({"query": query})

    # Then try RAG approach
    rag_response = rag_chain.invoke({"input": query})

    # Combine results
    combined_context = f"""
    Graph database information: {graph_response.get('result', 'No direct graph information found')}

    Retrieved documents: {rag_response.get('context', 'No relevant documents found')}
    """

    # Final reasoning using the LLM with combined context
    llm = initialize_llm()
    final_prompt = f"""
    Question: {query}

    Information from graph database and documents:
    {combined_context}

    Please provide a comprehensive answer that integrates both the graph relationships 
    and information from documents.
    """
    final_response = llm.invoke(final_prompt)

    return final_response


def main():
    # Initialize components
    llm = initialize_llm()
    graph = connect_to_neo4j()

    # Explore graph schema
    schema_info = explore_graph_schema(graph)
    print(schema_info)

    # Set up chains
    graph_qa_chain = setup_graph_qa_chain(graph, llm)
    vector_store = setup_vector_store(llm, graph)

    # Example documents - replace with your actual data
    sample_documents = [
        {"id": "1", "title": "What is Neo4j", "content": "Neo4j is a graph database management system...",
         "source": "documentation"},
        {"id": "2", "title": "Graph Relationships", "content": "In Neo4j, relationships connect nodes and represent...",
         "source": "blog"},
    ]

    # Ingest documents
    ingest_documents(vector_store, sample_documents)

    # Set up RAG chain
    rag_chain = setup_rag_chain(llm, vector_store)

    # Example query
    query = "What are the main relationships in the database and how are they used?"
    response = graphrag_query(query, graph_qa_chain, rag_chain)

    print(f"Query: {query}")
    print(f"Response: {response}")


if __name__ == "__main__":
    main()