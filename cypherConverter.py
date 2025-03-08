from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def process_text_chunk(chunk, llm):
    prompt = f"""
    Extract medical entities, anatomical structures, relationships, and properties from the following text section and represent them as a series of Cypher CREATE statements to build a Neo4j graph. Each node should be created using a statement like:

      CREATE (n:Label {{id: '...', name: '...', ...}});

    And each relationship should be created using a statement like:

      MATCH (a), (b) WHERE a.id = '...' AND b.id = '...'
      CREATE (a)-[:RELATIONSHIP_TYPE {{...}}]->(b);
    
    Make sure every node has a description property with a short, holistic description of the node and its releavant medical entity, anatomical structure, properties, etc.  
    (Include as much qualitative and quantitative information about the releavant node as possible, especially within the node details.)
    
    Do not include any markdown formatting, additional commentary, or extra text. Return only the raw Cypher statements.

    Text Section: "{chunk}"
    """

    messages = [
        SystemMessage(content="You are an expert in biomedical knowledge graph extraction for Neo4j."),
        HumanMessage(content=prompt)
    ]

    response = llm.invoke(messages)

    # Clean the response content: remove any unwanted prefixes
    cypher_output = response.content.strip()
    if cypher_output.lower().startswith("cypher"):
        idx = cypher_output.find("CREATE")
        if idx != -1:
            cypher_output = cypher_output[idx:]

    return cypher_output


def merge_cypher_statements(statements):
    # Merge all Cypher statements by concatenating them with newlines.
    merged = "\n".join(statements)
    return merged


def extract_graph_from_text(text):
    print(os.getenv("GOOGLE_API_KEY"))
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Adjusted for dense medical text
        chunk_overlap=300,  # Maintain context
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_text(text)
    statements = []

    for idx, chunk in enumerate(chunks, start=1):
        print(f"Processing chunk {idx}/{len(chunks)}")
        chunk_statements = process_text_chunk(chunk, llm)
        print("Cypher statements for chunk:", chunk_statements)
        with open("medical_knowledge_graph.cypher", "a", encoding="utf-8") as f:
            f.write(chunk_statements)

    return merge_cypher_statements(statements)


# Read medical text from file
with open("Anatomy_Gray.txt", "r", encoding="utf-8") as file:
    text_input = file.read()

# Process and merge chunks into Cypher statements
neo4j_cypher = extract_graph_from_text(text_input)


print("Cypher statements saved to medical_knowledge_graph.cypher")
