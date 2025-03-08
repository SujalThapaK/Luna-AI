import os

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_neo4j import Neo4jVector
from langchain_neo4j import Neo4jGraph
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from PIL import Image
import base64
from io import BytesIO
import re

# Initialize components
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Connect to Neo4j
graph = Neo4jGraph(
    url=os.getenv("neo4j_URL"),
    username="neo4j",
    password=os.getenv("neo4j_Pass")
)

# Hybrid Retriever Class
class HybridRetriever:
    def __init__(self, graph, embeddings):
        self.vector_store = Neo4jVector(
            embedding=embeddings,
            url=os.getenv("neo4j_URL"),
            username="neo4j",
            password=os.getenv("neo4j_Pass"),
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

# Custom prompt template
MEDICAL_PROMPT = PromptTemplate(
    input_variables=["question", "text_context", "graph_context"],
    template="""You are a medical knowledge assistant. Use the following information to answer.
    First, use the provided text context to construct your answer.
    Then, use the provided Graph relationships to add to or enhance the answer.

    Text Context:
    {text_context}

    Graph Relationships:
    {graph_context}

    Question: {question}
    Answer:"""
)

# Prompt for shortening the query
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

# Hybrid Chain
class HybridQAChain:
    def __init__(self, retriever):
        self.retriever = retriever

    def format_context(self, retrieval_result):
        graph_context = "\n".join(
            [f"{row['node']} -[{row['relationship']}]-> {row['related_node']}"
             for row in retrieval_result['graph']]
        )
        text_context = " ".join(
            [doc.text for doc in retrieval_result['vector']]
        )
        return {"graph_context": graph_context, "text_context": text_context}

    def run(self, question, image_path=None):
        retrieved = self.retriever.retrieve(question)
        formatted = self.format_context(retrieved)
        prompt = MEDICAL_PROMPT.format(
            question=question,
            **formatted
        )

        messages = []
        messages.append({"type": "text", "text": prompt})

        if image_path:
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
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
        else:
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

        human_message = HumanMessage(content=messages)
        return llm.invoke([human_message])

# Initialize components
retriever = HybridRetriever(graph, embeddings)
qa_chain = HybridQAChain(retriever)

# Example usage
question = '''A 14-year-old boy is brought to the emergency department by his parents because of a 1-month history of intermittent
right knee pain that has worsened during the past day. He rates his current pain as a 6 on a 10-point scale and says that
it worsens when he walks and lessens when he sits. During the past 2 weeks, he has been walking 1 mile daily in
preparation for participation in the school marching band. He has not taken any medications for his pain. He sustained
a right tibia and fibula fracture at the age of 8 years after a skateboarding accident, which was treated with internal
fixation and casting. He has asthma treated with inhaled budesonide daily and inhaled albuterol as needed. His mother
has type 2 diabetes mellitus, and his maternal grandmother has osteoporosis. The patient is 170 cm (5 ft 7 in; 77th
percentile) tall and weighs 88 kg (195 lb; >95th percentile); BMI is 31 kg/m2
(98th percentile). Temperature is 37.0°C
(98.6°F), pulse is 95/min, and blood pressure is 130/80 mm Hg. Physical examination shows hyperpigmented, thickened
skin at the nape of the neck. There is tenderness to palpation of the anterior aspect of the right hip and limited range of
motion on abduction, internal rotation, and flexion of the right hip. The left hip and knees are nontender; range of motion
is full in all directions. The remainder of the examination discloses no abnormalities. Which of the following factors in
this patient’s history most increased his risk for developing this condition?

(A) BMI
(B) Family history
(C) Medication use
(D) Previous fractures
(E) Recent physical activity'''
result = qa_chain.run(question, "./image1.png")
print(result.content)