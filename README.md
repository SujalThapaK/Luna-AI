# 🩺LUNA: Your Personal AI Medical Assistant

Luna-AI is an LLM trained on medical textbook(s), medical entrance exam questions and conversations between real-life conversations between Patients and General Practitioners.
<br/><br/>
The application uses GraphRAG (Based on Neo4j) to augment the existing capabilities and medical knowledge of a LLM0 (by default, Gemini). A primitve demo for the application has been made publicly available on:
<br/>https://luna-ai.streamlit.app/
## Architecture and Dataset
![Architecture_Diagram](https://github.com/user-attachments/assets/2c474742-0f1e-4920-bde1-626461facef6)
A diagram showcasing the Architecture of Luna AI has been shown above. Succinctly put, the whole application operates using four simple steps:
<br/>
1) <b>User Query Submission:</b> The user submits a medical-related query through the chatbox within Luna AI. The application extracts important keywords from the query to facilitate an optimized search.
   
2) <b>Neo4j Hybrid RAG:</b> The extracted keywords are then sent to Neo4j where two types of search are performed using it: <b>Vector Search</b> and <b>Graph Search</b>. Vector Search finds nodes with semantically similar medical terms using a vector index, and then a Graph Search is performed on said nodes, examining them and their relationships to find releavant information. The results of the Vector and Graph search are then combined to form the sum releavant context.
   
3) <b>LLM Prompt Augmentation:</b> The releavant context is then augmented to the original query string variable, and then sent via the LangChain API to an LLM of the user's choice (by default, Gemini).

4) <b>Displaying the AI response:</b> The response from the LLM is then displayed as a chat bubble in the Luna AI frontend via Streamlit.
<br/>
<br/>
As evident in the above architecture diagram, the Neo4j knowledge graph plays a key role in the functioning of Luna AI. Said knowledge graph contains information extracted from multiple academic and governmental sources, pain-stakingly restructured as nodes and relationships expressed as Cypher queries in order to upload them to Neo4j. The main sources of information for the knowledge graph has been displayed and listed below:<br/><br/>

![diagram-export-3-8-2025-8_32_36-AM](https://github.com/user-attachments/assets/549aa491-b54f-475f-94e6-cda1066d4b35)
Main Sources:
1) MedQuad: https://github.com/abachaa/MedQuAD
2) TREC 2017 Live Medical Task QA: https://github.com/abachaa/LiveQA_MedicalTask_TREC2017
3) USMLE Question Bank: https://huggingface.co/datasets/bigbio/med_qa/blob/main/data_clean.zip
4) Gray's Anatomy textbook: https://huggingface.co/datasets/bigbio/med_qa/blob/main/data_clean.zip
<br/>
<b>Note:</b> If the user wishes to recreate the knowledge graph and try out the application locally, they should setup the knowledge graph by:

- Cloning the Repo
- Setting the GOOGLE_API_KEY, Neo4j_URL and Neo4j_Pass variables in their .env file
- Executing LLM2.py, Main.py, Trec_QA_converter.py (optional) and US_qbank_converter.py (optional)

Finally, the Streamlit application can be run locally using:
 python -m streamlit run ./app.py
