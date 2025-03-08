from neo4j import GraphDatabase
import json
import os
from dotenv import loadenv

URI = os.getenv("neo4j_URL")
USERNAME = "neo4j"
PASSWORD = os.getenv("neo4j_Pass")

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

def create_nodes(tx, data):
    # Extract fields from JSONL data
    question_text = data["question"]
    meta_info = data["meta_info"]
    answer_letter = data["answer"]
    options = data["options"]

    # Create Question and MetaInfo nodes
    tx.run(
        """
        MERGE (q:Question {text: $question})
        MERGE (m:MetaInfo {name: $meta_info})
        MERGE (q)-[:TAGGED_AS]->(m)
        """,
        question=question_text,
        meta_info=meta_info
    )

    # Create Options and link to Question
    for letter, text in options.items():
        is_correct = (letter == answer_letter)
        tx.run(
            """
            MATCH (q:Question {text: $question})
            MERGE (o:Option {letter: $letter, text: $text})
            MERGE (q)-[:HAS_OPTION {correct: $is_correct}]->(o)
            """,
            question=question_text,
            letter=letter,
            text=text,
            is_correct=is_correct
        )

def process_jsonl(file_path):
    qNo = 0
    with driver.session() as session:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                print(f"Q.No {qNo} encountered")
                data = json.loads(line)
                session.execute_write(create_nodes, data)  # Fixed: Call create_nodes
                print(f"Q.No {qNo} added")
                qNo += 1

process_jsonl("US_qbank.jsonl")
driver.close()