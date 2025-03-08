from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

# URI examples: "neo4j://localhost", "neo4j+s://xxx.databases.neo4j.io"
URI = os.getenv("neo4j_URL")
AUTH = ("neo4j", os.getenv("neo4j_Pass"))

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()