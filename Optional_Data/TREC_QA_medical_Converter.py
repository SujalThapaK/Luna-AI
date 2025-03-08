import xml.etree.ElementTree as ET
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

URI = os.getenv("neo4j_URL")
USERNAME = "neo4j"
PASSWORD = os.getenv("neo4j_Pass")

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))


def create_question(tx, qid, subject, message, paraphrase, summary):
    query = (
        "MERGE (q:Question {qid: $qid}) "
        "SET q.subject = $subject, "
        "    q.message = $message, "
        "    q.nist_paraphrase = $paraphrase, "
        "    q.summary = $summary "
        "RETURN q"
    )
    tx.run(query, qid=qid, subject=subject, message=message,
           paraphrase=paraphrase, summary=summary)


def create_focus(tx, fid, category, text):
    query = (
        "MERGE (f:Focus {fid: $fid}) "
        "SET f.category = $category, f.text = $text "
        "RETURN f"
    )
    tx.run(query, fid=fid, category=category, text=text)


def link_question_focus(tx, qid, fid):
    query = (
        "MATCH (q:Question {qid: $qid}), (f:Focus {fid: $fid}) "
        "MERGE (q)-[:HAS_FOCUS]->(f)"
    )
    tx.run(query, qid=qid, fid=fid)


def create_type(tx, tid, text, has_focus):
    query = (
        "MERGE (t:Type {tid: $tid}) "
        "SET t.text = $text, t.has_focus = $has_focus "
        "RETURN t"
    )
    tx.run(query, tid=tid, text=text, has_focus=has_focus)


def link_question_type(tx, qid, tid):
    query = (
        "MATCH (q:Question {qid: $qid}), (t:Type {tid: $tid}) "
        "MERGE (q)-[:HAS_TYPE]->(t)"
    )
    tx.run(query, qid=qid, tid=tid)


def create_keyword(tx, qid, kid, category, text):
    keyword_unique_key = f"{qid}_{kid}"

    # Create/update shared Kid node (identified by kid alone)
    tx.run("MERGE (k:Kid {kid: $kid})", kid=kid)

    # Create unique Keyword node per question
    tx.run(
        "MERGE (kw:Keyword {unique_key: $unique_key}) "
        "SET kw.category = $category, kw.text = $text",
        unique_key=keyword_unique_key, category=category, text=text
    )

    # Link Keyword to shared Kid
    tx.run(
        "MATCH (kw:Keyword {unique_key: $unique_key}), (k:Kid {kid: $kid}) "
        "MERGE (kw)-[:HAS_KID]->(k)",
        unique_key=keyword_unique_key, kid=kid
    )


def link_question_keyword(tx, qid, kid):
    unique_key = f"{qid}_{kid}"
    tx.run(
        "MATCH (q:Question {qid: $qid}), (kw:Keyword {unique_key: $unique_key}) "
        "MERGE (q)-[:HAS_KEYWORD]->(kw)",
        qid=qid, unique_key=unique_key
    )


def create_answer(tx, aid, answer_text, url, comment):
    query = (
        "MERGE (a:Answer {aid: $aid}) "
        "SET a.answer = $answer_text, a.url = $url, a.comment = $comment "
        "RETURN a"
    )
    tx.run(query, aid=aid, answer_text=answer_text, url=url, comment=comment)


def link_question_answer(tx, qid, aid):
    query = (
        "MATCH (q:Question {qid: $qid}), (a:Answer {aid: $aid}) "
        "MERGE (q)-[:HAS_ANSWER]->(a)"
    )
    tx.run(query, qid=qid, aid=aid)


def import_xml_data(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    qno = 0
    with driver.session() as session:
        for question in root.findall("NLM-QUESTION"):
            print(f"Dealing with Question: {qno}")
            qid = question.get("qid")
            original = question.find("Original-Question")

            # Extract question properties
            subject = original.find("SUBJECT").text.strip() if original.find("SUBJECT") is not None and original.find(
                "SUBJECT").text else ""
            message = original.find("MESSAGE").text.strip() if original.find("MESSAGE") is not None and original.find(
                "MESSAGE").text else ""
            paraphrase = question.find("NIST-PARAPHRASE").text.strip() if question.find(
                "NIST-PARAPHRASE") is not None and question.find("NIST-PARAPHRASE").text else ""
            summary = question.find("NLM-Summary").text.strip() if question.find(
                "NLM-Summary") is not None and question.find("NLM-Summary").text else ""

            session.execute_write(create_question, qid, subject, message, paraphrase, summary)

            # Process annotations
            if (annotations := question.find("ANNOTATIONS")) is not None:
                for focus in annotations.findall("FOCUS"):
                    fid = focus.get("fid")
                    session.execute_write(create_focus, fid, focus.get("fcategory"),
                                          focus.text.strip() if focus.text else "")
                    session.execute_write(link_question_focus, qid, fid)

                for typ in annotations.findall("TYPE"):
                    tid = typ.get("tid")
                    session.execute_write(create_type, tid, typ.text.strip() if typ.text else "",
                                          typ.get("hasFocus", ""))
                    session.execute_write(link_question_type, qid, tid)

                for keyword in annotations.findall("KEYWORD"):
                    kid = keyword.get("kid")
                    session.execute_write(create_keyword, qid, kid, keyword.get("kcategory"),
                                          keyword.text.strip() if keyword.text else "")
                    session.execute_write(link_question_keyword, qid, kid)

            # Process answers
            if (ref_answers := question.find("ReferenceAnswers")) is not None:
                for ans in ref_answers.findall("RefAnswer") or ref_answers.findall("ReferenceAnswer"):
                    aid = ans.get("aid")
                    answer_text = ans.findtext("ANSWER", default="").strip()
                    answer_url = ans.findtext("AnswerURL", default="").strip()
                    comment = ans.findtext("COMMENT", default="").strip()
                    session.execute_write(create_answer, aid, answer_text, answer_url, comment)
                    session.execute_write(link_question_answer, qid, aid)

            qno += 1
            print(f"Added Question: {qno}")

    print("XML data successfully imported into Neo4j!")


if __name__ == "__main__":
    import_xml_data("./TREC-2017-LiveQA-Medical-Test-Questions-w-summaries.xml")
    driver.close()