# queries_and_answers.py

from q_and_a_setup import build_qa_chain

def run_queries():
    qa = build_qa_chain()

    queries = [
        "Who was Albert Einstein?",
        "What continents are there?",
        "What is the theory of relativity?",
        "Who discovered gravity?",
        "What is quantum mechanics?"
    ]

    for query in queries:
        res = qa.invoke(query)
        print("Q:", query)
        print("A:", res["result"])
        print("-" * 50)

if __name__ == "__main__":
    run_queries()