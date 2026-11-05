from rag import load_index, ask

QUESTIONS = [
    "Who was the commander of Apollo 11?",
    "What is the Hubble Space Telescope's mirror diameter?",
    "When was Sputnik 1 launched?",
    "Who won the 2022 FIFA World Cup?",
    "Which spacecraft has travelled the farthest and which one carried humans to the Moon?",
]

vectors, chunks = load_index()
rows = []
for i, q in enumerate(QUESTIONS, 1):
    retrieved, answer = ask(q, vectors, chunks, show=False)
    srcs = "; ".join(f"{c['source']}#{c['chunk_id']}" for c, _ in retrieved)
    answer = answer.replace("\n", " ").replace("|", "/")
    rows.append(f"| {i} | {q} | {srcs} | {answer} | TODO: your assessment |")

header = ("| # | Question | Retrieved chunks (sources) | System answer | Your assessment |\n"
          "| --- | --- | --- | --- | --- |\n")
open("results_table.md", "w", encoding="utf-8").write(header + "\n".join(rows))
print(header + "\n".join(rows))
