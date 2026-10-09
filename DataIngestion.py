text = """LangChain is a framework for building AI apps.
It helps connect language models with your own data.
RAG means Retrieval-Augmented Generation."""

with open("sample.txt", "w") as f:
    f.write(text)

print("sample.txt file created!")

