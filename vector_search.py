import chromadb
import pandas as pd
import chromadb
policies = pd.read_csv("company_policies.csv")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="company_policies"
)

for index, row in policies.iterrows():
    collection.upsert(
        ids=[str(index)],
        documents=[str(row["policy_text"])],
        metadatas=[{
            "title": str(row["title"]),
            "department": str(row["department"])
        }]
    )

print("Vector database created successfully!")
print("Number of policies:", collection.count())st.divider()
st.header("Method 3: LLM With Vector Index")

vector_question = st.text_input(
    "Ask a question using vector search:",
    key="vector_question"
)

if vector_question:
    with st.spinner("Searching relevant policies and generating an answer..."):
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection("company_policies")

        results = collection.query(
            query_texts=[vector_question],
            n_results=3
        )

        context = "\n\n".join(
            f"Policy: {metadata['title']}\n"
            f"Department: {metadata['department']}\n"
            f"Text: {document}"
            for document, metadata in zip(
                results["documents"][0],
                results["metadatas"][0]
            )
        )

        response = ollama.chat(
            model="llama3.2:1b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer only using the policies below. "
                        "Identify the relevant policy by title. "
                        "If the policies do not support an answer, "
                        "say you cannot find the answer.\n\n"
                        + context
                    ),
                },
                {"role": "user", "content": vector_question},
            ],
        )

        st.write(response["message"]["content"])
        st.caption("Relevant policies retrieved from the vector index:")
        for metadata in results["metadatas"][0]:
            st.write("•", metadata["title"])