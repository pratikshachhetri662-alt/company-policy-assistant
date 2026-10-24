import time
import chromadb


import streamlit as st
import pandas as pd
from groq import Groq
client = Groq(api_key=st.secrets["GROQ_API_KEY"])
st.set_page_config(page_title="Company Policy Assistant", layout="wide")

st.title("Company Policy Assistant")
st.write("Ask questions about company policies.")

policies = pd.read_csv("company_policies.csv")

st.success("Company policy database loaded successfully!")
st.write("Number of policies:", len(policies))

st.subheader("Available Company Policies")
st.dataframe(policies)
st.divider()
st.header("Method 1: Rules-Based Search")

question = st.text_input("Ask a company policy question:")

if question:
    start_time = time.perf_counter()
    words = [
        word.strip("?!.,").lower()
        for word in question.split()
        if len(word.strip("?!.,")) > 3
    ]

    matches = policies[
        policies["title"].fillna("").str.lower().apply(
            lambda text: any(word in text for word in words)
        )
        |
        policies["policy_text"].fillna("").str.lower().apply(
            lambda text: any(word in text for word in words)
        )
    ]

    if not matches.empty:
        st.success("Matching policies found!")
        st.dataframe(matches[["title", "department", "policy_text"]])
    else:
        st.warning("No matching policy found.")

    elapsed_time = time.perf_counter() - start_time
    st.caption(f"Response time: {elapsed_time:.4f} seconds")
    st.session_state["method1_result"] = {
        "answer": "No matching policy found." if matches.empty else "Matching policies found.",
        "policy": "None" if matches.empty else ", ".join(matches["title"].astype(str).tolist()),
        "time": f"{elapsed_time:.4f} seconds",
        "tokens": "N/A",
    }

st.divider()
st.header("Method 2: LLM Without Vector Index")

ai_question = st.text_input(
    "Ask the AI a company policy question:",
    key="ai_question"
)

if ai_question:
    start_time_ai = time.perf_counter()
    with st.spinner("AI is reading the company policies..."):
        policy_context = "\n\n".join(
            f"Policy: {row['title']}\n"
            f"Department: {row['department']}\n"
            f"Text: {row['policy_text']}"
            for _, row in policies.iterrows()
        )

        response = client.chat.completions.create(
           model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer using only the company policies below. "
                        "Identify the relevant policy by title. "
                        "Do not use outside knowledge or make assumptions. If the policies do not support an answer, "
                        "say you cannot find the answer.\n\n"
                        + policy_context
                    ),
                },
                {"role": "user", "content": ai_question},
            ],
        )
    if response:
         st.write(response.choices[0].message.content)
         st.caption(f"Tokens used: {response.usage.total_tokens}")
         elapsed_time_ai = time.perf_counter() - start_time_ai
         st.caption(f"Response time: {elapsed_time_ai:.4f} seconds")
         st.session_state["method2_result"] = {
         "answer": response.choices[0].message.content,
         "policy": "Identified in AI answer",
         "time": f"{elapsed_time_ai:.4f} seconds",
         "tokens": response.usage.total_tokens,
         }

st.divider()
st.header("Method 3: LLM With Vector Index")

vector_question = st.text_input(
    "Ask a question using vector search:",
    key="vector_question"
)

if vector_question:
    start_time_vector = time.perf_counter()
    with st.spinner("Searching relevant policies and generating an answer..."):
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection("company_policies")

        results = collection.query(
            query_texts=[vector_question],
            n_results=1
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

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "The retrieved policy is provided below. Answer the user by stating what that policy requires. Include its title. Do not claim the policy is missing when its text is provided. Do not invent additional requirements. "
                        "Identify the relevant policy by title. "
                        "Do not use outside knowledge or make assumptions. If the policies do not support an answer, "
                        "say you cannot find the answer.\n\n"
                        + context
                    ),
                },
                {"role": "user", "content": vector_question},
            ],
        )

        st.write(response.choices[0].message.content)
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
       st.caption(f"Tokens used: {response.usage.total_tokens}")
        elapsed_time_vector = time.perf_counter() - start_time_vector
        st.caption(f"Response time: {elapsed_time_vector:.4f} seconds")
        st.caption("Relevant policies retrieved from the vector index:")
        for metadata in results["metadatas"][0]:
            st.write("•", metadata["title"])

        st.session_state["method3_result"] = {
            "policy": ", ".join(
                metadata["title"] for metadata in results["metadatas"][0]
            ),
            "time": f"{elapsed_time_vector:.4f} seconds",
            "tokens": input_tokens + output_tokens,
        }

st.divider()
st.header("Comparison of All Three Methods")

comparison_data = pd.DataFrame([
    {"Method": "1. Rules-Based Search",
     "Answer": st.session_state.get("method1_result", {}).get("answer", "Not tested"),
     "Relevant Policy": st.session_state.get("method1_result", {}).get("policy", "Not tested"),
     "Response Time": st.session_state.get("method1_result", {}).get("time", "Not tested"),
     "Tokens Used": "N/A",
     "Unsupported Answers": "Returned unrelated policy in test"},
    {"Method": "2. LLM Without Vector Index",
     "Answer": st.session_state.get("method2_result", {}).get("answer", "Not tested"),
     "Relevant Policy": st.session_state.get("method2_result", {}).get("policy", "Not tested"),
     "Response Time": st.session_state.get("method2_result", {}).get("time", "Not tested"),
     "Tokens Used": st.session_state.get("method2_result", {}).get("tokens", "Not tested"),
     "Unsupported Answers": "Added unsupported suggestions in test"},
    {"Method": "3. LLM With Vector Index",
     "Answer": st.session_state.get("method3_result", {}).get("answer", "Not tested"),
     "Relevant Policy": st.session_state.get("method3_result", {}).get("policy", "Not tested"),
     "Response Time": st.session_state.get("method3_result", {}).get("time", "Not tested"),
     "Tokens Used": st.session_state.get("method3_result", {}).get("tokens", "Not tested"),
     "Unsupported Answers": "No unsupported details in test"},
])

st.dataframe(comparison_data, use_container_width=True)

st.subheader("Analysis and Recommended Approach")

st.write(
    "The three methods showed different strengths and weaknesses. "
    "Rules-based search was fastest in the dress code test (0.0246 seconds) "
    "and used no AI tokens, but returned several unrelated policies. "
    "The LLM without a vector index answered correctly in 5.0338 seconds "
    "using 2,205 tokens. The LLM with a vector index identified the "
    "Dress Code Policy and answered correctly in 3.8522 seconds using "
    "139 tokens. These are individual test results, not averages."
)

st.write(
    "Based on these tests, the LLM with a vector index is the preferred "
    "approach because it retrieves relevant policies and uses fewer tokens. "
    "For the unsupported CEO phone-number question, rules-based search "
    "returned an unrelated Sick Leave Policy, while the LLM without a "
    "vector index added unsupported contact suggestions. The vector-index "
    "approach did not invent a phone number or add unsupported details, "
    "although it retrieved an unrelated policy. More testing and safeguards "
    "would be needed before using it in a real company."
