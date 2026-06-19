from transformers import pipeline
from retriever import retrieve
from web_search import search_web
import hashlib


generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device="cpu"
)


def is_similar(text1, text2):
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    overlap = len(words1 & words2)

    similarity = overlap / max(
        len(words1),
        len(words2)
    )

    return similarity > 0.8


def generate_answer(query):

    # Retrieve documents
    local_docs = retrieve(query)

    # Retrieve web results
    web_docs = search_web(query)

    # Remove exact duplicates
    unique_chunks = []
    seen_hashes = set()

    for doc in local_docs:

        text = doc["text"].strip()

        chunk_hash = hashlib.md5(
            text.encode("utf-8")
        ).hexdigest()

        if chunk_hash not in seen_hashes:

            unique_chunks.append(doc)

            seen_hashes.add(chunk_hash)

    # Remove near duplicates
    filtered_chunks = []

    for doc in unique_chunks:

        duplicate = False

        for existing in filtered_chunks:

            if is_similar(
                doc["text"],
                existing["text"]
            ):
                duplicate = True
                break

        if not duplicate:
            filtered_chunks.append(doc)

    # Build context
    context_parts = []
    sources = []

    for doc in filtered_chunks[:3]:

        context = "\n\n".join(context_parts)

        if len(context) > 4000:
            context = context[:4000]

            sources.append(
                doc["source"]
            )

    for web in web_docs[:2]:

        context_parts.append(
            web["body"]
        )

        sources.append(
            web["title"]
        )

    context = "\n\n".join(
        context_parts
    )

    print("\n===== CONTEXT =====")
    print(context[:2000])
    print("===================\n")

    prompt = f"""
You are an expert professor.

Using the context provided, generate a complete and detailed answer.

Requirements:

1. Give a clear definition.
2. Explain the concept in detail.
3. Explain how it works.
4. Mention important algorithms if applicable.
5. Provide examples.
6. Mention real-world applications.
7. Mention advantages.
8. Mention disadvantages.
9. Use complete sentences.
10. Do not return incomplete or broken sentences.
11. If the context contains partial information, infer and complete the explanation.

Question: {query}

Using the context, generate a structured response.

Format:

Definition:
...

Detailed Explanation:
...

How It Works:
...

Examples:
...

Applications:
...

Advantages:
...

Disadvantages:
...


Context:
{context}

Question:
{query}

Detailed Answer:
"""

    response = generator(
        prompt,
        max_length=700,
        do_sample=False
    )

    answer = response[0]["generated_text"]

    return {
        "answer": answer,
        "sources": list(set(sources))
    }