import openai
from qdrant_client import QdrantClient


def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        input=text,
        model=model,
    )
    return response.data[0].embedding


def retrieve_data(query, qdrant_client, top_k=5):
    query_embedding = get_embedding(query)
    search_result = qdrant_client.query_points(
        collection_name="Amazon-items-collection-00",
        query=query_embedding,
        limit=top_k,
    )

    retrieved_context_ids = []
    retreived_context = []
    similarity_scores = []
    retreived_context_ratings = []
    for point in search_result.points:
        retrieved_context_ids.append(point.payload["parent_asin"])
        retreived_context.append(point.payload["description"])
        similarity_scores.append(point.score)
        retreived_context_ratings.append(point.payload["average_rating"])
    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retreived_context": retreived_context,
        "similarity_scores": similarity_scores,
        "retreived_context_ratings": retreived_context_ratings
    }


def process_context(context):
    formatted_context = ""
    for id, chunk, rating in zip(context["retrieved_context_ids"], context["retreived_context"], context["retreived_context_ratings"]):
        formatted_context += f"-ID: {id}, Rating: {rating}, Description: {chunk}\n"

    return formatted_context


def build_prompt(preprocessed_context, question):
    prompt = f"""
you are a shopping assistant that can answer questions about products in stock.

You will be given a question and a list of context. 

Instructions:
- You need to answer the question based on the provided context only.
- Never use word context and refer to it as avaialable products.

Context:
{preprocessed_context}

Question: {question}
"""
    return prompt


def generate_answer(prompt):
    response = openai.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": prompt},
        ],
        reasoning_effort="minimal"
        # temperature=0.7,
        # max_tokens=500,
    )
    return response.choices[0].message.content


def rag_pipeline(question, top_k=5):
    qdrant_client = QdrantClient(
        url="http://qdrant:6333")
    retrieved_context = retrieve_data(question, qdrant_client, top_k)
    preprocessed_context = process_context(retrieved_context)
    prompt = build_prompt(preprocessed_context, question)
    answer = generate_answer(prompt)
    return answer
