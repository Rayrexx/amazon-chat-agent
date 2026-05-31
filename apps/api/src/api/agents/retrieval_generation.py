import openai
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langsmith import traceable, get_current_run_tree
from pydantic import BaseModel, Field
import instructor
import numpy as np


class RAGUsedContext(BaseModel):
    id: str = Field(description="ID of the item used to answer the question")
    description: str = Field(
        description="Short Description of the item used to answer the question")


class RAGModelResponse(BaseModel):
    answer: str = Field(description="Answer to the question")
    references: list[RAGUsedContext] = Field(
        description="List of items used to answer the question")


@traceable(
    name="embed_query",
    run_type="embedding",
    metadata={"ls_provider": "openai",
              "ls_model_name": "text-embedding-3-small"}
)
def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        input=text,
        model=model,
    )

    current_run = get_current_run_tree()
    try:
        if current_run:
            current_run.metadata["usage_metadata"] = {
                "input_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens
            }
    except Exception as e:
        print("LangSmith metadata error:", e)
    return response.data[0].embedding


@traceable(
    name="retrieve_data",
    run_type="retriever",
)
def retrieve_data(query, qdrant_client, top_k=5):
    query_embedding = get_embedding(query)
    search_result = qdrant_client.query_points(
        collection_name="Amazon-items-collection-00",
        query=query_embedding,
        limit=top_k,
    )

    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []
    retrieved_context_ratings = []
    for point in search_result.points:
        retrieved_context_ids.append(point.payload["parent_asin"])
        retrieved_context.append(point.payload["description"])
        similarity_scores.append(point.score)
        retrieved_context_ratings.append(point.payload["average_rating"])
    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores,
        "retrieved_context_ratings": retrieved_context_ratings
    }


@traceable(
    name="format_retrieved_context",
    run_type="prompt"
)
def process_context(context):
    formatted_context = ""
    for id, chunk, rating in zip(context["retrieved_context_ids"], context["retrieved_context"], context["retrieved_context_ratings"]):
        formatted_context += f"-ID: {id}, Rating: {rating}, Description: {chunk}\n"

    return formatted_context


@traceable(
    name="build_prompt",
    run_type="prompt"
)
def build_prompt(preprocessed_context, question):
    prompt = f"""
you are a shopping assistant that can answer questions about products in stock.

You will be given a question and a list of context. 

Instructions:
- You need to answer the question based on the provided context only.
- Never use word context and refer to it as avaialable products.
- As an output you need to provide

* answer of the question based on provided context.
* list of the IDs of the chunks used to answer the question. Only return the ones that are used in the answer.
* short description (1-2 sentences) of the items based on the description provided in the context.

- the short description should have the name of the item.
- the answer to the question should contain detailed information about the products and returned with detailed specifications of the products in bullet points.

Context:
{preprocessed_context}

Question: {question}
"""
    return prompt


@traceable(
    name="generate_answer",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1-mini"}

)
def generate_answer(prompt):
    client = instructor.from_openai(openai.OpenAI())
    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": prompt},
        ],
        # reasoning_effort="minimal"
        temperature=0,
        response_model=RAGModelResponse
        # max_tokens=500,
    )

    current_run = get_current_run_tree()
    try:
        if current_run:
            current_run.metadata["usage_metadata"] = {
                "input_tokens": raw_response.usage.prompt_tokens,
                "output_tokens": raw_response.usage.completion_tokens,
                "total_tokens": raw_response.usage.total_tokens
            }

    except Exception as e:
        print("LangSmith metadata error:", e)

    return response


@traceable(
    name="rag_pipeline",
)
def rag_pipeline(question, qdrant_client, top_k=5):

    retrieved_context = retrieve_data(question, qdrant_client, top_k)
    preprocessed_context = process_context(retrieved_context)
    prompt = build_prompt(preprocessed_context, question)
    answer = generate_answer(prompt)
    final_result = {
        "answer": answer.answer,
        "references": answer.references,
        "question": question,
        "retrieved_context": retrieved_context["retrieved_context"],
        "retrieved_context_ids": retrieved_context["retrieved_context_ids"],
        "similarity_scores": retrieved_context["similarity_scores"],
    }

    return final_result


def rag_pipeline_wrapper(question, top_k=5):
    try:
        qdrant_client = QdrantClient(url="http://qdrant:6333")
        result = rag_pipeline(question, qdrant_client, top_k)
        used_context = []
        dummy_vector = np.zeros(1536).tolist()
        for ref in result.get("references", []):
            payload = qdrant_client.query_points(
                collection_name="Amazon-items-collection-00",
                query=dummy_vector,
                limit=1,
                with_payload=True,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="parent_asin",
                            match=MatchValue(value=ref.id)
                        )
                    ]
                )
            ).points[0].payload
            image_url = payload.get("image", "")
            price = payload.get("price", "")
            if image_url:
                used_context.append({
                    "image_url": image_url,
                    "price": price,
                    "description": ref.description,
                })
        return {
            "answer": result["answer"],
            "used_context": used_context
        }

    except Exception as e:
        print(f"Error in RAG pipeline: {e}")
        return {"error": "An error occurred while processing the request."}
