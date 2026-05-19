from api.agents.retrieval_generation import rag_pipeline

from langsmith import Client
from qdrant_client import QdrantClient

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall, Faithfulness, ResponseRelevancy

ls_client = Client()
qdrant_client = QdrantClient(url=f"http://localhost:6333")

ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini"))
ragas_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(model="text-embedding-3-small"))


async def ragas_faithfulness(run, example):
    sample = SingleTurnSample(
        user_input=run["question"],
        response=run["answer"],
        retrieved_contexts=run["retrieved_context"],
    )
    scorer = Faithfulness(llm=ragas_llm)

    return await scorer.single_turn_ascore(sample)


async def ragas_response_relevanacy(run, example):
    sample = SingleTurnSample(
        user_input=run["question"],
        response=run["answer"],
        retrieved_contexts=run["retrieved_context"],
    )
    scorer = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)

    return await scorer.single_turn_ascore(sample)


async def ragas_context_precision_id_based(run, example):
    sample = SingleTurnSample(
        retrieved_context_ids=run["retrieved_context_ids"],
        reference_context_ids=example["reference_context_ids"],
    )
    scorer = IDBasedContextPrecision()
    return await scorer.single_turn_ascore(sample)


async def ragas_context_recall_id_based(run, example):
    sample = SingleTurnSample(
        retrieved_context_ids=run["retrieved_context_ids"],
        reference_context_ids=example["reference_context_ids"],
    )
    scorer = IDBasedContextRecall()
    return await scorer.single_turn_ascore(sample)


results = ls_client.evaluate(
    lambda x: rag_pipeline(x["question"], qdrant_client),
    data="rag-evaluation-dataset",
    evaluators=[
        ragas_faithfulness,
        ragas_response_relevanacy,
        ragas_context_precision_id_based,
        ragas_context_recall_id_based
    ],
    experiment_prefix="retriever",
    max_concurrency=10
)
