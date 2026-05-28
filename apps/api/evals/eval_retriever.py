from api.agents.retrieval_generation import rag_pipeline

from langsmith import Client
from qdrant_client import QdrantClient

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall, Faithfulness, ResponseRelevancy

import asyncio


class EventLoopPolicy(asyncio.DefaultEventLoopPolicy):
    def get_event_loop(self):
        try:
            return super().get_event_loop()
        except RuntimeError:
            loop = self.new_event_loop()
            self.set_event_loop(loop)
            return loop


asyncio.set_event_loop_policy(EventLoopPolicy())

ls_client = Client()
qdrant_client = QdrantClient(url=f"http://localhost:6333")

ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini"))
ragas_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(model="text-embedding-3-small"))

faithfulness_scorer = Faithfulness(llm=ragas_llm)
response_relevancy_scorer = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
context_recall_scorer = IDBasedContextRecall()
precision_scorer = IDBasedContextPrecision()


async def ragas_faithfulness(run, example):
    sample = SingleTurnSample(
        user_input=run.inputs["question"],
        response=run.outputs["answer"],
        retrieved_contexts=run.outputs["retrieved_context"],
    )
    # scorer = Faithfulness(llm=ragas_llm)

    return await faithfulness_scorer.single_turn_ascore(sample)


async def ragas_response_relevanacy(run, example):
    sample = SingleTurnSample(
        user_input=run.inputs["question"],
        response=run.outputs["answer"],
        retrieved_contexts=run.outputs["retrieved_context"],
    )
    # scorer = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)

    return await response_relevancy_scorer.single_turn_ascore(sample)


async def ragas_context_precision_id_based(run, example):
    sample = SingleTurnSample(
        retrieved_context_ids=run.outputs["retrieved_context_ids"],
        reference_context_ids=example.outputs["reference_context_ids"],
    )
    # scorer = IDBasedContextPrecision()
    return await precision_scorer.single_turn_ascore(sample)


async def ragas_context_recall_id_based(run, example):
    # print(type(run))
    # print(run)
    # print(type(example))
    # print(example)
    sample = SingleTurnSample(
        retrieved_context_ids=run.outputs["retrieved_context_ids"],
        reference_context_ids=example.outputs["reference_context_ids"],
    )
    # scorer = IDBasedContextRecall()
    return await context_recall_scorer.single_turn_ascore(sample)


results = ls_client.evaluate(
    lambda inputs: rag_pipeline(question=inputs["question"],qdrant_client=qdrant_client),
    data="rag-evaluation-dataset",
    evaluators=[
        ragas_faithfulness,
        ragas_response_relevanacy,
        ragas_context_precision_id_based,
        ragas_context_recall_id_based
    ],
    experiment_prefix="retriever",
    max_concurrency=4
)
