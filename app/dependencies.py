from fastapi import Request

from app.cache.embedding_cache import EmbeddingCache
from app.cache.response_cache import ResponseCache
from app.chunkers.recursive_splitter import RecursiveSplitter
from app.core.config import Settings, get_settings
from app.embedders.openai_embedder import OpenAIEmbedder
from app.llm_clients.openai_chat_client import OpenAIChatClient
from app.pipelines.ingest_pipeline import IngestPipeline
from app.pipelines.query_pipeline import QueryPipeline
from app.retrievers.bm25_retriever import BM25Retriever
from app.retrievers.hybrid_retriever import HybridRetriever
from app.retrievers.reranker import LLMReranker
from app.retrievers.vector_retriever import VectorRetriever
from app.services.ask_service import AskService
from app.services.document_service import DocumentService
from app.storages.database import Database
from app.storages.faiss_store import FaissStore
from app.storages.redis_store import RedisStore
from app.validators.intent_classifier import IntentClassifier


class Container:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.database = Database(settings)
        self.redis_store = RedisStore(settings)
        self.faiss_store = FaissStore(settings)

        self.embedding_cache = EmbeddingCache(self.redis_store, settings)
        self.response_cache = ResponseCache(self.redis_store, settings)

        self.chat_client = OpenAIChatClient(settings)
        self.embedder = OpenAIEmbedder(settings, self.embedding_cache)

        self.splitter = RecursiveSplitter(settings)
        self.vector_retriever = VectorRetriever(
            embedder=self.embedder,
            vector_store=self.faiss_store,
            settings=settings,
        )
        self.bm25_retriever = BM25Retriever(
            faiss_store=self.faiss_store,
            settings=settings,
        )
        self.hybrid_retriever = HybridRetriever(
            vector_retriever=self.vector_retriever,
            bm25_retriever=self.bm25_retriever,
            settings=settings,
        )
        self.reranker = LLMReranker(
            chat_client=self.chat_client,
            settings=settings,
        )
        self.intent_classifier = IntentClassifier(
            embedder=self.embedder,
            settings=settings,
        )

        self.ingest_pipeline = IngestPipeline(
            splitter=self.splitter,
            embedder=self.embedder,
            vector_store=self.faiss_store,
            settings=settings,
        )
        self.query_pipeline = QueryPipeline(
            hybrid_retriever=self.hybrid_retriever,
            reranker=self.reranker,
            chat_client=self.chat_client,
            settings=settings,
        )

        self.document_service = DocumentService(
            database=self.database,
            vector_store=self.faiss_store,
            ingest_pipeline=self.ingest_pipeline,
            settings=settings,
        )
        self.ask_service = AskService(
            database=self.database,
            query_pipeline=self.query_pipeline,
            response_cache=self.response_cache,
            embedder=self.embedder,
            intent_classifier=self.intent_classifier,
            settings=settings,
        )

    async def shutdown(self) -> None:
        await self.redis_store.close()
        await self.database.dispose()


def build_container() -> Container:
    return Container(get_settings())


def _container(request: Request) -> Container:
    return request.app.state.container


def get_settings_dep(request: Request) -> Settings:
    return _container(request).settings


def get_database(request: Request) -> Database:
    return _container(request).database


def get_redis_store(request: Request) -> RedisStore:
    return _container(request).redis_store


def get_faiss_store(request: Request) -> FaissStore:
    return _container(request).faiss_store


def get_ask_service(request: Request) -> AskService:
    return _container(request).ask_service


def get_document_service(request: Request) -> DocumentService:
    return _container(request).document_service
