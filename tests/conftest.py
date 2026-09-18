"""Shared fixtures. Every test runs against a real ChromaDB and a real database."""

from __future__ import annotations
import shutil
import tempfile
from pathlib import Path

import pytest

from src.config import settings
from src.database.models import Base
from src.rag.embedder import TfidfSvdEmbedder, content_hash
from src.rag.vector_store import VectorStore
from src.rag.pipeline import RAGPipeline, chunk_text
from src.database import repository as repo


DOCS = [
    ("refund_policy", "Refund Policy", "policy", 5,
     "Enterprise customers may request a full refund within thirty days of the "
     "invoice date. Refund requests require written approval from the finance "
     "team. Partial refunds are calculated pro rata from the cancellation date."),
    ("sso_guide", "Single Sign On Guide", "integration", 45,
     "To enable single sign on navigate to settings then security then SSO. "
     "Select Azure Active Directory as the identity provider. You will need the "
     "tenant identifier and a client secret generated in the Azure portal."),
    ("rate_limits", "API Rate Limits", "api", 60,
     "Standard tier is limited to one hundred requests per minute. Enterprise "
     "tier is limited to one thousand requests per minute with burst capacity "
     "to two thousand requests for short periods."),
    ("webhooks", "Webhook Setup", "integration", 12,
     "Create a webhook endpoint under settings integrations webhooks. Provide a "
     "publicly reachable HTTPS URL. Every delivery includes a signature header "
     "computed as an HMAC SHA256 of the request body."),
]


@pytest.fixture(scope="session")
def tmp_root():
    d = Path(tempfile.mkdtemp(prefix="handoff_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="session")
def engine(tmp_root):
    from sqlalchemy import create_engine
    eng = create_engine(f"sqlite:///{tmp_root/'test.db'}",
                        connect_args={"check_same_thread": False}, future=True)
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def session(engine):
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    s = Session()
    try:
        yield s
        s.commit()
    finally:
        s.rollback()
        s.close()


@pytest.fixture(scope="session")
def embedder():
    e = TfidfSvdEmbedder(dim=16)
    corpus = []
    for _, _, _, _, content in DOCS:
        corpus.extend(chunk_text(content))
    e.fit(corpus)
    return e


@pytest.fixture
def store(embedder, tmp_root):
    import uuid
    s = VectorStore(embedder, collection_name=f"t_{uuid.uuid4().hex[:8]}",
                    persist_dir=str(tmp_root / "chroma"))
    yield s
    try:
        s.reset()
    except Exception:
        pass


@pytest.fixture
def pipeline(embedder, store):
    return RAGPipeline(embedder, store)


@pytest.fixture
def seeded(pipeline, session):
    """A pipeline with four documents indexed at varied embedding ages."""
    for key, title, cat, age, content in DOCS:
        pipeline.index_document(session, doc_key=key, title=title,
                                content=content, category=cat,
                                embedded_days_ago=age)
    return pipeline


@pytest.fixture
def drifted(seeded, session):
    """Same corpus, with rate_limits given content drift."""
    doc = repo.get_document(session, "rate_limits")
    new = doc.content + " UPDATE enterprise tier is now limited to five thousand requests per minute."
    repo.touch_document(session, "rate_limits", new, content_hash(new))
    return seeded
