"""
Seed the enterprise knowledge base with realistic documents.

Embedding ages are deliberately varied so the freshness monitor has a genuine
mix of fresh, stale, and critical content to score. Two documents are given
content drift: the source text is updated after embedding, so the vector store
holds text that no longer matches the source.

Usage:
    python scripts/seed_corpus.py
    python scripts/seed_corpus.py --reset
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.database.session import init_db, get_session
from src.database import repository as repo
from src.rag.embedder import build_embedder, content_hash
from src.rag.vector_store import VectorStore
from src.rag.pipeline import RAGPipeline, chunk_text


CORPUS = [
    {
        "doc_key": "refund_policy",
        "title": "Enterprise Refund and Cancellation Policy",
        "category": "policy",
        "embedded_days_ago": 6,
        "content": """
Enterprise Refund and Cancellation Policy.

Enterprise customers may request a full refund within thirty days of the initial
invoice date. Refund requests must be submitted through the account manager and
require written approval from the finance team. Partial refunds are calculated on
a pro rata basis from the cancellation date to the end of the billing period.

Annual contracts cancelled after the thirty day refund window are not eligible for
a refund of the remaining term. Customers may instead apply the unused balance as
credit toward a future renewal within twelve months.

Refunds are processed to the original payment method within fifteen business days
of approval. Wire transfer refunds may take up to twenty business days depending
on the receiving institution. Enterprise customers on net sixty payment terms who
have not yet paid the invoice will receive a revised invoice rather than a refund.

Usage based overage charges already incurred are not refundable. Professional
services and onboarding fees are non refundable once work has commenced.
""",
    },
    {
        "doc_key": "sso_azure_guide",
        "title": "Configuring Single Sign On with Azure Active Directory",
        "category": "integration",
        "embedded_days_ago": 48,
        "content": """
Configuring Single Sign On with Azure Active Directory.

To enable single sign on, navigate to Settings, then Security, then SSO. Select
Azure Active Directory as the identity provider. You will need your Azure tenant
identifier, the application client identifier, and a client secret generated in
the Azure portal.

In the Azure portal, register a new enterprise application. Set the reply URL to
the callback address shown on the SSO configuration page. Grant the application
the User.Read and Directory.Read.All delegated permissions, then grant admin
consent for your organization.

Copy the tenant identifier and client identifier into the SSO configuration form.
Generate a client secret with a twenty four month expiry and paste it into the
secret field. Save the configuration and use the Test Connection button before
enabling enforcement.

Once SSO enforcement is enabled, local password authentication is disabled for all
users except break glass administrator accounts. Configure at least two break glass
accounts before enforcing SSO to avoid lockout. Group claims can be mapped to
platform roles under the Role Mapping tab.
""",
    },
    {
        "doc_key": "data_export_api",
        "title": "Bulk Data Export API Reference",
        "category": "api",
        "embedded_days_ago": 11,
        "content": """
Bulk Data Export API Reference.

Bulk export is available through the export endpoint at version three of the API.
Authenticate with a bearer token in the Authorization header. The token must carry
the export scope, which is granted separately from read scope.

Submit an export job with a POST request specifying the dataset name, the output
format, and an optional date range filter. Supported formats are CSV, newline
delimited JSON, and Parquet. Parquet is recommended for datasets above one million
rows because it compresses significantly better.

Export jobs are asynchronous. The initial response returns a job identifier. Poll
the job status endpoint until the state field reads completed. Completed jobs
return a presigned download URL valid for twenty four hours.

Exports are limited to one concurrent job per organization on standard tier and
five concurrent jobs on enterprise tier. Jobs exceeding four hours of runtime are
terminated automatically. For datasets above fifty million rows, split the export
using date range filters.
""",
    },
    {
        "doc_key": "webhooks_setup",
        "title": "Webhook Notification Setup",
        "category": "integration",
        "embedded_days_ago": 19,
        "content": """
Webhook Notification Setup.

Create a webhook endpoint under Settings, then Integrations, then Webhooks. Provide
a publicly reachable HTTPS URL. Plain HTTP endpoints are rejected. The endpoint must
respond with a two hundred status code within five seconds.

Select the event types your endpoint should receive. Available events include record
created, record updated, record deleted, export completed, and user invited. Each
event type can be enabled independently per endpoint.

Every webhook delivery includes a signature header computed as an HMAC SHA256 of the
raw request body using your webhook signing secret. Verify this signature before
processing any payload. Rotate the signing secret from the webhook detail page.

Failed deliveries are retried with exponential backoff at one minute, five minutes,
thirty minutes, two hours, and twelve hours. After five failed attempts the endpoint
is marked unhealthy and deliveries are paused until manually resumed.
""",
    },
    {
        "doc_key": "data_retention",
        "title": "Data Retention and Deletion Policy",
        "category": "policy",
        "embedded_days_ago": 4,
        "content": """
Data Retention and Deletion Policy.

Customer records are retained for seven years from the date of creation to satisfy
enterprise compliance and audit requirements. This retention period applies to
transactional records, audit logs, and user activity history.

Deleted records enter a soft delete state for ninety days during which they can be
restored by an administrator. After ninety days, soft deleted records are purged
from primary storage. Backup copies are purged on the subsequent backup rotation,
which completes within an additional thirty five days.

Customers may request early deletion of specific records through a data subject
request. Verified requests are processed within thirty days as required by
applicable privacy regulation. Records under active legal hold are excluded from
early deletion and retained until the hold is released.

Audit logs are retained for seven years and cannot be deleted through the
application interface, as they form part of the compliance evidence chain.
""",
    },
    {
        "doc_key": "salesforce_integration",
        "title": "Salesforce CRM Integration Guide",
        "category": "integration",
        "embedded_days_ago": 71,
        "content": """
Salesforce CRM Integration Guide.

Install the connector from the integrations marketplace. You will need your
Salesforce organization identifier and a connected app configured with OAuth. The
connecting user must hold the API Enabled and Modify All Data permissions.

Field mapping is configured per object. Standard objects including Account,
Contact, Opportunity, and Lead are mapped automatically using default field
correspondences. Custom objects and custom fields require explicit mapping in the
Field Mapping tab.

Synchronization runs every fifteen minutes by default and can be adjusted to five
minute or hourly intervals. Bidirectional sync is available for Account and Contact
objects. Opportunity sync is one directional from Salesforce into the platform.

Conflict resolution follows a last write wins strategy based on the record
modification timestamp. Records modified in both systems within the same sync
window are flagged for manual review rather than resolved automatically.
""",
    },
    {
        "doc_key": "report_branding",
        "title": "Custom Report Branding",
        "category": "configuration",
        "embedded_days_ago": 16,
        "content": """
Custom Report Branding.

Upload your organization logo under Settings, then Branding. Accepted formats are
PNG and SVG. The recommended logo dimensions are four hundred pixels wide by one
hundred twenty pixels tall. Files above two megabytes are rejected.

Define a primary and secondary brand color using hexadecimal color codes. These
colors are applied to report headers, chart series defaults, and exported PDF
cover pages. A live preview updates as colors are selected.

Custom fonts can be applied by uploading a WOFF2 file. If no custom font is
uploaded, reports render in the default system font stack. Font files must be
licensed for embedding.

Branding settings apply to all generated reports, scheduled email digests, and the
customer facing portal. Individual reports can override branding through the report
settings panel if per report branding is enabled for your plan.
""",
    },
    {
        "doc_key": "api_tokens",
        "title": "API Token Management",
        "category": "api",
        "embedded_days_ago": 24,
        "content": """
API Token Management.

Create and manage API tokens under Settings, then API, then Tokens. Each token is
scoped to a specific set of permissions selected at creation time. Scopes cannot be
modified after creation. To change scopes, revoke the token and create a new one.

Token values are displayed exactly once at creation and are not recoverable
afterward. Store tokens in a secrets manager. Tokens committed to source control
are automatically revoked when detected by the secret scanning integration.

Revoke a token by selecting Revoke next to it in the token list. Revocation takes
effect within sixty seconds across all regions. Revoked tokens cannot be restored.

Tokens can be given an expiry date of thirty, ninety, one hundred eighty, or three
hundred sixty five days. Tokens without an expiry date are permitted only for
service accounts and require administrator approval. Token usage is logged and
visible in the audit trail.
""",
    },
    {
        "doc_key": "rate_limits",
        "title": "API Rate Limits by Tier",
        "category": "api",
        "embedded_days_ago": 58,
        "content": """
API Rate Limits by Tier.

Standard tier is limited to one hundred requests per minute per organization.
Professional tier is limited to five hundred requests per minute. Enterprise tier
is limited to one thousand requests per minute with burst capacity to two thousand
for periods up to thirty seconds.

Rate limits are enforced per organization, not per token. Multiple tokens belonging
to the same organization share the same limit pool. The bulk export endpoint has a
separate and lower limit of ten requests per minute on all tiers.

Responses include rate limit headers reporting the limit, the remaining allowance,
and the reset timestamp. When the limit is exceeded, the API returns a four hundred
twenty nine status code with a retry after header.

Clients should implement exponential backoff on four hundred twenty nine responses.
Sustained limit violations may result in temporary throttling beyond the stated
limit. Contact your account manager to discuss limit increases.
""",
    },
    {
        "doc_key": "custom_sql_analytics",
        "title": "Custom SQL Analytics",
        "category": "analytics",
        "embedded_days_ago": 13,
        "content": """
Custom SQL Analytics.

Navigate to Analytics, then Custom Queries to access the SQL editor. Queries run
against a read only replica of your organization data. Write operations are
rejected. The available schema is documented in the Schema Reference panel.

Queries are subject to a sixty second execution timeout and a result set limit of
one hundred thousand rows. Queries exceeding either limit are cancelled. Use
aggregation and date range filters to stay within these bounds.

Saved queries can be scheduled to run daily, weekly, or monthly and deliver results
by email or to a configured webhook endpoint. Scheduled queries run under the
permissions of the user who created them.

Query results can be exported directly to CSV or pushed into a dashboard widget.
Widgets refresh on the schedule configured for the parent dashboard rather than on
the query schedule.
""",
    },
]

# Documents whose source text is edited after embedding, creating content drift.
DRIFT_TARGETS = {
    "rate_limits": (
        "\n\nUPDATE. Effective this quarter, enterprise tier rate limits have been "
        "raised to two thousand five hundred requests per minute with burst capacity "
        "to five thousand. The bulk export endpoint limit has been raised to twenty "
        "five requests per minute on enterprise tier. These limits supersede all "
        "figures stated above."
    ),
    "salesforce_integration": (
        "\n\nUPDATE. Opportunity synchronization is now bidirectional for enterprise "
        "tier customers. The default synchronization interval has been reduced to five "
        "minutes for all tiers. Conflict resolution now supports a field level merge "
        "strategy configurable per object."
    ),
}


def seed(reset: bool = False) -> dict:
    print("Initializing database...")
    init_db(drop_first=reset)

    embedder = build_embedder()
    print(f"Embedder backend: {embedder.name}")

    # Fit on the full corpus before any indexing
    corpus_chunks: list[str] = []
    for doc in CORPUS:
        corpus_chunks.extend(chunk_text(doc["content"]))
    embedder.fit(corpus_chunks)
    print(f"Fitted on {len(corpus_chunks)} chunks, dim={embedder.dim}")

    if hasattr(embedder, "save"):
        embedder.save(settings.DATA_DIR / "embedder.pkl")
        print("Saved fitted embedder to data/embedder.pkl")

    store = VectorStore(embedder)
    if reset:
        store.reset()
        print("Vector store reset")

    pipeline = RAGPipeline(embedder, store)

    indexed = []
    with get_session() as session:
        for doc in CORPUS:
            result = pipeline.index_document(
                session,
                doc_key=doc["doc_key"],
                title=doc["title"],
                content=doc["content"],
                category=doc["category"],
                embedded_days_ago=doc["embedded_days_ago"],
            )
            indexed.append(result)
            print(f"  indexed {doc['doc_key']:26s} "
                  f"chunks={result['chunks']:2d} "
                  f"age={doc['embedded_days_ago']:3d}d")

        # Introduce content drift on selected documents
        print("\nApplying content drift...")
        for key, addendum in DRIFT_TARGETS.items():
            doc = repo.get_document(session, key)
            if doc:
                new_content = doc.content + addendum
                repo.touch_document(session, key, new_content, content_hash(new_content))
                print(f"  drift applied to {key} "
                      f"(source updated, embeddings unchanged)")

        stats = repo.corpus_stats(session)

    print(f"\nVector store: {store.count()} chunks")
    print(f"Database: {stats}")
    return {"indexed": indexed, "stats": stats, "chunks": store.count()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the enterprise corpus")
    parser.add_argument("--reset", action="store_true",
                        help="Drop all tables and reset the vector store first")
    args = parser.parse_args()
    seed(reset=args.reset)
    print("\nSeeding complete.")
