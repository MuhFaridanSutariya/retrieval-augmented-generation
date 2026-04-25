from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "corpus" / "sample_complex.pdf"


def _build_story():
    styles = getSampleStyleSheet()
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    body = styles["BodyText"]
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=18, bulletIndent=6)

    story = []

    story.append(Paragraph("Aurora Platform — Quarterly Operations Report", h1))
    story.append(Paragraph("Reporting period: 2026-Q1 (January – March)", body))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Executive Summary", h2))
    story.append(
        Paragraph(
            "Aurora handled 18.4 million queries in Q1 2026, a 22 percent increase over Q4 2025. "
            "Median /ask latency dropped from 1.9 seconds to 1.4 seconds after the rollout of "
            "hybrid retrieval on 2026-02-14. The on-call team responded to four production incidents; "
            "all were resolved within the SLA. Customer-reported accuracy on the internal eval set "
            "improved from 0.71 to 0.78 measured by exact-match on the golden questions.",
            body,
        )
    )
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Service-Level Metrics", h2))
    sla_data = [
        ["Metric", "Q4 2025", "Q1 2026", "SLA Target"],
        ["p50 latency (ms)", "1900", "1400", "≤ 2000"],
        ["p95 latency (ms)", "5200", "3100", "≤ 5000"],
        ["p99 latency (ms)", "8800", "5400", "≤ 8000"],
        ["Availability (%)", "99.81", "99.94", "≥ 99.9"],
        ["Error rate (%)", "0.62", "0.18", "≤ 0.5"],
        ["Cache hit rate (%)", "27.4", "34.7", "≥ 30"],
    ]
    sla_table = Table(sla_data, hAlign="LEFT")
    sla_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(sla_table)
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Cost Breakdown", h2))
    cost_data = [
        ["Component", "Q4 2025 (USD)", "Q1 2026 (USD)", "Change"],
        ["OpenAI chat (gpt-5.4)", "12,400", "16,820", "+35.6%"],
        ["OpenAI embeddings", "640", "780", "+21.9%"],
        ["Vector store (FAISS hosting)", "0", "0", "—"],
        ["Postgres + Redis", "210", "220", "+4.8%"],
        ["Cloud egress", "85", "120", "+41.2%"],
        ["Total", "13,335", "17,940", "+34.5%"],
    ]
    cost_table = Table(cost_data, hAlign="LEFT")
    cost_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("BACKGROUND", (0, -1), (-1, -1), colors.whitesmoke),
                ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
            ]
        )
    )
    story.append(cost_table)

    story.append(PageBreak())

    story.append(Paragraph("Production Incidents", h2))
    story.append(
        Paragraph(
            "Four incidents were declared during Q1. Severities follow the standard scale: "
            "S1 = full outage, S2 = degraded service for a class of users, S3 = minor issue.",
            body,
        )
    )
    incidents_data = [
        ["Date", "Severity", "Duration", "Root cause", "Mitigation"],
        [
            "2026-01-08",
            "S2",
            "32 min",
            "OpenAI rate-limit cascade",
            "Added per-tenant token bucket",
        ],
        [
            "2026-01-22",
            "S3",
            "11 min",
            "Stale BM25 index after redeploy",
            "Generation counter in FaissStore",
        ],
        [
            "2026-02-09",
            "S1",
            "47 min",
            "Postgres connection-pool exhaustion",
            "Bumped pool_size from 10 to 25",
        ],
        [
            "2026-03-17",
            "S2",
            "18 min",
            "Reranker JSON parse failure on emoji input",
            "Defensive parse fallback to fused order",
        ],
    ]
    incidents_table = Table(incidents_data, hAlign="LEFT", colWidths=[0.9*inch, 0.6*inch, 0.7*inch, 1.8*inch, 2.0*inch])
    incidents_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(incidents_table)
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Roadmap Highlights", h2))
    bullets = [
        "Q2 2026 — Introduce optional cross-encoder reranker for queries flagged as high-stakes by the safety classifier.",
        "Q2 2026 — Migrate FAISS to Qdrant once corpus exceeds 10 million chunks; the wrapper interface in storages/ is already vendor-neutral.",
        "Q3 2026 — Roll out OCR fallback (Tesseract) for scanned PDFs, gated behind an UPLOAD_ENABLE_OCR feature flag.",
        "Q3 2026 — Customer-facing eval dashboard surfacing the daily golden-set pass rate and prompt-version comparison.",
        "Q4 2026 — Multi-tenant cost reporting per workspace, surfacing prompt_tokens / completion_tokens by user.",
    ]
    for item in bullets:
        story.append(Paragraph(f"&bull; {item}", bullet))

    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("Owner: Platform Engineering. Distribution: internal only.", body))

    return story


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    doc.build(_build_story())
    size = OUTPUT_PATH.stat().st_size
    print(f"wrote {OUTPUT_PATH} ({size} bytes)")


if __name__ == "__main__":
    main()
