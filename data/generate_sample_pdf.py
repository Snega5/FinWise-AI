"""Create a small sample policy PDF (no third-party libraries) for testing document Q&A.

Run: python data/generate_sample_pdf.py
Also importable: make_pdf(pages) returns PDF bytes (used by the tests).
"""
import textwrap
from pathlib import Path


def _esc(s: str) -> str:
    s = s.encode("latin-1", "replace").decode("latin-1")
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(pages: list[list[str]]) -> bytes:
    """Build a text PDF. `pages` is a list of pages; each page is a list of text lines."""
    n = len(pages)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n))
    objs: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, lines in enumerate(pages):
        content_id = 5 + 2 * i
        objs.append(
            (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
             f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>").encode()
        )
        body = ["BT", "/F1 11 Tf", "15 TL", "56 740 Td"]
        for line in lines:
            body.append(f"({_esc(line)}) Tj T*")
        body.append("ET")
        stream = "\n".join(body).encode("latin-1")
        objs.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def _wrap(text: str, width: int = 88) -> list[str]:
    return textwrap.wrap(text, width) + [""]


POLICY_PAGES = [
    ["Sharma Household Money Policy", "Version 1.0 - Synthetic sample document for testing", ""]
    + _wrap("1. Savings target: The household saves at least 25% of net monthly income. The "
            "transfer to savings is made on salary day, before any other spending.")
    + _wrap("2. Emergency fund: The household keeps 6 months of essential expenses in a separate "
            "savings account. It is not used for planned purchases or holidays.")
    + _wrap("3. Spending cap: Dining out and entertainment together must not exceed 12% of "
            "monthly income. Any overspend is made up in the following month.")
    + _wrap("4. Large purchases: Any single non-essential purchase above Rs. 15,000 requires a "
            "7-day waiting period and the agreement of both partners."),
    ["Sharma Household Money Policy (continued)", ""]
    + _wrap("5. Debt: Credit card balances are paid in full every month. The household never "
            "pays interest on purchases. Loans are taken only for a home or education.")
    + _wrap("6. Review: The household reviews this policy and the monthly budget on the first "
            "Sunday of every quarter and updates the targets if income changes."),
]

if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "sample_documents"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "household_money_policy.pdf"
    path.write_bytes(make_pdf(POLICY_PAGES))
    print(f"Wrote {path}")
