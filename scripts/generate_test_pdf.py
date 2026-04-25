from pathlib import Path

from app.utils.sample_pdf import SAMPLE_FILENAME, generate_sample_complex_pdf

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "corpus" / SAMPLE_FILENAME


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pdf_bytes = generate_sample_complex_pdf()
    OUTPUT_PATH.write_bytes(pdf_bytes)
    print(f"wrote {OUTPUT_PATH} ({len(pdf_bytes)} bytes)")


if __name__ == "__main__":
    main()
