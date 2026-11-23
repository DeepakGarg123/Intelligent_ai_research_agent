from pathlib import Path

from backend.vision.vision_provider import extract_text_from_image


project_root = Path(__file__).resolve().parents[2]

image_path = project_root / "sample.png"

if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path}")

with open(image_path, "rb") as image_file:
    image_bytes = image_file.read()

print("Starting Vision Model Test...")

extracted_text = extract_text_from_image(
    image_bytes,
    mime_type="image/png"
)

print("\nExtracted Text:")
print("------------------------------")
print(extracted_text)
print("------------------------------")
