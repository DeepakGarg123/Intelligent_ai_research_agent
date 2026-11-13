
import json

from backend.llm_provider import invoke_with_fallback


def create_batches(chunks, batch_size=20):

    batches = []

    for start in range(0, len(chunks), batch_size):

        batch = chunks[start : start + batch_size]

        batches.append(batch)

    return batches


def extract_information_from_batch(batch, query):

    batch_context = ""

    for index, document in enumerate(batch, start=1):

        metadata = document.metadata

        source = metadata.get("source", "Unknown")
        page = metadata.get("page_label", metadata.get("page", "Unknown"))

        batch_context += (
            f"\n[Chunk {index} | Source: {source} | Page: {page}]\n"
            f"{document.page_content}\n"
        )

    prompt = f"""
    You are a document information extraction system.

    User request:
    {query}

    Examine the document content carefully.

    Extract all items relevant to the user's request
    that are explicitly mentioned in the provided content.

    Rules:
    1. Do not invent information.
    2. Do not include unrelated items.
    3. Preserve the source filename and page number.
    4. If no relevant information is found, return an empty list.
    5. Return valid JSON only.

    Required JSON format:

    {{
        "items": [
            {{
                "name": "Item name",
                "description": "Short description",
                "source": "Filename",
                "page": "Page number"
            }}
        ]
    }}

    Document content:
    {batch_context}
    """

    response = invoke_with_fallback(prompt)

    content = response.content

    if isinstance(content, list):
        content = "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )

    content = content.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(content, strict=False)

    except json.JSONDecodeError as error:

        print("LLM returned invalid JSON:")
        print(content)

        raise ValueError(f"Could not parse LLM response: {error}")

    return data.get("items", [])


def remove_duplicate_projects(items):

    unique_projects = {}

    for item in items:

        name = item.get("name", "").strip()

        if not name:
            continue

        key = name.lower()

        if key not in unique_projects:

            unique_projects[key] = {
                "name": name,
                "description": item.get("description", ""),
                "source": item.get("source", "Unknown"),
                "pages": [item.get("page", "Unknown")]
            }

        else:

            existing_pages = unique_projects[key]["pages"]

            new_page = item.get("page", "Unknown")

            if new_page not in existing_pages:
                existing_pages.append(new_page)

    return list(unique_projects.values())


def search_entire_document(chunks, query):

    batches = create_batches(chunks)

    all_results = []

    for index, batch in enumerate(batches, start=1):

        print(f"Processing batch {index}/{len(batches)}")

        results = extract_information_from_batch(batch, query)

        all_results.extend(results)

    unique_results = remove_duplicate_projects(all_results)

    return unique_results
