from sdg_hub.utils.chunking import chunk_document
from datasets import Dataset
from tqdm import tqdm
from pathlib import Path

# output directory
output_dir = f"sdg_demo_output/"

chunk_size = 100
max_model_context_length = 2048


def load_documents(data_folder):
    # Dictionary to store content
    documents = []

    base_path = Path(data_folder)

    # Iterate over each chapter folder
    for chapter_folder in base_path.iterdir():
        if chapter_folder.is_dir():
            chapter_name = chapter_folder.name
            for section_file in chapter_folder.glob("*.md"):
                section_name = section_file.stem  # 'Section_01', without '.md'
                # Read file content
                with open(section_file, "r", encoding="utf-8") as f:
                    content = f.read()
                # Store in dictionary

                document_content = {}
                document_content["chapter_name"] = chapter_name
                document_content["section_name"] = section_name
                document_content["text"] = content

                documents.append(document_content)
    return documents


bns_documents = load_documents("data/")
print(f"Read {len(bns_documents)} number of Sections")

bns_documents_chunked = []

max_documents = 100
doc_count = 0
for each_doc in tqdm(bns_documents, desc="For each document"):
    document = [
        {
            "document": chunk,
            "chapter_name": each_doc["chapter_name"],
            "section_name": each_doc["section_name"],
        }
        for chunk in chunk_document(
            each_doc["text"],
            server_ctx_size=max_model_context_length,
            chunk_word_count=chunk_size,
        )
    ]
    bns_documents_chunked.extend(document)
    doc_count += 1

    if doc_count > max_documents:
        break

print(f"Chunked data into {len(bns_documents_chunked)} chunks")

doc_with_icl = []

icl_context = """
Section 49: Punishment of abetment if act abetted is committed in consequence and where no express provision is made for its punishment Whoever abets any offence shall, if the act abetted is committed in consequence of the abetment, and no express provision is made by this Sanhita for the punishment of such abetment, be punished with the punishment provided for the offence. Explanation: An act or offence is said to be committed in consequence of abetment, when it is committed in consequence of the instigation, or in pursuance of the conspiracy, or with the aid which constitutes the abetment. Illustrations. (a) A instigates B to give false evidence. B, in consequence of the instigation, commits that offence. A is guilty of abetting that offence, and is liable to the same punishment as B. (b) A and B conspire to poison Z. A, in pursuance of the conspiracy, procures the poison and delivers it to B in order that he may administer it to Z. B, in pursuance of the conspiracy, administers the poison to Z in As absence and thereby causes Zs death. Here B is guilty of murder. A is guilty of abetting that offence by conspiracy, and is liable to the punishment for murder.
"""

for each_document in bns_documents_chunked:
    icl_dict = {}
    icl_dict["icl_document"] = icl_context

    icl_dict["icl_query"] = (
        "If I encouraged someone to commit a crime and they actually went ahead and did it, but I wasn’t physically present at the scene, can I still be punished the same as the person who committed the crime under Indian law?"
    )
    icl_dict["icl_response"] = (
        f"Based on Chapter {each_document['chapter_name']} and section {each_document['section_name']} of Bharatiya Nyaya Sanhita, Yes, you can still be punished the same as the person who committed the crime."
    )

    icl_dict["chapter_name"] = each_document["chapter_name"]
    icl_dict["section_name"] = each_document["section_name"]
    icl_dict["text"] = each_document["document"]

    doc_with_icl.append(icl_dict)

seed_data = Dataset.from_list(doc_with_icl)

seed_data.to_json(
    f"{output_dir}/seed_data.jsonl", orient="records", lines=True, force_ascii=False
)
