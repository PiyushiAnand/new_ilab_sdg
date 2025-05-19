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

print(f"Chunked data into {len(bns_documents_chunked)} chunks")

doc_with_icl = []

icl_context_1 = """
CHAPTER XVII: OF OFFENCES AGAINST PROPERTY

Subchapter: Of criminal trespass

Section 329: Criminal trespass and house-trespass
(1) Whoever enters into or upon property in the possession of another with intent to commit an offence or to intimidate, insult or annoy any person in possession of such property or having lawfully entered into or upon such property, unlawfully remains there with intent thereby to intimidate, insult or annoy any such person or with intent to commit an offence is said to commit criminal trespass. (2) Whoever commits criminal trespass by entering into or remaining in any building, tent or vessel used as a human dwelling or any building used as a place for worship, or as a place for the custody of property, is said to commit house-trespass.
Explanation: The introduction of any part of the criminal trespassers body is entering sufficient to constitute house-trespass. (3) Whoever commits criminal trespass shall be punished with imprisonment of either description for a term which may extend to three months, or with fine which may extend to five thousand rupees, or with both. (4) Whoever commits house-trespass shall be punished with imprisonment of either description for a term which may extend to one year, or with fine which may extend to five thousand rupees, or with both.
"""

icl_scenario_1 = "I am living in a rented 1 BHK house in Bangalore for past two years. Everything was going smoothly. Last month the house owner asked me to vacate the house in a week. He has threatened me that if not then he will discontinue the water and electricity supply. I don't know what to do."

icl_response_1 = "According to Chapter XVII and section 329 of Bharatiya Nyaya Sanhita, illegally or forcefully removing a tenant from property they peacefully occupy constitutes criminal trespass."

icl_context_2 = """
CHAPTER XIX: OF CRIMINAL INTIMIDATION, INSULT, ANNOYANCE, DEFAMATION, ETC.

Section 351: Criminal intimidation
(1) Whoever threatens another by any means, with any injury to his person, reputation or property, or to the person or reputation of any one in whom that person is interested, with intent to cause alarm to that person, or to cause that person to do any act which he is not legally bound to do, or to omit to do any act which that person is legally entitled to do, as the means of avoiding the execution of such threat, commits criminal intimidation.
Explanation: A threat to injure the reputation of any deceased person in whom the person threatened is interested, is within this section.
Illustration.
A, for the purpose of inducing B to resist from prosecuting a civil suit, threatens to burn Bs house. A is guilty of criminal intimidation. (2) Whoever commits the offence of criminal intimidation shall be punished with imprisonment of either description for a term which may extend to two years, or with fine, or with both. (3) Whoever commits the offence of criminal intimidation by threatening to cause death or grievous hurt, or to cause the destruction of any property by fire, or to cause an offence punishable with death or imprisonment for life, or with imprisonment for a term which may extend to seven years, or to impute unchastity to a woman, shall be punished with imprisonment of either description for a term which may extend to seven years, or with fine, or with both. (4) Whoever commits the offence of criminal intimidation by an anonymous communication, or having taken precaution to conceal the name or abode of the person from whom the threat comes, shall be punished with imprisonment of either description for a term which may extend to two years, in addition to the punishment provided for the offence under sub-section (1).
"""

icl_scenario_2 = "I am living in a rented 1 BHK house in Bangalore for past two years. Everything was going smoothly. Last month the house owner asked me to vacate the house in a week. He has threatened me that if not then he will discontinue the water and electricity supply. I don't know what to do."

icl_response_2 = "Under Chapter XVII and section 329 of Bharatiya Nyaya Sanhita, property owners can be held liable for criminal trespass if they employ threats or coercive tactics, such as disconnecting essential utilities like water or electricity, to force out tenants in peaceful possession.RetryClaude can make mistakes. Please double-check responses."

for each_document in bns_documents_chunked:
    icl_dict = {}
    icl_dict["icl_document_1"] = icl_context_1

    icl_dict["icl_query_1"] = icl_scenario_1
    icl_dict["icl_response_1"] = icl_response_1

    icl_dict["icl_document_2"] = icl_context_1

    icl_dict["icl_query_2"] = icl_scenario_2
    icl_dict["icl_response_2"] = icl_response_2

    icl_dict["chapter_name"] = each_document["chapter_name"]
    icl_dict["section_name"] = each_document["section_name"]
    icl_dict["text"] = each_document["document"]

    doc_with_icl.append(icl_dict)

seed_data = Dataset.from_list(doc_with_icl)

seed_data.to_json(
    f"{output_dir}/seed_data.jsonl", orient="records", lines=True, force_ascii=False
)
