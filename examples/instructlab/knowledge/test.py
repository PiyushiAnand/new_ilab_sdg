import sys
import subprocess
from datasets import load_dataset
from openai import OpenAI
import os

# First Party
from sdg_hub.flow import Flow
from sdg_hub.pipeline import Pipeline
from sdg_hub.sdg import SDG
from sdg_hub.utils.docprocessor import DocProcessor

# Configuration
endpoint = "http://localhost:8000/v1"
openai_api_key = "EMPTY"
openai_api_base = endpoint
data_dir = 'document_collection/ibm-annual-report'
output_dir = "sdg_demo_output"

# Step 1: Parse documents using docparser.py
# Equivalent to: !OMP_NUM_THREADS=32 mamba run -n docling python ../scripts/docparser.py --input-dir {data_dir} --output-dir {data_dir}
subprocess.run([
    "python", "../scripts/docparser.py",
    "--input-dir", data_dir,
    "--output-dir", data_dir
], env={**os.environ, "OMP_NUM_THREADS": "32"})


# Step 2: Process documents into datasets
dp = DocProcessor(data_dir, user_config_path=f'{data_dir}/qna.yaml')

# Option 1: JSON-based seed data
seed_data = dp.get_processed_dataset()

# Option 2: Markdown-based seed data (you can use this instead or in addition)
# seed_data = dp.get_processed_markdown_dataset([f"{data_dir}/ibm-annual-report-2024.md"])

# Save seed data
seed_data.to_json(f'{output_dir}/seed_data.jsonl', orient='records', lines=True)

# Step 3: Setup OpenAI-compatible client
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
teacher_model = client.models.list().data[0].id
print("Using model:", teacher_model)

# Step 4: Load and configure SDG flow
knowledge_agentic_pipeline = "/opt/app-root/src/.local/share/instructlab/new_ilab_sdg/src/sdg_hub/flows/generation/knowledge/translate_knowledge.yaml"
flow_cfg = Flow(client).get_flow_from_file(knowledge_agentic_pipeline)
sdg = SDG(
    [Pipeline(flow_cfg)],
    num_workers=1,
    batch_size=1,
    save_freq=1000,
)

# Step 5: Load seed data and sample
number_of_samples = 5
ds = load_dataset('json', data_files=f'{output_dir}/seed_data.jsonl', split='train')
ds = ds.shuffle(seed=42).select(range(number_of_samples))

# Step 6: Generate using SDG
generated_data = sdg.generate(ds, checkpoint_dir="Tmp")
print("Generated data:", generated_data)
generated_data.to_json("sdg_demo_output/generated_output.jsonl", orient="records", lines=True)

