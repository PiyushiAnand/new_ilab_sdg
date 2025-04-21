from datasets import load_dataset
from openai import OpenAI

# First Party
from sdg_hub.flow import Flow
from sdg_hub.pipeline import Pipeline
from sdg_hub.sdg import SDG
from sdg_hub.utils.docprocessor import DocProcessor
import sys
endpoint = f"http://localhost:8000/v1"
openai_api_key = "EMPTY"
openai_api_base = endpoint

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
teacher_model = client.models.list().data[0].id
print(teacher_model)

knowledge_agentic_pipeline = "../../../src/instructlab/sdg/flows/generation/knowledge/translate_knowledge.yaml"
flow_cfg = Flow(client).get_flow_from_file(knowledge_agentic_pipeline)
sdg = SDG(
    [Pipeline(flow_cfg)],
    num_workers=1,
    batch_size=1,
    save_freq=1000,
)

number_of_samples = 5
seed_data_dir = f"sdg_demo_output/"
ds = load_dataset('json', data_files=f'{seed_data_dir}/seed_data.jsonl', split='train')
ds = ds.shuffle(seed=42).select(range(number_of_samples))

# Checkpoint directory is used to save the intermediate datasets
generated_data = sdg.generate(ds, checkpoint_dir="Tmp")