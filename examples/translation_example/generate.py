# Third Party
from datasets import load_dataset
from openai import OpenAI
import click

# First Party
from sdg_hub.flow import Flow
from sdg_hub.logger_config import setup_logger
from sdg_hub.pipeline import Pipeline
from sdg_hub.sdg import SDG
from sdg_hub.prompts import PromptRegistry
from sdg_hub.blocks import BlockRegistry, Block
from blocks.translation_block import TranslationBlock
from transformers import AutoTokenizer
import re
import logging
from typing import List
from datasets import Dataset
from tqdm import tqdm

logger = setup_logger(__name__)


### Granite 3.3 2B Chat Template
@PromptRegistry.register("ibm-granite/granite-3.3-2b-instruct")
def granite_3_3_2b_chat_template():
    return """{# Alias tools -> available_tools #}\n{%- if tools and not available_tools -%}\n    {%- set available_tools = tools -%}\n{%- endif -%}\n{%- if messages[0]['role'] == 'system' %}\n     {%- set system_message = messages[0]['content'] %}\n     {%- set loop_messages = messages[1:] %}\n {%- else %}\n     {%- set system_message = \" Knowledge Cutoff Date: April 2024.\n. You are Granite, developed by IBM.\" %}\n     {%- if available_tools and documents %}\n         {%- set system_message = system_message + \" You are a helpful assistant with access to the following tools. When a tool is required to answer the user's query, respond only with <|tool_call|> followed by a JSON list of tools used. If a tool does not exist in the provided list of tools, notify the user that you do not have the ability to fulfill the request. \nWrite the response to the user's input by strictly aligning with the facts in the provided documents. If the information needed to answer the question is not available in the documents, inform the user that the question cannot be answered based on the available data.\" %}\n     {%- elif available_tools %}\n         {%- set system_message = system_message + \" You are a helpful assistant with access to the following tools. When a tool is required to answer the user's query, respond only with <|tool_call|> followed by a JSON list of tools used. If a tool does not exist in the provided list of tools, notify the user that you do not have the ability to fulfill the request.\" %}\n     {%- elif documents %}\n         {%- set system_message = system_message + \" Write the response to the user's input by strictly aligning with the facts in the provided documents. If the information needed to answer the question is not available in the documents, inform the user that the question cannot be answered based on the available data.\" %}\n    {%- elif thinking %}\n    {%- set system_message = system_message + \" You are a helpful AI assistant.\nRespond to every user query in a comprehensive and detailed way. You can write down your thoughts and reasoning process before responding. In the thought process, engage in a comprehensive cycle of analysis, summarization, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. In the response section, based on various attempts, explorations, and reflections from the thoughts section, systematically present the final solution that you deem correct. The response should summarize the thought process. Write your thoughts between <think></think> and write your response between <response></response> for each user query.\" %}\n     {%- else %}\n         {%- set system_message = system_message + \" You are a helpful AI assistant.\" %}\n     {%- endif %}\n     {%- if 'citations' in controls and documents %}\n         {%- set system_message = system_message + ' \nUse the symbols <|start_of_cite|> and <|end_of_cite|> to indicate when a fact comes from a document in the search result, e.g <|start_of_cite|> {document_id: 1}my fact <|end_of_cite|> for a fact from document 1. Afterwards, list all the citations with their corresponding documents in an ordered list.' %}\n     {%- endif %}\n     {%- if 'hallucinations' in controls and documents %}\n         {%- set system_message = system_message + ' \nFinally, after the response is written, include a numbered list of sentences from the response with a corresponding risk value that are hallucinated and not based in the documents.' %}\n     {%- endif %}\n     {%- set loop_messages = messages %}\n {%- endif %}\n {{- '<|start_of_role|>system<|end_of_role|>' + system_message + '<|end_of_text|>\n' }}\n {%- if available_tools %}\n     {{- '<|start_of_role|>available_tools<|end_of_role|>' }}\n     {{- available_tools | tojson(indent=4) }}\n     {{- '<|end_of_text|>\n' }}\n {%- endif %}\n {%- if documents %}\n     {%- for document in documents %}\n         {{- '<|start_of_role|>document {\"document_id\": \"' + document['doc_id'] | string + '\"}<|end_of_role|>\n' }}\n         {{- document['text'] }}\n         {{- '<|end_of_text|>\n' }}\n              {%- endfor %}\n {%- endif %}\n {%- for message in loop_messages %}\n     {{- '<|start_of_role|>' + message['role'] + '<|end_of_role|>' + message['content'] + '<|end_of_text|>\n' }}\n     {%- if loop.last and add_generation_prompt %}\n         {{- '<|start_of_role|>assistant' }}\n             {%- if controls %}\n                 {{- ' ' + controls | tojson()}}\n             {%- endif %}\n         {{- '<|end_of_role|>' }}\n     {%- endif %}\n {%- endfor %}"""


@click.command()
@click.option(
    "--ds_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the dataset.",
)
@click.option("--bs", type=int, default=8, show_default=True, help="Batch size.")
@click.option(
    "--num_workers", type=int, default=32, show_default=True, help="Number of workers."
)
@click.option(
    "--save_path", type=click.Path(), required=True, help="Path to save the output."
)
@click.option(
    "--llm_endpoint", type=str, required=True, help="LLM Endpoint for data processing."
)
@click.option(
    "--translation_endpoint",
    type=str,
    required=True,
    help="Endpoint for Translation Model.",
)
@click.option(
    "--flow", type=str, required=True, help="Flow configuration for the process."
)
@click.option(
    "--checkpoint_dir",
    type=click.Path(),
    required=True,
    help="Path to save checkpoints.",
)
@click.option(
    "--save_freq",
    type=int,
    default=2,
    show_default=True,
    help="Frequency to save checkpoints.",
)
@click.option("--debug", is_flag=True, help="Enable debug mode.")
@click.option(
    "--dataset_start_index", type=int, default=0, help="Start index of the dataset."
)
@click.option(
    "--dataset_end_index", type=int, default=None, help="End index of the dataset."
)
def main(
    ds_path,
    bs,
    num_workers,
    save_path,
    llm_endpoint,
    translation_endpoint,
    flow,
    checkpoint_dir,
    save_freq,
    debug,
    dataset_start_index,
    dataset_end_index,
):
    """
    Main function to process the dataset.

    Parameters:
    ds_path (str): Path to the dataset.
    bs (int): Batch size.
    num_workers (int): Number of workers.
    save_path (str): Path to save the output.
    llm_endpoint (str): LLM Endpoint for data processing.
    translation_endpoint (str): Translation Endpoint.
    flow (str): Flow configuration for the process.
    checkpoint_dir (str): Path to save checkpoints.
    save_freq (int): Frequency to save checkpoints.
    debug (bool): Enable debug mode.
    """
    logger.info(f"Generation configuration: {locals()}\n\n")
    ds = load_dataset("json", data_files=ds_path, split="train")
    if dataset_start_index is not None and dataset_end_index is not None:
        if dataset_end_index > len(ds):
            dataset_end_index = len(ds)
        ds = ds.select(range(dataset_start_index, dataset_end_index))
        logger.info(f"Dataset sliced from {dataset_start_index} to {dataset_end_index}")

    if debug:
        # For debugging, use a smaller subset of the dataset
        ds = ds.shuffle(seed=42).select(range(5))

    logger.warning(f"Dataset: {ds}")

    openai_api_key = "EMPTY"
    llm_openai_api_base = llm_endpoint

    llm_client = OpenAI(
        api_key=openai_api_key,
        base_url=llm_openai_api_base,
    )

    # Verify we can see the model
    teacher_model = llm_client.models.list().data[0].id
    logger.warning(f"Connected to model: {teacher_model}")

    translation_openai_api_base = translation_endpoint

    translation_client = OpenAI(
        api_key=openai_api_key,
        base_url=translation_openai_api_base,
    )

    flow_cfg = Flow(llm_client=llm_client).get_flow_from_file(flow)

    for index in range(len(flow_cfg)):
        if issubclass(flow_cfg[index]["block_type"], TranslationBlock):
            flow_cfg[index]["block_config"]["client"] = translation_client

    sdg = SDG(
        [Pipeline(flow_cfg)],
        num_workers=num_workers,
        batch_size=bs,
        save_freq=save_freq,
    )

    generated_data = sdg.generate(ds, checkpoint_dir=checkpoint_dir)

    save_path = save_path.replace(
        ".jsonl", f"_{dataset_start_index}_{dataset_end_index}.jsonl"
    )
    generated_data.to_json(save_path, orient="records", lines=True, force_ascii=False)
    logger.info(f"Data saved to {save_path}")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
