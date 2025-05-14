# Synthetic Data Generation using Translation
In this example, we demonstrate how we can use `sdg_instruct` to generate synthetic data for low-resource language using translation. The intuition is that we do not have very teacher models for low-resource languages but good teacher model for English and a decent translation system. rather than generating synthetic data in the low-resource language we generate the data in English and then translate to low-resource language. We consider Hindi as the low-resource language in this tutorial.

## Pre-Requisite

Run the following command to install the required packages 
```bash
pip install -r requirements.txt
```

## Data Preparation
The documents in Hindi are present in the folder [wikipedia](wikipedia). The documents are stored in markdown format. We also create a `yaml` file containing few-shot context-question-response triplets. The sample documents and `qna.yaml` is present here [wikipedia](wikipedia). We now need to convert the data into the required format. We provide a notebook file for the same [here](document_pre_processing.ipynb). Once the notebook is run we will have the following files in the `sdg_demo_output` folder

```
sdg_demo_output/
└── seed_data.jsonl
```

## Serving the Translation and Teacher Model 
Before running the translation pipeline, we need to load the NLLB translation model and the teacher model for SDG.

To load the NLLB translation model using FastAPI, run the following command from a different terminal

```bash
uvicorn nllb_server:app --reload
```

From another terminal, load the teacher model using VLLM, run the following command

```bash
vllm serve ibm-granite/granite-3.3-2b-instruct --port 8082 --max_model_len 2048
```

## Synthetic Data Generation
We have provided the notebook [here](translation-data-generation.ipynb). The documents in Hindi are translated to English using Facebook nllb-200-distilled-600M model. This translated document in English is fed to the granite-3.3-2b-instruct teacher model to generate question and answer pairs in English. The generated question-answer pairs are translated back to Hindi using the same Facebook nllb-200-distilled-600M model.
