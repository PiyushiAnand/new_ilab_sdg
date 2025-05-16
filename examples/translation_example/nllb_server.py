from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = FastAPI()

# Load the model and tokenizer
MODEL_NAME = "facebook/nllb-200-1.3B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


# Input format mimicking OpenAI completion endpoint
class CompletionRequest(BaseModel):
    prompt: str
    source_lang: str = "eng_Latn"
    target_lang: str = "hin_Deva"
    max_length: int = 512


@app.post("/v1/completions")
async def translate(req: CompletionRequest):
    # Set source and target language
    tokenizer.src_lang = req.source_lang

    # Tokenize input
    inputs = tokenizer(req.prompt, return_tensors="pt")

    # Generate translation
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(req.target_lang),
            max_length=req.max_length,
        )

    # Decode output
    translated_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

    # Return response in OpenAI-style format
    return {
        "id": "nllb-translation",
        "object": "text_completion",
        "model": MODEL_NAME,
        "choices": [
            {
                "text": translated_text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
    }


@app.get("/v1/completions")
async def translate(source_lang: str, target_lang: str, prompt: str):
    # Set source and target language
    tokenizer.src_lang = source_lang

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate translation
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang),
            max_length=512,
        )

    # Decode output
    translated_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

    # Return response in OpenAI-style format
    return {
        "id": "nllb-translation",
        "object": "text_completion",
        "model": MODEL_NAME,
        "choices": [
            {
                "text": translated_text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
    }
