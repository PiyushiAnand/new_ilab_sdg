from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from IndicTransToolkit.processor import IndicProcessor


app = FastAPI()

# Load the model and tokenizer
EN_INDIC_MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"
en_indic_tokenizer = AutoTokenizer.from_pretrained(
    EN_INDIC_MODEL_NAME, trust_remote_code=True
)
en_indic_model = AutoModelForSeq2SeqLM.from_pretrained(
    EN_INDIC_MODEL_NAME, trust_remote_code=True
)
en_indic_model.eval()

INDIC_EN_MODEL_NAME = "ai4bharat/indictrans2-indic-en-dist-200M"
indic_en_tokenizer = AutoTokenizer.from_pretrained(
    INDIC_EN_MODEL_NAME, trust_remote_code=True
)
indic_en_model = AutoModelForSeq2SeqLM.from_pretrained(
    INDIC_EN_MODEL_NAME, trust_remote_code=True
)
indic_en_model.eval()

ip = IndicProcessor(inference=True)


def translate_text(text_batch, model, tokenizer):
    inputs = tokenizer(
        text_batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
        )

    return tokenizer.batch_decode(
        output,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )


# Input format mimicking OpenAI completion endpoint
class CompletionRequest(BaseModel):
    prompt: str
    source_lang: str = "kan_Knda"
    target_lang: str = "eng_Latn"
    max_length: int = 512


@app.post("/v1/completions")
async def translate(req: CompletionRequest):
    batch = ip.preprocess_batch(
        [req.prompt],
        src_lang=req.source_lang,
        tgt_lang=req.target_lang,
    )

    if req.source_lang == "eng_Latn":
        generated_tokens = translate_text(batch, en_indic_model, en_indic_tokenizer)
    else:
        generated_tokens = translate_text(batch, indic_en_model, indic_en_tokenizer)

    translations = ip.postprocess_batch(generated_tokens, lang=req.target_lang)

    # Return response in OpenAI-style format
    return {
        "id": "indictrans-translation",
        "object": "text_completion",
        "model": (
            INDIC_EN_MODEL_NAME
            if req.target_lang == "eng_Latn"
            else EN_INDIC_MODEL_NAME
        ),
        "choices": [
            {
                "text": translations[0],
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
    }


@app.get("/v1/completions")
async def translate(source_lang: str, target_lang: str, prompt: str):
    # Create a request object and reuse the POST endpoint logic
    req = CompletionRequest(
        prompt=prompt, source_lang=source_lang, target_lang=target_lang
    )
    return await translate(req)
