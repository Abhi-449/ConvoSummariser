# llm_service.py
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFacePipeline

def load_llm(model_name="google/flan-t5-base"):
    device = 0 if torch.cuda.is_available() else -1
    print(f"[INFO] Using {'GPU' if device == 0 else 'CPU'}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=512,   # 👈 ensure int
        num_beams=4,
        truncation = True,    )

    return HuggingFacePipeline(pipeline=pipe)
