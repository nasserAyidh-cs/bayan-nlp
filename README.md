# Bayan | بيان

## Project Overview

Bayan is an Arabic and English complaint analysis service developed during the SDAIA Academy Applied NLP program.

## Problem

Complaints arrive as free text. Bayan extracts structured information that can help service teams identify the topic and find related cases.

## Data and Models

- Bayan complaint data for topic classification.
- `bayan_ner_4000.csv` for named entity recognition.
- `bayan_qa_600.jsonl` for extractive QA in Lab 3b.
- `bayan_cases.csv` for semantic search across 20,000 cases.
- Fine-tuned XLM-R for topic classification, a trained Bayan NER model, and multilingual sentence embeddings with a FAISS index.

## Workflow

The API cleans the complaint and masks supported personal identifiers, then classifies the topic, extracts entities, and retrieves similar cases. The batch extension processes multiple complaints in one request and returns one result per complaint.

## Technologies

Python, PyTorch, Hugging Face Transformers, ONNX Runtime, Sentence Transformers, FAISS, FastAPI, Uvicorn, and Google Colab.

## How to Run the Demo

1. Open `Copy_of_Day4_Bayan_Lab.ipynb` in Google Colab.
2. Mount Drive. Set the paths to the trained topic and NER models, `bench_mix.npy`, and `bayan_cases.csv`. To reuse the search index, provide its matching FAISS file and manifest.
3. Run the notebook in order through the API startup cell.
4. In Swagger, open `POST /v1/analyze-batch`, choose **Try it out**, and send:

```json
{
  "complaints": [
    {"text": "أبلغت نورة عن انقطاع المياه في جدة."},
    {"text": "I was charged twice for the same bill."}
  ]
}
```

## Results and Evaluation

| Check | Observed result |
|---|---|
| NER entity-level F1 | 1.000 on the Lab 3b validation split |
| Batch API | HTTP 200 for an Arabic and English complaint in one request |
| Phone masking | The test phone number was replaced with `[PHONE]` in the API response |
| Full analysis HTTP load test | 160/160 successful requests with 16 concurrent users. p50: 3,893.55 ms. p99: 6,310.45 ms. |

The load test measured `/v1/analyze`, including classification, NER, and search. Classifier-only HTTP latency was not measured. The service was demonstrated in Colab. The standalone `make serve` command is pending.

## Limitations and Future Improvements

Improve Arabic entity boundaries and avoid labeling masking placeholders as entities. Add a standalone service entry point and benchmark the classifier separately.

## SDAIA Academy

Project developed as part of the SDAIA Academy Applied NLP program.
