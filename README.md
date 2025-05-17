# 📢 AMI Meeting Transcript Summarizer & QA System

A two-stage pipeline that uses LLMs to generate high-quality meeting summaries and question–answer pairs, then fine-tunes lightweight sequence-to-sequence and QA models for on-prem inference. Includes a Gradio interface so end-users can paste any meeting transcript, get a 200-word narrative summary, and ask follow-up questions.

---

## 🚀 Project Overview

Academic collaboration to automate post-meeting deliverables:

1. **Data Generation**  
   - Used OpenAI’s API (GPT-3.5-turbo model) to produce “enhanced” abstractive summaries (200–250 words) of AMI meeting transcripts.  
   - Generated 5 JSON QA pairs per meeting from transcript + summary prompts.

2. **Model Fine-Tuning**  
   - **Summarization**: instruction-tuned `flan-t5-base` fine-tuned on ~100 transcript→summary examples.  
   - **Question Answering**: fine-tuned `distilbert-base-uncased` and `flan-t5-base`, went with `flan-t5-base` on the exploded QA dataset.

3. **Evaluation & Metrics**  
   - **Summaries**: ROUGE-1/2/L, BERTScore F1, embedding-based relevance.  
   - **QA**: SQuAD Exact-Match & F1.  
   - Achieved ▶ ROUGE-1 ≈ 0.35, BERTScore F1 ≈ 0.84, Relevance ≈ 0.55; QA EM ≈ 0.0 (proof-of-concept).

4. **Deployment**  
   - Gradio app that:  
     1. Accepts raw transcript.  
     2. Runs on-device summarizer.  
     3. Calls fine-tuned QA model for on-the-fly QA.  
     4. Returns summary + answers to user’s questions.
     5. Returns summary + 5 QA pairs if user didn't enter a question.

---

## 💼 Business Impact & Applications

Modern organizations lose valuable time manually documenting and distributing key insights from meetings. Our pipeline delivers a scalable solution for automating meeting intelligence:

### ✅ Executive Benefits

| Outcome                     | Business Value                                                                 |
|----------------------------|--------------------------------------------------------------------------------|
| 📝 Auto-summarized meetings | Frees up hours of manual note-taking for PMs, analysts, and execs              |
| 🤖 Instant follow-up Q&A    | Enables searchable “AI meeting assistants” with contextual memory              |
| 💬 Alignment & traceability | Ensures everyone is on the same page — key decisions, blockers, action items   |
| 📉 Reduced meeting fatigue  | Allows async stakeholders to consume content faster (via summary or Q&A)       |
| 🛠️ Integratable             | Can be embedded in CRM, PM, or video conferencing tools for seamless deployment|

### 🧠 Use Cases

- **Post-Meeting Recap**: Auto-generated summaries posted to Slack, Teams, or Notion.
- **Project Kickoff Logs**: Instantly extract who-decided-what from product and client meetings.
- **Compliance Tracking**: Archived Q&A logs for legal, HR, or financial audits.
- **Async Workflows**: Read a 200-word digest instead of watching a 60-minute call.

> By combining LLM-driven generation with lightweight fine-tuning, this project shows how open-source models can power real-world enterprise tools that enhance **productivity, transparency, and decision velocity**.

---

## 📦 Data & Labeling Process

We built our dataset from scratch using the public **AMI Meeting Corpus**:

- 🗂️ Source: Annotated meeting transcripts (~100 full multi-speaker dialogues)
- 📄 Transcript Length: Ranged from 1,500 to 16,000 tokens each

### ✍️ Summary Creation

Used OpenAI API to generate "enhanced" summaries (~200–250 words) via carefully engineered prompts:

- Focused on decisions, participants, blockers, financials, and action items
- Summaries were checked for logical completeness and tone

### ❓ QA Pair Generation

From each transcript+summary pair, we prompted the LLM to create 5 question–answer pairs in strict JSON format:

- Enforced output schema: `{ "question": ..., "answer": ... }`
- Implemented fallback answers if parsing failed

### 🧹 Dataset Statistics

| Component      | Count |
|----------------|------:|
| Meeting transcripts | ~100 |
| Summaries           | ~100 |
| QA pairs            | ~500 |
| Final fine-tune rows (after exploding JSON) | ~1,000 |

This dataset enabled us to fine-tune two downstream tasks:
- Summarization: sequence-to-sequence generation (T5)
- QA: extractive SQuAD-style answer prediction (DistilBERT & T5)

> Despite modest scale, the dataset enabled tangible improvements on standard metrics. Future iterations can scale using automatic LLM validation and synthetic data augmentation.

---

## 🛠 Methodology & Technical Approach

1. **Data Generation with OpenAI API**  
   - Prompt engineering to enforce: purpose, participants, decisions, action-items, financials.  
   - Fallback logic for malformed JSON → generic QA pairs.

2. **Summarization Fine-Tuning**  
   - Base model: `google/flan-t5-base`  
   - Input: raw transcript (≤ 512 tokens)  
   - Target: 200–250 word paragraph  
   - Trainer: Hugging Face `Seq2SeqTrainer` (4.51.3)  
   - Metrics: ROUGE, BERTScore, embedding-based cosine relevance.

3. **QA Fine-Tuning**  
   - Base model: `distilbert-base-uncased-distilled-squad`  
   - Create SQuAD-style dataset via exploding JSON QA pairs.  
   - Sliding‐window tokenization (384 max, stride 128).  
   - Trainer: HF `Trainer` with SQuAD Exact-Match & F1.

4. **Inference Pipeline & Gradio UI**  
   - Load fine-tuned summarizer locally for offline deployment.  
   - Use fine-tuned QA model for ad-hoc questions.  
   - Simple `gradio.Interface` for copy-paste transcripts and question inputs.

---

## 📊 Key Results & Findings

| Metric                | Value     |
|-----------------------|----------:|
| ROUGE-1 (summaries)   | 0.3457    |
| ROUGE-2               | 0.1117    |
| ROUGE-L               | 0.2488    |
| BERTScore F1          | 0.8383    |
| Embedding Relevance   | 0.5492    |
| QA Exact Match (EM)   | 0.0       |
| QA F1                 | 0.0       |

> **Note**: QA fine-tuning on ~1 K QA examples is proof-of-concept—EM and F1 will improve with more data and multi-task training.

---

## ⚠️ Challenges Encountered

- **Token-length limits**: had to chunk & truncate 16 K-token transcripts.  
- **Brittle JSON prompts**: fallback QA logic needed to avoid empty outputs.  
- **Data scarcity**: only ~100 summary examples → risk of overfitting & repetition.  
- **Domain mismatch**: many base models pre-trained on news, not meeting language.

---

## 🔮 Future Work & Takeaways

- **Data Augmentation**: overlapping windows, synthetic transcripts, paraphrasing.  
- **Multi-Task Training**: jointly fine-tune summarization + QA in one model.  
- **Human Evaluation**: recruit raters for summary quality & question validity.  
- **Integration**: plug into Zoom/Teams for live post-meeting summaries & chatbot.  
- **Domain Adaptation**: fine-tune on corporate-specific meeting data for improved fidelity.

---

## ▶️ How to Run

### 🔧 Prerequisites

- Python 3.8+
- NVIDIA GPU (recommended for training)
- Bash terminal (for running shell script blocks)

### 🧩 Step-by-Step

1. **Clone the repo**  
   ```bash
   git clone https://github.com/vaishj91/meeting_summary_and_qa/tree/main
   cd meeting-summarizer-qa

2. **Place your dataset**
   Make sure the file `ami_gpt35_multitask.jsonl` exists in the project root.
   This file should be in JSONL format with the following structure per line:
   ```bash
   {"input": "<meeting transcript>", "output": "<summary or QA content>"}
   ```

4. **Run fine-tuning and launch the app**
   Rename the pipeline script if needed and execute:
   ```bash
   mv 03.sh run_pipeline.sh
   bash run_pipeline.sh
   ```

4. **Use the Web UI**
   After training, a Gradio app will launch at:
   ```bash
   http://localhost:7860
   ```
   You can:
   - Paste a raw meeting transcript
   - Ask a custom question about the content (optional)
   - Receive a summary and either:
     1. 5 auto-generated QA pairs
     2. A direct answer to your question

---

## 📁 File Structure

```bash
├── 01_dataset_generation.ipynb        # Generates synthetic summaries and QA pairs using GPT
├── 02_training_and_eval.ipynb         # Trains and evaluates
├── 03_fine-tuning_and_gradio.ipynb    # Fine-tunes model, loads it, and launches Gradio interface [entrypoint]
├── ami_gpt35_multitask.jsonl          # Serialized multitask dataset (summaries + QA)
└── README.md
