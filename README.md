# 📝 Automated Handwritten Assignment Evaluator
### AI-Powered Assessment Using Computer Vision, NLP & Agentic AI

An AI-powered system that automatically evaluates handwritten student assignments by extracting text from handwritten images, comparing answers against reference solutions, and generating marks and feedback using Natural Language Processing and semantic similarity.

The project also integrates an **AI Agent-based verification mechanism** to identify potentially underestimated answers and award additional marks when contextual correctness is detected.

---

## 🚀 Project Overview

Traditional assignment evaluation requires teachers to manually read handwritten submissions, compare them with model answers, and assign marks.

This project aims to automate and assist this process through a multi-stage AI pipeline.

The system accepts:

1. A handwritten student assignment (PNG/JPG).
2. A model answer or grading key (PDF).
3. A configurable maximum assignment score.

It then processes the submission through OCR, NLP preprocessing, semantic similarity analysis, automated grading, and AI-based contextual verification.

### Key Objectives

- Extract handwritten text using Optical Character Recognition.
- Improve image quality through computer vision preprocessing.
- Clean and normalize extracted text using NLP.
- Compare student answers with model answers using semantic similarity.
- Automatically assign marks based on similarity scores.
- Generate meaningful evaluation feedback.
- Use an AI verification agent to identify potentially underestimated answers.

---

## 🧠 System Architecture

The project follows a modular, multi-stage assessment pipeline.

```text
       Student Handwritten Assignment
                    |
                    v
          Image Preprocessing
             (OpenCV)
                    |
                    v
             Tesseract OCR
                    |
                    v
          Extracted Handwritten Text
                    |
                    v
             NLP Processing
       (Regex, Tokenization, Stopwords,
             Spell Correction)
                    |
                    v
           Semantic Similarity
            (Sentence-BERT)
                    |
                    v
            Cosine Similarity
                    |
                    v
           Automated Grading
                    |
                    v
            AI Agent Verification
           (DistilBERT-MNLI)
                    |
                    v
             Feedback Generation
                    |
                    v
             Final Evaluation
```

## 🔍 Core Modules

### 1. Computer Vision & OCR

The first stage converts handwritten assignment images into machine-readable text.

**Technologies:**
- OpenCV
- Tesseract OCR
- Pytesseract
- NumPy

**Image preprocessing includes:**

- Image loading and validation.
- Grayscale conversion.
- Gaussian Blur for noise reduction.
- Adaptive Gaussian Thresholding for image binarization.

The processed image is passed to Tesseract OCR using its configured page segmentation mode (`PSM 6`) to extract text.

### 2. NLP Text Processing

OCR output may contain spelling mistakes, unnecessary characters, and formatting inconsistencies.

The NLP module cleans and prepares the extracted text for semantic analysis.

**Processing steps:**

- Regex-based text cleaning.
- Removal of unwanted special characters.
- Whitespace normalization.
- Tokenization using NLTK.
- English stopword removal.
- Spelling correction using PySpellChecker.
- Reconstruction of the processed text.

### 3. Semantic Similarity Analysis

Traditional keyword matching may fail to recognize answers that express the same concept using different words.

To address this, the project uses Sentence-BERT to generate contextual sentence embeddings.

**Model:** `all-MiniLM-L6-v2`

The system:

1. Converts the student answer into an embedding.
2. Converts the model answer into an embedding.
3. Computes cosine similarity between the two representations.
4. Produces a similarity score between 0 and 1.

Cosine similarity:

\[
\text{Similarity}(A,B)=
\frac{A\cdot B}{\|A\|\|B\|}
\]

This helps identify conceptual similarity rather than relying exclusively on exact keyword matches.

### 4. Automated Grading System

The similarity score is converted into marks using a predefined grading rubric.

| Similarity Score | Grade | Marks Awarded |
|---|---|---|
| ≥ 0.85 | Excellent | 100% |
| ≥ 0.65 | Good | 75% |
| ≥ 0.40 | Average | 50% |
| ≥ 0.20 | Poor | 25% |
| < 0.20 | Fail | 0% |

The maximum assignment score is configurable.

For example, for a 10-mark question:

- Similarity 0.90 → 10 marks
- Similarity 0.70 → 7.5 marks
- Similarity 0.45 → 5 marks
- Similarity 0.25 → 2.5 marks
- Similarity 0.15 → 0 marks

### 5. Agentic AI – Contextual Grade Verification

One of the key features of this project is its AI Agent-based verification mechanism.

A low semantic similarity score does not necessarily mean that a student's answer is completely incorrect. A student may express the correct concept using different terminology or sentence structures.

To address this issue, the system integrates a local zero-shot classification model.

**Model:** `typeform/distilbert-base-uncased-mnli`

**Agent verification workflow:**

1. The grading system identifies answers classified as Average or Poor.
2. The AI Agent receives the student's original OCR-cleaned answer and the model answer.
3. A zero-shot classifier checks whether the student's response is contextually related to the reference answer.
4. The classifier evaluates two candidate labels:
   - The model answer.
   - Completely unrelated topic.
5. If the model-answer label receives confidence above 60%, the Agent considers the answer eligible for grade rescue.
6. An additional 2.5 marks are awarded, subject to the maximum score.

**Example:**

```text
Initial Evaluation
       |
       v
Similarity Score = 0.45
       |
       v
Grade = Average
       |
       v
AI Agent Verification
       |
       v
DistilBERT Context Classification
       |
       v
Confidence > 60% ?
       |
    Yes / No
       |
       v
Award Additional 2.5 Marks
       |
       v
Updated Evaluation
```

The Agent's purpose is to provide a secondary contextual check rather than relying exclusively on numerical similarity.

### 6. Feedback Generation

The system generates a feedback summary based on the evaluation category.

The Streamlit interface displays the final awarded marks, similarity score, and feedback.

It also provides an expandable processing-details section containing:

- Raw OCR output.
- Cleaned NLP output.
- Extracted model answer.
- Evaluation information.
- Agent verification notes, when applicable.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Computer Vision | OpenCV, NumPy |
| OCR | Tesseract, Pytesseract |
| NLP | NLTK, PySpellChecker |
| Semantic Similarity | Sentence-BERT |
| Transformer Models | Hugging Face Transformers |
| AI Agent | DistilBERT-MNLI |
| PDF Processing | PyPDF2 |
| Frontend | Streamlit |
| Image Processing | Pillow |
| Machine Learning Utilities | Scikit-learn |

## 📂 Project Structure

```text
automatic_handwritten_assignment_evaluator/
│
├── data/
│   ├── model_answers.json
│   ├── sample_images/
│   └── processed_images/
│
├── src/
│   ├── ocr_pipeline.py
│   ├── nlp_processing.py
│   ├── similarity.py
│   ├── scoring.py
│   ├── feedback.py
│   ├── agent_verifier.py
│   ├── pdf_utils.py
│   └── pipeline_runner.py
│
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Yugoku225/automatic_handwritten_assignment_evaluator.git

cd automatic_handwritten_assignment_evaluator
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**macOS / Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR

Tesseract must also be installed separately on your system.

**macOS**

```bash
brew install tesseract
```

**Ubuntu / Debian**

```bash
sudo apt install tesseract-ocr
```

**Windows**

Install Tesseract using its Windows installer and ensure its installation directory is added to the system PATH.

### 5. Download NLTK Resources

Run Python and execute:

```python
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

### 6. Run the Application

Launch the Streamlit interface:

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

---

## 💻 Application Workflow

1. Open the Streamlit application.
2. Upload the model answer PDF.
3. Upload the student's handwritten assignment image.
4. Specify the maximum assignment score.
5. Click **Evaluate Assignment**.
6. Wait for OCR extraction, NLP processing, and semantic analysis.
7. View the awarded marks and similarity score.
8. Review the generated feedback.
9. Expand processing details to inspect intermediate outputs and Agent verification notes.

## 🔬 Key Learning Outcomes

This project provided practical experience with:

- Developing an end-to-end Computer Vision pipeline.
- Integrating OCR into an NLP application.
- Working with contextual embeddings using Sentence-BERT.
- Implementing cosine similarity for semantic comparison.
- Designing rule-based automated grading systems.
- Integrating transformer-based zero-shot classification.
- Building an Agent-assisted verification mechanism.
- Developing an interactive AI application using Streamlit.
- Designing modular and reusable Python pipelines.

## 🔮 Future Enhancements

- Fine-tune TrOCR specifically for handwritten student assignments.
- Implement line-level handwriting segmentation for improved OCR accuracy.
- Support multi-page PDF answer sheets.
- Introduce question-wise answer extraction and evaluation.
- Develop rubric-based marking instead of relying only on overall similarity.
- Integrate advanced semantic evaluation using LLMs.
- Add teacher review and manual mark correction.
- Generate downloadable assessment reports.
- Support multilingual handwritten answers.
- Introduce confidence-based human verification.

## ⚠️ Limitations

- OCR accuracy depends on handwriting quality and image clarity.
- The current Streamlit workflow accepts a single handwritten image per evaluation.
- PDF model-answer extraction requires selectable text; scanned PDFs may need an additional OCR stage.
- Semantic similarity is not equivalent to factual correctness.
- The Agent's contextual verification is a secondary classification check and does not guarantee that an answer deserves additional marks.
- The current grading rubric is predefined and may require customization for different subjects.

**Disclaimer:** This project is an academic prototype intended to assist educational assessment. Final academic grading should involve human verification.

---

## 👨‍💻 Author

**Yugant Raj**

B.Tech – Computer Science

GitHub: [@Yugoku225](https://github.com/Yugoku225)

[Project Repository](https://github.com/Yugoku225/automatic_handwritten_assignment_evaluator)

---

⭐ If you find this project interesting, consider starring the repository!
