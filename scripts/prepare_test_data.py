import os
import urllib.request
import json
from pathlib import Path
import docx

DATA_DIR = Path("tests/evaluation/definitive_test_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_file(url: str, filename: str):
    path = DATA_DIR / filename
    if not path.exists():
        print(f"Downloading {filename}...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(path, 'wb') as out_file:
            out_file.write(response.read())
    else:
        print(f"{filename} already exists.")
    return path

def create_docx(filename: str):
    path = DATA_DIR / filename
    if not path.exists():
        print(f"Generating {filename}...")
        doc = docx.Document()
        doc.add_heading('Project Alpha Operations Manual', 0)
        doc.add_heading('Section 1: Budget and Planning', level=1)
        doc.add_paragraph('Project Alpha is the flagship initiative for Q3. The total allocated budget for this project is strictly capped at $4.2 million USD. Any expenditures exceeding this must be approved by the board.')
        
        doc.add_heading('Section 2: Timeline', level=1)
        doc.add_paragraph('The kickoff is scheduled for October 15th. We expect the beta launch to occur by January 10th of the following year.')
        doc.add_paragraph('The team lead for the beta launch is Robert Oppenheimer, based in the New York office.')
        doc.save(path)
    return path

def create_audio(filename: str):
    path = DATA_DIR / filename
    if not path.exists():
        print(f"Generating {filename}...")
        from gtts import gTTS
        tts = gTTS('This is a spoken audio test for the Motif RAG system. The system should correctly process this sentence and answer questions about it. The secret code word is Antigravity.')
        tts.save(str(path))
    return path

def main():
    download_file("https://arxiv.org/pdf/1810.04805.pdf", "bert_paper.pdf")
    download_file("https://raw.githubusercontent.com/facebook/react/main/README.md", "react_readme.md")
    download_file("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv", "titanic.csv")
    download_file("https://en.wikipedia.org/wiki/Machine_learning", "machine_learning.html")
    download_file("https://upload.wikimedia.org/wikipedia/en/thumb/8/80/Wikipedia-logo-v2.svg/500px-Wikipedia-logo-v2.svg.png", "wikipedia_logo.png")
    
    create_docx("project_alpha.docx")
    create_audio("audio_test.mp3")

    ground_truth = [
        # PDF (BERT Paper)
        {
            "query": "In the BERT paper, what does BERT stand for?",
            "expected_answer": "Bidirectional Encoder Representations from Transformers",
            "format": "pdf"
        },
        {
            "query": "According to the BERT paper abstract, what is the size of the BERT_BASE model in terms of parameters?",
            "expected_answer": "110M parameters",
            "format": "pdf"
        },
        {
            "query": "What two pre-training tasks are used for BERT?",
            "expected_answer": "Masked language model (MLM) and next sentence prediction (NSP)",
            "format": "pdf"
        },
        # MD (React README)
        {
            "query": "According to the React README, how can you add React to an HTML page?",
            "expected_answer": "You can add React to an HTML page with a <script> tag.",
            "format": "md"
        },
        {
            "query": "What is the primary URL for React's documentation?",
            "expected_answer": "https://react.dev/",
            "format": "md"
        },
        {
            "query": "Does React use a Virtual DOM?",
            "expected_answer": "Yes, React relies on a Virtual DOM for efficient updates.",
            "format": "md"
        },
        # CSV (Titanic Dataset)
        {
            "query": "In the Titanic dataset, what is the passenger class (Pclass) for the passenger named 'Cumings, Mrs. John Bradley'?",
            "expected_answer": "1",
            "format": "csv"
        },
        {
            "query": "Did passenger 'Braund, Mr. Owen Harris' survive?",
            "expected_answer": "No, he did not survive (Survived=0).",
            "format": "csv"
        },
        {
            "query": "How many siblings/spouses aboard (SibSp) did 'Heikkinen, Miss. Laina' have?",
            "expected_answer": "0",
            "format": "csv"
        },
        # HTML (Machine Learning Wikipedia)
        {
            "query": "In the Machine Learning HTML document, how is machine learning broadly defined?",
            "expected_answer": "Machine learning is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data.",
            "format": "html"
        },
        {
            "query": "What are the three main categories of machine learning?",
            "expected_answer": "Supervised learning, unsupervised learning, and reinforcement learning.",
            "format": "html"
        },
        {
            "query": "According to the Machine Learning HTML, who coined the term 'machine learning'?",
            "expected_answer": "Arthur Samuel in 1959.",
            "format": "html"
        },
        # Image (Wikipedia Logo)
        {
            "query": "What words are written on the Wikipedia logo image?",
            "expected_answer": "The Free Encyclopedia",
            "format": "image"
        },
        {
            "query": "Is the word 'WIKIPEDIA' in all caps on the Wikipedia logo?",
            "expected_answer": "Yes",
            "format": "image"
        },
        {
            "query": "What language is the Wikipedia logo representing in English?",
            "expected_answer": "English",
            "format": "image"
        },
        # Audio (gTTS Audio Test)
        {
            "query": "In the audio test, what is the secret code word?",
            "expected_answer": "Antigravity",
            "format": "audio"
        },
        {
            "query": "What is the spoken audio testing?",
            "expected_answer": "The Motif RAG system.",
            "format": "audio"
        },
        {
            "query": "Does the speaker expect the system to process the sentence correctly?",
            "expected_answer": "Yes.",
            "format": "audio"
        },
        # DOCX (Project Alpha)
        {
            "query": "What is the total allocated budget for Project Alpha?",
            "expected_answer": "$4.2 million USD",
            "format": "docx"
        },
        {
            "query": "When is the beta launch for Project Alpha scheduled to occur?",
            "expected_answer": "January 10th of the following year.",
            "format": "docx"
        },
        {
            "query": "Who is the team lead for the beta launch?",
            "expected_answer": "Robert Oppenheimer",
            "format": "docx"
        }
    ]

    with open(DATA_DIR / "ground_truth.json", "w") as f:
        json.dump(ground_truth, f, indent=4)
        
    print(f"Test data and {len(ground_truth)} ground truth questions prepared successfully.")

if __name__ == "__main__":
    main()
