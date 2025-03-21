
# AI Resume Screening & Ranking System


## Overview
This AI-powered resume screening and ranking system uses Natural Language Processing (NLP) and Machine Learning models to analyze and rank resumes based on their relevance to a given job description. It supports both **TF-IDF** and **BERT similarity** for ranking.

## Features
- Extracts text from **PDF and DOCX** resumes, including image-based PDFs via OCR.
- Uses **TF-IDF** and **BERT** for computing similarity scores.
- Ranks resumes based on **normalized scores (10-100 scale)**.
- Provides a **search functionality** to filter resumes by name or keyword.
- Allows **downloading** of ranked results as CSV.
- Converts **DOCX resumes to PDF** for easy access.

## Video of Project :  
https://youtu.be/XIvfQNL8eMU?si=9JnOeLYm1NU-knwz

## Installation
To run this project locally, follow these steps:

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Pip package manager
- Virtual environment (optional but recommended)

### Clone the Repository
```sh
git clone https://github.com/yourusername/AI-Resume-Ranking.git
cd AI-Resume-Ranking
```

### Create a Virtual Environment (Optional)
```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

## Usage
Run the Streamlit application using the following command:
```sh
streamlit run app.py
```

### How to Use
1. **Enter a Job Description** in the provided text area.
2. **Upload Resumes (PDF/DOCX)**.
3. Click **Process** to rank resumes.
4. **View ranked results** based on relevance.
5. Use **Search Resumes** to find specific resumes.
6. **Download results** in CSV format.

## Technologies Used
- **Python** (Backend Processing)
- **Streamlit** (Web UI)
- **spaCy** (NLP Processing)
- **Sentence Transformers (BERT)** (Semantic Similarity)
- **TF-IDF** (Text Vectorization)
- **PyMuPDF & Pytesseract** (PDF Processing & OCR)
- **Pandas, NumPy, Scikit-Learn** (Data Handling & ML)

## Folder Structure
```
AI-Resume-Ranking/
│── images/                # UI images
│── app.py                 # Main Streamlit application
│── requirements.txt       # Python dependencies
│── README.md              # Documentation
│── data/                  # (Optional) Sample resumes
└── models/                # (Optional) Pretrained NLP models
```

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Author
Developed by **Vivek Phadol** Feel free to contribute and improve this project!
