import pdfminer
from pdfminer.high_level import extract_text
from docx import Document
import pandas as pd

async def extract_text_from_pdf(file):
    content = extract_text(file.file)
    return content

async def extract_text_from_docx(file):
    doc = Document(file.file)
    content = '\n'.join([p.text for p in doc.paragraphs])
    return content

async def extract_text_from_csv(file):
    df = pd.read_csv(file.file)
    content = df.to_string()
    return content

async def extract_text_from_excel(file):
    df = pd.read_excel(file.file)
    content = df.to_string()
    return content

async def extract_text_from_txt(file):
    content = file.file.read().decode('utf-8')
    return content
