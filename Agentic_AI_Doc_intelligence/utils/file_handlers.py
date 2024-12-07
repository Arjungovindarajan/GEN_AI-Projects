from fastapi import UploadFile
from utils.text_extractors import extract_text_from_pdf, extract_text_from_docx, extract_text_from_csv, extract_text_from_excel, extract_text_from_txt

async def handle_upload(file: UploadFile):
    if file.content_type == 'application/pdf':
        content = await extract_text_from_pdf(file)
    elif file.content_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        content = await extract_text_from_docx(file)
    elif file.content_type in ['text/csv', 'application/vnd.ms-excel']:
        content = await extract_text_from_csv(file)
    elif file.content_type == 'text/plain':
        content = await extract_text_from_txt(file)
    else:
        content = await extract_text_from_txt(file)  # Default to text for unknown types
    return content
