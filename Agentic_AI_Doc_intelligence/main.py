from fastapi import FastAPI, File, UploadFile
from utils.file_handlers import handle_upload
from models.ai_agent import ai_agent

app = FastAPI()

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    content = await handle_upload(file)
    return {"filename": file.filename, "content": content}

@app.post("/ask/")
async def ask_question(question: str, filename: str, method: str = "gpt-j"):
    # Retrieve the file content (from memory, database, or cache)
    file_content = "Retrieve file content based on filename"  # Pseudo-code
    # Use the AI Agent to process the query
    answer = ai_agent.process_query(question, context=file_content, method=method)
    return {"answer": answer}
