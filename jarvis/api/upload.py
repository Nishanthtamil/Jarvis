from __future__ import annotations
import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from docx import Document
from tools.pdf import extract_pdf_file
from store.db import add_uploaded_file, list_uploaded_files, delete_uploaded_file
from store.vector import add_documents_to_collection, delete_collection_items, _chunk_text
from llm.client import count_tokens

router = APIRouter(prefix="/files", tags=["files"])

UPLOAD_COLLECTION = "user_uploads"

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1].lower()
    
    # Ensure tmp directory exists
    os.makedirs("/tmp", exist_ok=True)
    tmp_path = f"/tmp/{file_id}{ext}"
    
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    try:
        content = ""
        if ext == ".pdf":
            result = extract_pdf_file(tmp_path)
            if not result["ok"]:
                raise HTTPException(status_code=400, detail=f"PDF extraction failed: {result['error']}")
            content = result["markdown"]
        elif ext == ".docx":
            try:
                doc = Document(tmp_path)
                content = "\n".join([p.text for p in doc.paragraphs])
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"DOCX extraction failed: {e}")
        elif ext in [".txt", ".md"]:
            try:
                with open(tmp_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Text extraction failed: {e}")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        token_count = count_tokens(content)
        chunks = _chunk_text(content)
        
        if chunks:
            await add_documents_to_collection(
                collection_name=UPLOAD_COLLECTION,
                documents=chunks,
                metadatas=[{"file_id": file_id, "filename": file.filename}]
            )
        
        await add_uploaded_file(file_id, file.filename, token_count, UPLOAD_COLLECTION)
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "token_count": token_count,
            "status": "indexed"
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@router.get("")
async def list_files():
    files = await list_uploaded_files()
    return files

@router.delete("/{file_id}")
async def delete_file(file_id: str):
    await delete_collection_items(UPLOAD_COLLECTION, {"file_id": file_id})
    await delete_uploaded_file(file_id)
    return {"status": "deleted"}
