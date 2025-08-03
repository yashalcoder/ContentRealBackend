import os, tempfile
from app.utils.extractors import extract_text_from_pdf, extract_text_from_docx

async def handle_file_upload(file, fileType, youtubeUrl):
    print("in file controlelr")
    if fileType == "youtube" and youtubeUrl:
        return {
            "status": "success",
            "source": "youtube",
            "url": youtubeUrl,
            "text": "This is dummy text extracted from YouTube audio."
        }

    if file:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        if suffix == ".pdf":
            extracted_text = extract_text_from_pdf(tmp_path)
        elif suffix == ".docx":
            extracted_text = extract_text_from_docx(tmp_path)
        elif suffix == ".txt":
            with open(tmp_path, 'r', encoding='utf-8') as f:
                extracted_text = f.read()
        else:
            extracted_text = "Unsupported file format."

        os.remove(tmp_path)

        return {
            "status": "success",
            "source": "file",
            "filename": file.filename,
            "fileType": fileType,
            "text": extracted_text
        }

    return {"status": "error", "message": "No file or URL provided"}
