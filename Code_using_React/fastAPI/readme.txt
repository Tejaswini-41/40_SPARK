python environment:

pythom -m venv venv
venv\Scripts\activate

pip install -r requirements.txt


to run:
uvicorn main:app --reload --port 8000


Swagger UI: http://127.0.0.1:8000/docs
