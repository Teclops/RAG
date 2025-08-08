# HackRX RAG System

**Description**  
A Retrieval-Augmented Generation (RAG) system for querying insurance policy documents using FastAPI and Google’s Generative AI.

## Requirements

Install dependencies:
```bash
python -m venv venv
source venv/bin/activate       # (or `venv\Scripts\activate` on Windows)
pip install -r requirements.txt
````

## Configuration

Create a `.env` file in your project root and define:

```ini
# Place API keys, env vars here
# Example:
GOOGLE_API_KEY=your_key
```

## Running the App

Start the server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Endpoints

* `POST /hackrx/run` — Submit a PDF URL and questions; returns answers.
* `GET /health` — Health check.
* `GET /` — API metadata.
* `GET /docs` — Interactive Swagger UI.

## Usage Example

```bash
curl -X POST http://localhost:8000/hackrx/run \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "documents": "https://example.com/policy.pdf",
        "questions": ["What is the waiting period?", "What is the coverage limit?"]
      }'
```

## License

Add your license here (e.g., MIT).

## Contributing

Contributions are welcome—please open an issue or submit a PR!

````

This will render perfectly on GitHub without any “Copy/Edit” artifacts.  

If you want, I can now give you the **clean `requirements.txt`** so that running  
```bash
pip install -r requirements.txt
````
