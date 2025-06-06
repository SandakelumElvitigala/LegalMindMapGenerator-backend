# 🧠 Legal Mind Map Generator

A FastAPI-based backend service that extracts structured graph data (entities, events, clues, places, and relationships) from uploaded legal documents (PDF/DOCX) using the Groq API and LLMs. The output is a node-edge graph representing legal case details, ideal for visualization as mind maps.

---

## 📦 Features

* ✅ Upload legal documents in **PDF** or **DOCX** format
* 🧠 Extracts:

  * Entities (people, organizations, laws, courts, etc.)
  * Incidents (events, filings, crimes)
  * Clues (evidence, testimonies, documents)
  * Places, relationships, case metadata
* 📊 Returns a structured **JSON graph** with nodes and edges
* ⚙️ Integrates with **Groq's LLM API** (`llama3-70b-8192`)
* 🧪 Text pre-processing and robust JSON parsing
* 📁 Stores files and graph data in-memory (extendable to DB)
* 🔒 CORS support for frontend integration

---

## 🚀 Getting Started

### Prerequisites

* Python 3.9+
* Groq API Key (LLM access)

### Clone the Repository

```bash
git clone https://github.com/SandakelumElvitigala/LegalMindMapGenerator-backend.git
cd LegalMindMapGenerator-backend
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Setup Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> ✅ The `load_dotenv()` will load this key automatically.

---

## ▶️ Running the Server

```bash
uvicorn main:app --reload
```

---

## 📂 API Endpoints

### 1. **Upload Document**

**POST** `/upload/`

**FormData:**

* `file`: Upload `.pdf` or `.docx`

**Response:**

```json
{
  "graph_id": "uuid",
  "graph": {
    "nodes": [...],
    "edges": [...]
  }
}
```

---

### 2. **Get Graph by ID**

**GET** `/graph/{graph_id}`

Returns graph data for a given ID.

---

### 3. **Update Graph**

**PUT** `/graph/{graph_id}`

**Body:**

```json
{
  "graph_id": "uuid",
  "nodes": [...],
  "edges": [...]
}
```

---

## 🧰 Project Structure

```
.
├── main.py                 # FastAPI application entry point
├── utils.py                # Custom JSON cleaners, validators, sanitizers
├── uploads/                # Uploaded files are stored here
├── .env                    # API key for Groq (excluded from version control)
└── requirements.txt        # Python dependencies
```

---

## 📦 Dependencies

* `fastapi` + `uvicorn` - API framework and server
* `httpx` - Async HTTP requests
* `python-dotenv` - Load API key
* `PyPDF2` - Extract text from PDF
* `docx2txt` - Extract text from DOCX
* `uuid`, `os`, `tempfile`, `logging`, `json`, etc.

---

## 🛡️ Error Handling

* Invalid file types → `400 Bad Request`
* Groq API issues → `500/502/504`
* JSON parsing issues → `422`
* Missing graphs → `404`

---

## 📈 Future Enhancements

* Persistent database (PostgreSQL, TimescaleDB, etc.)
* Frontend visualization (React + vis.js or Cytoscape)
* User authentication
* Versioned graph history
* Support for additional formats (TXT, HTML, etc.)

---

## 📄 License


