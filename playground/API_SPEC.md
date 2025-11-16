# Playground API Specification

This document specifies the API for the Indic Tokenization Lab playground UI.

## Overview

The playground will provide a web-based interface for:
- Interactive tokenizer comparison
- Real-time tokenization visualization
- Metric display and scorecards
- Failure case browsing

## API Endpoints

### 1. Tokenize Text

**Endpoint:** `POST /api/tokenize`

**Request:**
```json
{
  "text": "यहाँ आपका हिंदी वाक्य जाएगा।",
  "tokenizer_ids": ["indicbert", "mbert", "gpe_hi_v0"],
  "lang": "hi"
}
```

**Response:**
```json
{
  "text": "यहाँ आपका हिंदी वाक्य जाएगा।",
  "lang": "hi",
  "results": [
    {
      "tokenizer_id": "indicbert",
      "tokenizer_name": "AI4Bharat IndicBERT",
      "tokens": ["यहाँ", "आपका", "हिंदी", "वाक्य", "जाएगा", "।"],
      "stats": {
        "num_tokens": 6,
        "chars": 26,
        "chars_per_token": 4.33,
        "grapheme_violations": 0
      }
    }
  ]
}
```

### 2. Evaluate Tokenizers

**Endpoint:** `POST /api/evaluate`

**Request:**
```json
{
  "text": "यहाँ आपका हिंदी वाक्य जाएगा।",
  "tokenizer_ids": ["indicbert", "mbert"],
  "lang": "hi"
}
```

**Response:**
```json
{
  "text": "यहाँ आपका हिंदी वाक्य जाएगा।",
  "results": [
    {
      "tokenizer_id": "indicbert",
      "metrics": {
        "fertility": 1.0,
        "chars_per_token": 4.33,
        "grapheme_violations": 0,
        "grapheme_violation_rate": 0.0
      }
    }
  ]
}
```

### 3. List Available Tokenizers

**Endpoint:** `GET /api/tokenizers`

**Response:**
```json
{
  "tokenizers": [
    {
      "id": "indicbert",
      "name": "AI4Bharat IndicBERT",
      "type": "hf",
      "available": true
    }
  ]
}
```

### 4. Load Curated Examples

**Endpoint:** `GET /api/examples?lang=hi&domain=news`

**Response:**
```json
{
  "examples": [
    {
      "text": "यहाँ आपका हिंदी वाक्य जाएगा।",
      "lang": "hi",
      "domain": "news",
      "tags": ["compound"]
    }
  ]
}
```

### 5. Generate Scorecard

**Endpoint:** `POST /api/scorecard`

**Request:**
```json
{
  "texts": ["text1", "text2"],
  "tokenizer_ids": ["indicbert", "mbert"],
  "lang": "hi"
}
```

**Response:**
```json
{
  "scorecard": {
    "timestamp": "2025-01-01T00:00:00",
    "results": {
      "indicbert": {
        "avg_fertility": 1.2,
        "avg_chars_per_token": 4.5,
        "total_grapheme_violations": 0
      }
    }
  }
}
```

## Implementation Notes

### Backend Options

1. **FastAPI** (Recommended)
   - Modern, fast, async support
   - Automatic API documentation
   - Type validation

2. **Flask**
   - Simple, lightweight
   - Good for prototyping

3. **CLI-based**
   - Simple HTTP server wrapper
   - Call Python scripts via subprocess

### Frontend Options

1. **Next.js** (Recommended)
   - React-based
   - Server-side rendering
   - Good TypeScript support

2. **Vanilla JS**
   - Simple, no build step
   - Good for prototyping

3. **Streamlit**
   - Python-based UI
   - Quick prototyping
   - Good for demos

## Data Flow

```
User Input (Text)
    ↓
Frontend (UI)
    ↓
API Endpoint
    ↓
Python Backend
    ├─→ Load Tokenizers (Registry)
    ├─→ Tokenize Text
    ├─→ Calculate Metrics
    └─→ Return Results
    ↓
Frontend (Display)
    ├─→ Token Visualization
    ├─→ Metrics Display
    └─→ Comparison Tables
```

## Future Enhancements

- WebSocket support for real-time updates
- Batch processing for large texts
- Export functionality (JSON, CSV, Markdown)
- Visualization of token boundaries
- Interactive grapheme violation highlighting
- Comparison diff view

