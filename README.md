# 🔎 Mini Search Engine

A lightweight information retrieval system built using Python and Streamlit.

This project demonstrates the fundamental concepts behind a search engine,
including text preprocessing, inverted indexing, TF-IDF, cosine similarity,
document ranking, and search analytics.

---

## 📌 Features

- 📄 Text document indexing
- 🔤 Tokenization
- 🚫 Stop-word removal
- 🌱 Porter stemming
- 📚 Inverted index
- 🔍 Query processing
- 📈 TF-IDF ranking
- 📐 Cosine similarity
- 🏆 Ranked search results
- 📝 Query-aware snippets
- 📤 `.txt` document upload
- 🔄 Automatic index rebuilding
- 🔀 AND / OR search modes
- 📊 Search statistics
- 📈 Search analytics
- 🔎 Index explorer
- 🕘 Search history
- 🖥️ Streamlit web interface

---

## 🎯 Project Objective

The objective of this project is to understand how a basic information
retrieval system works by implementing the major stages of a search engine
from scratch.

The system takes a collection of text documents, processes them, creates
an inverted index, and retrieves and ranks documents according to their
relevance to a user's query.

---

## 🧠 How It Works

The search engine follows this pipeline:

```text
Documents
    ↓
Tokenization
    ↓
Stop-word Removal
    ↓
Stemming
    ↓
Inverted Index
    ↓
TF-IDF
    ↓
Document Vectors
    ↓
User Query
    ↓
Query Processing
    ↓
Query Vector
    ↓
Cosine Similarity
    ↓
Ranking
    ↓
Search Results