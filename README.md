# 🧠 Smart Invention Validator

An AI-powered assistant to help inventors, analysts, and IP professionals validate invention novelty, extract keywords, generate patent claims, compare with prior art, and predict classification codes (IPC/CPC).

---

## 🚀 Features

* ✍️ **Abstract Input** – Manual entry with smart feedback
* 🔑 **Keyword Extraction** – Extracts key terms via KeyBERT and LLM fallback
* 🔍 **Patent Search** – Uses Google Patents for relevant prior art
* 🧠 **Semantic Similarity** – Highlights overlap with existing inventions
* 📜 **Claim Generation** – Auto-generates structured patent claims via LLM
* ⚖️ **Risk Scoring** – Assesses novelty, length, similarity & coverage
* 🔮 **IPC/CPC Prediction** – Predicts classification codes and meanings
* 🌳 **IPC/CPC Tree View** – Visual hierarchy and code co-occurrence matrix
* 📦 **Summary & Export** – Downloadable report with abstract, codes, claims

---

## 🧱 Tech Stack

| Layer          | Technology Used                                       |
| -------------- | ----------------------------------------------------- |
| Frontend       | Streamlit / Gradio (Gradio version optional)          |
| NLP & AI       | KeyBERT, Sentence-Transformers, Together AI (Mixtral) |
| Patent Data    | Google Patents Search API                             |
| Classification | Custom logic + code descriptions from JSON            |
| Deployment     | Hugging Face Spaces / Local Streamlit                 |

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/smart-invention-validator.git
cd smart-invention-validator
pip install -r requirements.txt
```

### 🔐 API Keys (Local only)

Create `.streamlit/secrets.toml`:

```toml
[together]
api_key = "your-together-api-key"
```

**For cloud deployment**, set this as an environment variable:

```
TOGETHER_API_KEY = your-key-here
```

---

## 🧪 Run the App

```bash
streamlit run app.py
```

For Gradio version:

```bash
python app_gradio.py
```

---

## 📁 Project Structure

```
├── app.py                  # Main Streamlit interface
├── app_gradio.py          # Optional Gradio version
├── requirements.txt
├── data/                  # IPC/CPC definitions
├── modules/               # Keyword, claim, prediction logic
├── .streamlit/
│   └── secrets.toml       # API key for Together AI (local only)
```

---

## 🙏 Acknowledgements

* [Together AI](https://www.together.ai/) for free LLM access
* [KeyBERT](https://github.com/MaartenGr/KeyBERT) for keyword extraction
* [Google Patents](https://patents.google.com) for open patent search data

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
