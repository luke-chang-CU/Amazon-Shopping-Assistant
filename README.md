# Amazon-Shopping-Assistant

### Environment Setup

1. **Clone the Repository**

```bash
git clone https://github.com/luke-chang-CU/Amazon-Shopping-Assistant.git
cd Amazon-Shopping-Assistant
```

---

2. **Set Up Your Environment Variables**

This project uses environment variables for sensitive credentials like API keys.
We provide a `.env-example` file as a template.

* Create your own `.env` file by copying the example:

```bash
cp .env-example .env
```

* Open `.env` and fill in your API keys or other secrets. For example:

```
OPENAI_API_KEY=your_real_key_here
```

> ⚠️ **Important:** Do **not** commit your `.env` file to Git. It's excluded via `.gitignore`.

---

3. **Set Up a Virtual Environment**

To avoid dependency conflicts, it's recommended to use a Python virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

---

4. **Install Required Packages**

We manage dependencies with `requirements.txt`. To install:

```bash
pip install -r requirements.txt
```

---
5. **Download the Dataset**

The project uses a CSV dataset which you can download with gdown:

```bash
gdown https://drive.google.com/uc?id=1nkCURmybXFdQY0FxbOcKb_KGy5QB4TDo -O data/Amazon2023DS_partial_NLP.csv
```

---

6. **Run the App**

After setting up your environment and installing dependencies, you can run the project:

```bash
streamlit run app.py
```

Or version without OpenAI API

```
streamlit run app_offline.py
```

---


