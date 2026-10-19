# 🔍 LangSearch AI

An AI-powered search assistant that combines the intelligence of Large Language Models with trusted knowledge sources like **Wikipedia** and **ArXiv**.

Ask questions naturally and receive accurate, AI-generated answers with real-time information retrieval.

---

## 🌐 Live Demo

```bash
https://langsearch-chatbot-codertheashish.vercel.app/
```

---

## ✨ Features

- 🤖 AI-Powered Conversational Assistant
- 🔍 Wikipedia Search Integration
- 📚 ArXiv Research Paper Search
- 💬 Interactive Chat Interface
- ⚡ Fast Responses with Groq LLM
- 🧠 Tool-Calling Agent Loop (Wikipedia + ArXiv)
- 📝 Chat History Support
- 🔐 API Key Kept Server-Side (never exposed to the browser)
- 🎨 Clean & Responsive UI (Light / Dark mode)
- ☁️ One-Click Deploy on Vercel
- 💻 Easy to Run Locally

---

## 🔍 Search Sources

- 📖 Wikipedia
- 📄 ArXiv Research Papers
- 🤖 Groq LLM

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Backend Logic |
| Flask | Serverless API (`/api/chat`) |
| HTML / CSS / JavaScript | Chat Interface |
| Groq API (GPT-OSS 120B) | Large Language Model |
| Wikipedia API | Knowledge Retrieval |
| ArXiv API | Research Paper Search |
| Vercel | Hosting & Deployment |

---

## 📂 Project Structure

```
LangSearch-AI/
│
├── api/
│   └── chat.py
├── public/
│   └── index.html
├── requirements.txt
├── vercel.json
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Clone Repository

```bash
git clone https://github.com/codertheashish/LangSearch-Chatbot.git
```

### Go to Project Folder

```bash
cd LangSearch-Chatbot
```

### Install Vercel CLI

```bash
npm i -g vercel
```

### Add Groq API Key

Create a **.env** file and add:

```env
GROQ_API_KEY=your_groq_api_key
```

> The key is read only on the server. Optionally set `GROQ_MODEL` to override the default model (`openai/gpt-oss-120b`).

### Run Application

```bash
vercel dev
```

Open `http://localhost:3000` in your browser.

---

## ☁️ Deploy on Vercel

1. Push this project to GitHub.
2. Go to [vercel.com](https://vercel.com) → **Add New** → **Project**.
3. Import your repository and click **Deploy** (no build settings needed).
4. **Required:** add `GROQ_API_KEY` in **Project Settings → Environment Variables** (Production), then **Redeploy** — env changes only apply to new deployments.

---

## 💡 Example Questions

- What is Machine Learning?
- Explain Deep Learning.
- Search research papers on Generative AI.
- What is LangChain?
- Latest AI research on LLMs.

---

## 📱 Responsive Design

Optimized for:

- 💻 Desktop
- 📱 Mobile
- 📟 Tablets

---

## ⚠️ Requirements

- Python 3.11+
- Node.js (for Vercel CLI, local run only)
- Groq API Key
- Internet Connection

---

## 🔮 Future Enhancements

- 🌍 Web Search Integration
- 📄 PDF Question Answering
- 📁 Multiple Document Support
- 💾 Chat Export
- 🎙️ Voice Assistant
- 🌐 Multi-Language Support
- 📊 Search Analytics

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome.

Feel free to fork this repository and submit a pull request.

---

## 👨‍💻 Author

**Ashish Kumar Prajapati**

- GitHub :
[codertheashish](https://github.com/codertheashish)
- LinkedIn :
[codertheashish](https://www.linkedin.com/in/codertheashish/)
- Instagram :
[codertheashish](https://www.instagram.com/codertheashish/)

---

⭐ If you like this project, don't forget to give it a star on GitHub.