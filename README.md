# 🩻 **Multimodal RAG for Chest X-rays (MIMIC-CXR)**

A **Multimodal Retrieval-Augmented Generation (RAG)** application for **chest X-ray analysis**, leveraging OpenAI’s Groq LLM and CLIP (Contrastive Language-Image Pretraining) models to retrieve and analyze similar MIMIC-CXR reports.

This project combines both **image** and **text** embeddings to provide clinically relevant interpretations of chest X-rays based on a user's question, incorporating retrieved similar cases to guide the LLM’s diagnostic analysis.

---

## 🚀 **Features**

- **Multimodal RAG Pipeline**: Uses both **image** and **text** inputs to retrieve similar chest X-ray reports and generate findings & impressions.
- **Integration with Groq LLM**: Harnesses Groq's LLM for generating detailed radiology reports from user queries.
- **MIMIC-CXR Dataset**: Built on a subset of the MIMIC-CXR dataset, containing annotated radiology reports for chest X-rays.
- **Fast Similarity Search**: Uses **FAISS** (Facebook AI Similarity Search) to efficiently find similar past radiology cases based on embeddings.

## 🔧 **Installation**

To run this project, you need to set up a Python environment with all dependencies installed. Follow the steps below:

### 1. Clone the Repository

```bash
git clone https://github.com/Noor-Fatima-Afzal/Chest-X-Ray-Analysis.git
cd Chest-X-Ray-Analysis```

---

## 📖 How to Use

1. **Upload a Chest X-ray**: Click on the "Upload a chest X-ray" button to upload a chest X-ray in JPEG or PNG format.
2. **Ask Your Question**: Type in a question about the uploaded image (e.g., "What abnormalities do you see in the X-ray?").
3. **Run the Analysis**: Click the "Run Multimodal RAG" button to retrieve similar cases and generate findings & impression using the Groq LLM.

## ⚙️ Configuration

You can customize the following settings via the sidebar:

- **Top-K Retrieved Reports**: Adjust the number of retrieved similar reports.
- **Groq Model**: Specify the Groq model name to use for generating responses.
- **API Key**: Manually input the GROQ API Key (optional).

## 🧠 How It Works

The pipeline consists of several steps:

1. **Image & Text Embedding**: Converts both the uploaded chest X-ray and user query into embeddings using the CLIP model.
2. **Similarity Search**: The embeddings are used to search for similar past cases from the MIMIC-CXR dataset using FAISS.
3. **LLM Generation**: The retrieved similar cases are sent to the Groq LLM, which generates a Findings and Impression for the user's query based on the past cases.

## 🛠️ Technologies Used

- **Streamlit**: For building the interactive web application.
- **CLIP Model**: For generating image-text embeddings.
- **Groq LLM**: For generating detailed findings and impressions.
- **FAISS**: For efficient similarity search.
- **Pandas**: For data manipulation.

## 🔒 Privacy & Security

The GROQ API Key must be securely set in Streamlit secrets or environment variables to ensure proper functionality of the app. Please avoid exposing your API key publicly.

## 📝 Contributing

We welcome contributions! Please feel free to fork the repository, submit issues, and create pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Acknowledgements

- **MIMIC-CXR dataset**: For providing the annotated radiology reports.
- **CLIP**: For powerful image-text embeddings.
- **Groq LLM**: For enabling text generation in the field of radiology.

