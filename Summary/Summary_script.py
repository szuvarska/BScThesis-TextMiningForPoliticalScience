from transformers import pipeline

summarizer = None


def download_summary_model():
    global summarizer
    print("Downloading 'facebook/bart-large-cnn'...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    print("Model downloaded!")


def summarize_text(text: str) -> str:
    global summarizer
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]["summary_text"]
