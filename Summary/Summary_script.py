from transformers import pipeline
from transformers import AutoTokenizer
summarizer = None

def split_text_into_chunks(text: str, max_tokens: int = 1000) -> list:
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    sentences = []
    current_chunk = []
    current_length = 0
    for sentence in text.split(". "):
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_length = len(sentence_tokens)
        if current_length + sentence_length > max_tokens:
            sentences.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    if current_chunk:
        sentences.append(" ".join(current_chunk))
    return sentences

def download_summary_model():
    global summarizer
    print("Downloading 'facebook/bart-large-cnn'...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    print("Model downloaded!")


def summarize_text(text: str) -> str:
    global summarizer
    chunks = split_text_into_chunks(text)
    summaries = []
    print(f"Summarizing {len(chunks)} chunks...")
    for chunk in chunks:
        summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
        summaries.append(summary[0]["summary_text"])
    return " ".join(summaries)

if __name__ == '__main__':
    download_summary_model()
    print("Model loaded successfully!")