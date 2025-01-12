from transformers import AutoTokenizer
from transformers import pipeline

summarizer = None


def split_text_into_chunks(text: str, max_tokens: int = 1000) -> list:
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    sentences = [sentence.strip() for sentence in text.split(". ") if sentence.strip()]
    sentence_tokens = [tokenizer.encode(sentence, add_special_tokens=False) for sentence in sentences]
    total_tokens = sum(len(tokens) for tokens in sentence_tokens)

    if total_tokens <= max_tokens:
        return [" ".join(sentences)]

    no_of_chunks = total_tokens // max_tokens + 1
    avg_tokens_per_chunk = total_tokens // no_of_chunks
    print(f"Splitting into {no_of_chunks} chunks with {avg_tokens_per_chunk} tokens each")

    chunks = []
    chunk = []
    current_tokens = 0
    for sentence, tokens in zip(sentences, sentence_tokens):
        if current_tokens + len(tokens) > avg_tokens_per_chunk:
            chunks.append(" ".join(chunk))
            chunk = []
            current_tokens = 0
        chunk.append(sentence)
        current_tokens += len(tokens)

    return chunks

def download_summary_model():
    global summarizer
    print("Downloading 'facebook/bart-large-cnn'...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    print("Model downloaded!")


def summarize_text(text: str) -> str:
    global summarizer
    print(f"Lenght of text: {len(text)}")
    chunks = split_text_into_chunks(text)
    summaries = []
    print(f"Summarizing {len(chunks)} chunks...")
    for chunk in chunks:
        print(f"Chunk len: {len(chunk)}")
        summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
        summaries.append(summary[0]["summary_text"])
    return " ".join(summaries)

if __name__ == '__main__':
    download_summary_model()
    print("Model loaded successfully!")