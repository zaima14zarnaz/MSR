import csv
import re
from transformers import CLIPTokenizer

descriptions_csv = "/home/zaimaz/Desktop/research1/QAGNet/Dataset/IRSR_ASSR/train.csv"
MAX_WORDS = 30

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

def split_sentences(text):
    text = text.strip()
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]

num_captions = 0
total_phrases = 0
total_words = 0
total_tokens = 0
truncated_count = 0
max_observed_phrase_length_before_truncation = 0

with open(descriptions_csv, "r", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)

    for row in reader:
        caption = row[2]
        phrases = split_sentences(caption)

        num_captions += 1
        total_phrases += len(phrases)

        for phrase in phrases:
            orig_words = phrase.split()
            orig_word_len = len(orig_words)
            orig_token_len = len(tokenizer(phrase, add_special_tokens=True)["input_ids"])

            trimmed_phrase = " ".join(orig_words[:MAX_WORDS])
            trimmed_word_len = min(orig_word_len, MAX_WORDS)
            trimmed_token_len = len(tokenizer(trimmed_phrase, add_special_tokens=True)["input_ids"])

            total_words += trimmed_word_len
            total_tokens += trimmed_token_len
            truncated_count += int(orig_word_len > MAX_WORDS)
            max_observed_phrase_length_before_truncation = max(
                max_observed_phrase_length_before_truncation,
                orig_token_len
            )

avg_phrases_per_caption = total_phrases / num_captions if num_captions else 0
avg_words_per_phrase = total_words / total_phrases if total_phrases else 0
avg_tokens_per_phrase = total_tokens / total_phrases if total_phrases else 0
avg_tokens_per_caption = total_tokens / num_captions if num_captions else 0
pct_phrases_truncated = 100 * truncated_count / total_phrases if total_phrases else 0

print(f"Number of captions: {num_captions}")
print(f"Total phrases: {total_phrases}")
print(f"Average phrases per caption: {avg_phrases_per_caption:.4f}")
print(f"Average words per phrase: {avg_words_per_phrase:.4f}")
print(f"Average tokens per phrase: {avg_tokens_per_phrase:.4f}")
print(f"Average tokens per caption: {avg_tokens_per_caption:.4f}")
print(f"Percentage of phrases truncated: {pct_phrases_truncated:.2f}%")
print(f"Maximum observed phrase length before truncation (tokens): {max_observed_phrase_length_before_truncation}")