from textblob import TextBlob

with open("Data_1.txt", "r", encoding="utf-8") as file:
    text = file.read()


blob = TextBlob(text)
tokens = blob.words

print("=== TextBlob Tokenization ===")
print(tokens)

print("\nNumber of tokens:", len(tokens))

print("\nFirst 20 tokens:")
print(tokens[:20])
