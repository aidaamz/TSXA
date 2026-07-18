from sklearn.feature_extraction.text import CountVectorizer

with open('Data_1.txt', 'r', encoding='utf-8') as file:
    corpus = file.read()

print("\nBasic Tokenization using CountVectorizer:")   
print("\nText from Data_1.txt:")
print(corpus)

vectorizer = CountVectorizer(lowercase=True, stop_words='english')

analyzer = vectorizer.build_analyzer()

tokens = analyzer(corpus)

print("\nTokens after preprocessing:")
print(tokens)




from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path

text = Path('Data_1.txt').read_text(encoding='utf-8')

# create a list of documents with CountVectorizer
corpus = [text]

vectorizer = CountVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1, 2), # single words + two-word phrases
    min_df=1, # keep words appearing in at least 1 document
)

# Tokenize and build vocabaulary and count words
X = vectorizer.fit_transform(corpus)

tokens = vectorizer.build_analyzer()(text)
features = vectorizer.get_feature_names_out()

print("\nAdvanced Feature Extraction using CountVectorizer:")
print("\nTokens after preprocessing:")
print(tokens)

print("\nVocabulary:")
print(features)

print("\nWord counts:")
for word, count in zip(features, X.toarray()[0]):
    print(f"{word}: {count}")