import nltk
import string
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.stem import RegexpStemmer, PorterStemmer, LancasterStemmer

nltk.download("punkt")
nltk.download("punkt_tab")

# reading 
with open("Data_1.txt", "r", encoding="utf-8") as file:
    text = file.read()

print("Original Text:")
print(text)

tokens = word_tokenize(text) 

words = [word for word in tokens if word not in string.punctuation] 
regex_stemmer = RegexpStemmer("ing$|ed$|s$", min=4)
porter_stemmer = PorterStemmer()
lancaster_stemmer = LancasterStemmer()

regex_output = [regex_stemmer.stem(word) for word in words]
porter_output = [porter_stemmer.stem(word) for word in words]
lancaster_output = [lancaster_stemmer.stem(word) for word in words]

print("\nRegular Expression Stemmer Output:")
print(regex_output)

print("\nPorter Stemmer Output:")
print(porter_output)

print("\nLancaster Stemmer Output:")
print(lancaster_output)

df = pd.DataFrame({
    "Original Word": words,
    "Regex Stemmer": regex_output,
    "Porter Stemmer": porter_output,
    "Lancaster Stemmer": lancaster_output
})

print("\nComparison Table:")
print(df)

print(df.iloc[20:31])