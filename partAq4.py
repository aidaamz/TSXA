import nltk
from nltk.util import bigrams
from nltk.util import ngrams
from nltk.util import everygrams
from nltk.util import pad_sequence
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE, Laplace


# Defining the training corpus here as three separate sentences,
# Matching the three sentences given in Data_3.txt
text1 = "He read a book"
text2 = "I read a different book"
text3 = "He read a book by Danielle"

# This is the sentence that needs to calculate the probability for
test_text = "I read a book by Danielle"

# Tokenizing each sentence into individual words so NLTK can work with them
token1 = nltk.tokenize.word_tokenize(text1)
token2 = nltk.tokenize.word_tokenize(text2)
token3 = nltk.tokenize.word_tokenize(text3)
test_tokens = nltk.tokenize.word_tokenize(test_text)

#Adding padding to the tokens
# Adding <s> and </s> tags to the start and end of each sentence here.
# The actual model training below pads everything internally on its own.
padded_token1 = list(pad_both_ends(token1, n=2))
padded_token2 = list(pad_both_ends(token2, n=2))
padded_token3 = list(pad_both_ends(token3, n=2))
padded_test_tokens = list(pad_both_ends(test_tokens, n=2))

print("Padded tokens for text1:", padded_token1)
print("Padded tokens for text2:", padded_token2)
print("Padded tokens for text3:", padded_token3)
print("Padded tokens for test_text:", padded_test_tokens)

#Unsmoothed model
# Setting n=2 here so every model built below is a bigram model
n = 2

#Building MLE model
# Using padded_everygram_pipeline to automatically pad the training sentences
# generating the n-grams + vocabulary needed to train the model.
# MLE is the unsmoothed model - it computes probabilities purely as count(bigram)/count(unigram),
# with no adjustment for unseen bigrams.
train_data, vocab_data = padded_everygram_pipeline(n, [token1, token2, token3])
model = MLE(n)
model.fit(train_data, vocab_data)

#Build Laplace smoothed model
# I need a brand new pipeline call here because train_data and vocab_data above
# already got consumed (used up) when I called model.fit() for the MLE model.
# If I reused the same generators, this Laplace model would train on nothing.
# Laplace smoothing adds 1 to every bigram count and |V| to every denominator,
# so unseen bigrams don't end up with a probability of zero.
trained_data_laplace, vocab_data_laplace = padded_everygram_pipeline(n, [token1, token2, token3])
model_laplace = Laplace(n)
model_laplace.fit(trained_data_laplace, vocab_data_laplace)

# Converting the padded test sentence into a list of (context, word) bigram pairs here,
#  so that each pair can be scored individually by the model in the loop below.
list_of_bigrams = list(bigrams(padded_test_tokens))
print("\n------------------------------")
print("Bigrams for test_text:", list_of_bigrams)
print("------------------------------\n")


# Setting the running product at 1 (not 0) because have to multiply probabilities together,
sentence_prob_mle = 1
print("Calculating the probability of the test sentence using MLE:")
for context, word in list_of_bigrams:
    # Looping through every bigram pair generated above, one at a time.
    prob = model.score(word, [context]) #convert pair into a probability
    print("P(", word, "|", context, ") = ", prob)
    # Multiplying this probability into my running total, since the full sentence probability
    # This is the product of every individual bigram probability in the chain.
    sentence_prob_mle = sentence_prob_mle * prob

print("\nFinal sentence probability using MLE:", sentence_prob_mle)

#Calculating the probability of the test sentence using Laplace smoothing
# Repeating the exact same loop logic as above, but this time scoring with the Laplace-smoothed model
sentence_prob_laplace = 1
print("Calculating the probability of the test sentence using Laplace smoothing:")
for context, word in list_of_bigrams:
    prob = model_laplace.score(word, [context]) #convert pair into a probability
    print("P(", word, "|", context, ") = ", prob)
    sentence_prob_laplace = sentence_prob_laplace * prob

# Reformatting the final Laplace probability from scientific notation (e.g. 5.78e-06)
# into a plain decimal string with 10 decimal places, for my ease.
sentence_prob_laplace_formatted = "{:.10f}".format(sentence_prob_laplace)

print("\nFinal sentence probability using Laplace smoothing:", sentence_prob_laplace_formatted)

len_vocab = len(model.vocab)
print("\nVocabulary size:", len_vocab)

