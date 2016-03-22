import nltk
from nltk.probability import FreqDist

nltk.download('punkt')

def build_vocabulary(data_file):
	
	input_file = open(data_file, "r")
	input_file_contents = input_file.read()

	words = nltk.tokenize.word_tokenize(input_file_contents, 'english')
	fdist = FreqDist(words)
	print(fdist)
	# print(fdist.most_common(2000))

	output = open("vocabulary.txt", "w")
	

	for word, frequency in fdist.most_common(2000):
            if frequency >= 2:
                output.write(word + "\n")

	output.close()
	return 1

if build_vocabulary("data_without_stopwords.txt") :
	print("Vocabulary builded successfully!")
