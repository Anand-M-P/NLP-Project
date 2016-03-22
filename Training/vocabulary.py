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

	output_file1 = open("vocabulary.txt", "w")
	output_file2 = open("vocab_freq.txt", "w")
	
	for word, frequency in fdist.most_common(2000):		
            if frequency >= 2 :
                output_file1.write(word + "\n")
                output_file2.write(word + " : " + str(frequency) + "\n")
                


	output_file1.close()
	output_file2.close()
	return 1

# if build_vocabulary("data_without_stopwords.txt") :
# 	print("Vocabulary builded successfully!")
