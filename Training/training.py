import math
import vocabulary
import stopwords
from nltk.sentiment.vader import NEGATE



def multinomial_naive_bayes_unigram(training_file, test_file, data, stop_words):
    
    stopwords.remove_stopwords(data, stop_words)
    vocabulary.build_vocabulary("data_without_stopwords.txt")
    
    vocab_file = open("vocabulary.txt", "r")
    
    vocab_count = 0
    for word in vocab_file :
        vocab_count += 1
    
    vocab_count -= 2
    
    separate_superdoc(training_file, stop_words)
    
    vocabulary.build_vocabulary("positive.txt")
    positive_superdoc = open("vocab_freq.txt", "r")
    positive_vocab = {"_pos":0}
    
    for each_line in positive_superdoc:
        words_in_line = each_line.split()
        positive_vocab[words_in_line[0]] = words_in_line[2]
        
    positive_superdoc.close()
    
    vocabulary.build_vocabulary("negative.txt")
    negative_superdoc = open("vocab_freq.txt", "r")
    negative_vocab = {"_neg":0}
    
    for each_line in negative_superdoc:
        words_in_line = each_line.split()
        negative_vocab[words_in_line[0]] = words_in_line[2]
        
    negative_superdoc.close()
    
    stopwords.remove_stopwords(test_file, stop_words)
    test_handler = open("data_without_stopwords.txt","r")
    tp=0
    tn=0
    fp=0
    fn=0
    review_number=1;
    for each_review in test_handler:
        FLAG = 0;
        words_in_review = each_review.split()
        if words_in_review[0] == "+":
            FLAG = 1
        else:
            FLAG = 0
                
        prob_pos_given_review = predict_positive(words_in_review, positive_vocab, negative_vocab, vocab_count)
        prob_neg_given_review = predict_negative(words_in_review, positive_vocab, negative_vocab, vocab_count)
        REVIEW = 0
        
        if prob_pos_given_review < prob_neg_given_review:
            REVIEW = 0
        else:
            REVIEW = 1
        
        if FLAG==1 and REVIEW==1:
            tp=tp+1
        elif FLAG==1 and REVIEW==0:
            fn=fn+1
        elif FLAG==0 and REVIEW==1:
            fp=fp+1
        else:
            tn=tn+1
        
        print(str(review_number) + ". original = " + str(FLAG) +" prediction = " + str(REVIEW)+"\n")
        review_number = review_number+1
        
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    print(accuracy)                                    
    return

def predict_positive(words_in_review, positive_vocab, negative_vocab, vocab_count):
    prob_pos = math.log2(float(positive_vocab["+"])/(float(positive_vocab["+"]) + float(negative_vocab["-"])))
    
    probability = prob_pos
    total_pos_vocab=0
    
    for key in positive_vocab:
        total_pos_vocab = total_pos_vocab + float(positive_vocab[key])
    
    total_pos_vocab = total_pos_vocab - float(positive_vocab["+"])
        
    for i in range(len(words_in_review)):
        if i==0:
            continue
        count=0
        if words_in_review[i] in positive_vocab:
            count = float(positive_vocab[words_in_review[i]])
            
        prob_word_given_pos = math.log2((count + 1)/(total_pos_vocab + vocab_count))
        probability = probability + prob_word_given_pos
    return probability     


def predict_negative(words_in_review, positive_vocab, negative_vocab, vocab_count):
    
    prob_neg = math.log2(float(negative_vocab["-"])/(float(positive_vocab["+"]) + float(negative_vocab["-"])))
    
    probability = prob_neg
    total_neg_vocab=0
    
    for key in negative_vocab:
        total_neg_vocab = total_neg_vocab + float(negative_vocab[key])
    
    total_neg_vocab = total_neg_vocab - float(negative_vocab["-"])
        
    for i in range(len(words_in_review)):
        if i==0:
            continue
        count=0
        if words_in_review[i] in negative_vocab:
            count = float(negative_vocab[words_in_review[i]])
            
        prob_word_given_neg = math.log2((count + 1)/(total_neg_vocab + vocab_count))
        probability = probability + prob_word_given_neg
    return probability     






def separate_superdoc(training_file, stop_words):
    stopwords.remove_stopwords(training_file, stop_words)
    training_without_stopwords = open("data_without_stopwords.txt", "r")
    positive_superdoc = open("positive.txt", "w")
    negative_superdoc = open("negative.txt", "w")
    
    for line in training_without_stopwords:
        words_in_line = line.split();
        
        FLAG = 0
        
        for word in words_in_line :
            if word == '+' :
                positive_superdoc.write(word)
                FLAG = 1
                continue
            if word == '-' :
                negative_superdoc.write(word)
                FLAG = 0
                continue
            if FLAG == 1 :
                positive_superdoc.write(" " + word)
            else :
                negative_superdoc.write(" " + word)
            
        if FLAG == 1 :
            positive_superdoc.write("\n")
        else :
            negative_superdoc.write("\n")
            
    positive_superdoc.close()
    negative_superdoc.close()   
    return

# data = input("Enter the name of data file: ")
# stop_words = input("Enter the name of stopwords file : ")

data = "data.txt"
stop_words = "stopwords.txt"

multinomial_naive_bayes_unigram(data, data, data, stop_words)
#separate_superdoc(data, stop_words)
print("Separating Done!!")