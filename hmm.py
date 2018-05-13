import math
from sys import argv

tagDict = {}    # Dictionary of words linked to a list of tags seen with that word in the training data
wordCounts = {} # Dictionary of words and word/tag pairs linked to their counts found in the training data
tagCounts = {}  # Dictionary of tags and tag/prevTag pairs linked to their counts found in the training data
allTags = []    # List of all tags (excluding the ### tag) seen in the training data
singlestt = {}  # Dictionary of tags linked to the number of tag/tag singletons found in the training data
singlestw = {}  # Dictionary of tags linked to the number of word/tag singletons found in the training data

#Given a list of word/tag pairs, collect the appropriate counts to fill
#the global variables
def trainHMM(trainWords):
	global tagDict
	global wordCounts
	global tagCounts
	global allTags
	global singlestt
	global singlestw
	prevTag = ""

	#For each word/tag pair in the list of training words, collect the counts needed
	#to fill the global variables
	for wordTag in trainWords:
		wordTagList = wordTag.split("/")
		word = wordTagList[0]
		tag = wordTagList[1]
		#Fill wordCounts with word/tag pair and adjust singlestw counts associated
		#with the current tag
		if tag not in singlestw:
			singlestw[tag] = 0
		if wordTag in wordCounts:
			wordCounts[wordTag] = wordCounts[wordTag] + 1
			if wordCounts[wordTag] == 2:
				singlestw[tag] = singlestw[tag] - 1
		else:
			wordCounts[wordTag] = 1
			singlestw[tag] = singlestw[tag] + 1
		#Fill the list of all possible tags
		if tag not in allTags and tag != "###":
			allTags.append(tag)
		#Fill wordCounts with the current word
		if word in wordCounts:
                        wordCounts[word] = wordCounts[word] + 1
                else:
                        wordCounts[word] = 1
		#Fill tagCounts with the current tag
		if tag in tagCounts:
                        tagCounts[tag] = tagCounts[tag] + 1
                else:
                        tagCounts[tag] = 1
		#Fill tagCounts with the tag/prevTag pair and adjust singlestt conuts associated
		#with the previous tag
		if prevTag != "":
			tagTag = tag + "/" + prevTag
			if prevTag not in singlestt:
				singlestt[prevTag] = 0
			if tagTag in tagCounts:
				tagCounts[tagTag] = tagCounts[tagTag] + 1
				if tagCounts[tagTag] == 2:
					singlestt[prevTag] = singlestt[prevTag] - 1
			else:
				tagCounts[tagTag] = 1
				singlestt[prevTag] = singlestt[prevTag] + 1
		#Fill tag dictionary with the possible tags asscoiated with the current word
		if word not in tagDict:
			tagDict[word] = [tag]
		if tag not in tagDict[word]:
			tagDict[word].append(tag)
		prevTag = tag

#Given a list of test words, a list of train words, and the vocabulary of the trainging data,
#it guesses the tag sequence that generated the words from the testWords list using the viterbi method.  
#It then computes and prints the accuracy of the model
def testHMM(testWords, trainWords, vocab):
	wordSequence = []
	tagSequence = []
	correctTags = 0
	totalTags = 0
	correctKnown = 0
	correctNovel = 0
	totalNovel = 0
	totalKnown = 0

	#For each word/tag paur in the test words list, creates two different lists,
	#one for the tag sequence, the other for the words.
	for wordTag in testWords:
		wordSequence.append(wordTag.split("/")[0])
		tagSequence.append(wordTag.split("/")[1])

	guessTagSequence = viterbi(wordSequence, len(trainWords), len(vocab)) 

	#For each guessed tag, check to see if we guessed correctly
	for i in range(0, len(guessTagSequence)):
		if guessTagSequence[i] != "###" and tagSequence[i] != "###":
			totalTags = totalTags + 1
			if wordSequence[i] not in wordCounts:
				totalNovel = totalNovel + 1
			else:
				totalKnown = totalKnown + 1
			print("Guessed '" + guessTagSequence[i] + "' when it was supposed to be '" + tagSequence[i] + "'.")
			if guessTagSequence[i] == tagSequence[i]:
				correctTags = correctTags + 1
				if wordSequence[i] not in wordCounts:
					correctNovel = correctNovel + 1
				else:   
					correctKnown = correctKnown + 1
	#Makes sure we aren't dividing by zero
	if totalNovel == 0:
		totalNovel = 1
	if totalKnown == 0:
		totalKnown = 1

	print("Tagging accuracy (Viterbi decoding): " + str(round((float(correctTags)/float(totalTags)) * 100, 2)) + "% (known: " + str(round((float(correctKnown)/float(totalKnown)) * 100, 2)) + "% novel: " + str(round((float(correctNovel)/float(totalNovel)) * 100, 2)) + "%)")

#Given a list of observations, the size of the training data, and the size of the vocabulary,
#guess a sequence of tags that could produce the given observation sequence using the Viterbi method.
def viterbi(words, trainSize, vocabSize):
	global tagDict
	global tagCounts
	global wordCounts
	global allTags
	guessTags = []
	bestPathProbs = {} #Stores both the Viterbi probability at each node, and the probability of the guessed tag sequence at each time i
	backpointer = {}

	#For each possbile word/tag pair in the given list of words, initialize the 
	#their Viterbi probabilities to negative infinity
	for i in range(1, len(words)):
		if words[i] in wordCounts:
			for tag in tagDict[words[i]]:
				bestPathProbs[str(i) + "/" + tag] = float("-inf")
		else:
			for tag in allTags:
				bestPathProbs[str(i) + "/" + tag] = float("-inf")
		bestPathProbs[str(i)] = float("-inf")
	bestPathProbs["0/###"] = math.log(1, 2)
	bestPathProbs[str(len(words)-1) + "/###"] = float("-inf")

	#For each word in the words list, find the most likely tag to produce it given its context
	for i in range(1, len(words)):
		temp = allTags
		if (words[i] in tagDict):
			temp = tagDict[words[i]]
		for tag_i in temp: 
			temp2 = allTags
			if (words[i-1] in tagDict):
				temp2 = tagDict[words[i-1]]
			for prevTag_i in temp2:
				pttBackoff = float(tagCounts[tag_i]) / float(trainSize)
				ptwBackoff = float(1) / float(trainSize + vocabSize)
				if (words[i] in wordCounts):
					ptwBackoff = float(wordCounts[words[i]] + 1) / float(trainSize + vocabSize)
				transProb = math.log(pttBackoff * (1 + singlestt[prevTag_i]) / (float(tagCounts[prevTag_i] + (1 + singlestt[prevTag_i]))), 2)
				emisProb = math.log(ptwBackoff * (1 + singlestw[tag_i]) / (float(tagCounts[tag_i] + (1 + singlestw[tag_i]))), 2)

				if ((tag_i + "/" + prevTag_i) in tagCounts):
					transProb = math.log((float(tagCounts[tag_i + "/" + prevTag_i] + (1 + singlestt[prevTag_i]) * pttBackoff)) / float(tagCounts[prevTag_i] + (1 + singlestt[prevTag_i])), 2)
				if ((words[i] + "/" + tag_i) in wordCounts):
					if (words[i] == "###" and tag_i == "###"):
						emisProb = math.log(1, 2)
					else:
						emisProb = math.log((float(wordCounts[words[i] + "/" + tag_i] + (1 + singlestw[tag_i]) * ptwBackoff)) / (float(tagCounts[tag_i] + (1 + singlestw[tag_i]))), 2)
				prob = transProb + emisProb
				currPathProb = bestPathProbs[str(i-1) + "/" + prevTag_i] + prob
				if currPathProb > bestPathProbs[str(i)]:
					bestPathProbs[str(i)] = currPathProb
					backpointer[i] = prevTag_i
				if currPathProb >= bestPathProbs[str(i) + "/" + tag_i]:
					bestPathProbs[str(i) + "/" + tag_i] = currPathProb
	guessTags.append("###")

	#For each tag in the backpointer, append it to the list of guessed tags.
	for i in range(len(words) - 1, 0, -1):
		guessTags.append(backpointer[i])
	guessTags.reverse() 

	return guessTags

#Given a list of training words, collect the vocabulary from those words unioned
#with the words in the enraw.txt file
def getVocab(train):
	raw = open("data/en/enraw.txt")
	vocabList = []
	setList = {}

	#For each word in the raw data, append it to the vocab list
	for word in raw:
		word = word[:-1]
		vocabList.append(word)
	#For each word in the training data, append it to the vocab list
	for word in train:
		word = word.split("/")
		vocabList.append(word[0])
	
	vocabList.append("OOV")
	setList = set(vocabList) #Compute the union

	return setList

# main
def main():
	#Setup files
	trainFile = open("data/" + argv[1][0] + argv[1][1] + "/" + argv[1])
        testFile = open("data/" + argv[2][0] + argv[2][1] + "/" + argv[2])
	#Get words
	trainWords = trainFile.read().split()
	testWords = testFile.read().split()
	#Get vocabulary from the training and raw data
	vocab = getVocab(trainWords)
	
	trainHMM(trainWords[:len(trainWords)-1])

	testHMM(testWords, trainWords, vocab)

	trainFile.close()
        testFile.close()
main()
