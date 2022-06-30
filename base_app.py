"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
#streamlit dependencies

import streamlit as st
import joblib, os
from PIL import Image
import json
import requests


## data dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from textblob import TextBlob
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
#from nlppreprocess import NLP # pip install nlppreprocess
#import en_core_web_sm
from nltk import pos_tag

# Imports for data visualisation
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file
# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """
	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Sentiment Classifier")
	st.header("Climate change tweet classification")
	
	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Tweet Classifier", "Information", "Analysis and Visuals", 'Contact App Developers']
	selection = st.sidebar.selectbox("Navigator", options)
	st.sidebar.image("Screenshot (219).png", use_column_width=True)
	# Building out the "Information" page
	if selection == "Information":
		st.subheader("Classification")
		# You can read a markdown file from supporting resources folder
		st.markdown("We, as human beings, make multiple decisions throughout the day. For example, when to wake up, what to wear, who to call, which route to take when traveling, how to sit, the list goes on and on. While several of these are repetitive and we do not usually take notice (and allow it to be done subconsciously), there are many others that are new and require conscious thought.  And we learn along the way. Businesses, similarly, apply their past learning to decision-making related to operations and new initiatives e.g. relating to customer classification, products, etc. However, it gets a little more complex here as there are multiple stakeholders involved. Additionally, the decisions need to be accurate owing to their wider impact. With the evolution in digital technology, humans have developed multiple assets; machines being one of them. We have learned (and continue) to use machines for analyzing data using statistics to generate useful insights that serve as an aid in making decisions and forecasts.")
		
		st.subheader("Logistic Regression")
		st.write("check out this [link](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regression#sklearn.linear_model.LogisticRegression)")
		
		st.subheader('Random Forest')
		st.write("check out this [link](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)")
		
		st.subheader('Naïve Bayes')
		st.write("check out this [link](https://towardsdatascience.com/naive-bayes-explained-9d2b96f4a9c0)")
		
		st.subheader('K-Nearest Neighbors')
		st.write("check out this [link](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html?highlight=knn#sklearn.impute.KNNImputer)")
		
		st.subheader('Passive Aggressive Classifier')
		st.write("check out this [link](https://thecleverprogrammer.com/2021/02/10/passive-aggressive-classifier-in-machine-learning/#:~:text=Passive%20Aggressive%20Classifier%20belongs%20to%20the%20category%20of,classifications%20and%20responding%20as%20aggressive%20for%20any%20miscalculation)")
		#st.markdown('Passive Aggressive Classifier belongs to the category of online learning algorithms in machine learning. It works by responding as passive for correct classifications and responding as aggressive for any miscalculation.')
		
		st.subheader('Bernoulli Naive Bayes')
		st.write("check out this [link](https://thecleverprogrammer.com/2021/07/27/bernoulli-naive-bayes-in-machine-learning/#:~:text=Bernoulli%20Naive%20Bayes%20is%20one%20of%20the%20variants,in%20the%20form%20of%20binary%20values%20such%20as%3A)")
		
		st.subheader('Stacking Classifier')
		st.write("check out this [link](http://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/#:~:text=Stacking%20is%20an%20ensemble%20learning%20technique%20to%20combine,of%20the%20individual%20classification%20models%20in%20the%20ensemble)")
		#st.markdown('Stacking is an ensemble learning technique to combine multiple classification models via a meta-classifier. The individual classification models are trained based on the complete training set; then, the meta-classifier is fitted based on the outputs -- meta-features -- of the individual classification models in the ensemble. ')
		

	# Building out the predication page
	if selection == "Tweet Classifier":
		st.subheader("Prediction with ML Models")
		
		st.markdown('Insert a climate change related tweet and see the predicted sentiment!')
		st.markdown('The focus of this section is to:')
		st.markdown('1. Insert a tweet in the textbox.')
		st.markdown('2. Choose one of the various models for prediction.')
		st.markdown('3. *Copy a tweet by clicking the checkbox below and copying tweet into textbox.')
		st.markdown('4. Learn if the tweet supports or opposes man-made climate chang.e')
		
		st.subheader('Copy Tweet!')
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['message', 'sentiment']]) # will write the df to the page

		# Creating a text box for user input
		tweet_text = st.text_area("Enter text below:")
		st.markdown('DISCLAIMER: Please ensure to use Ctrl + enter, to save your text for predictions.')
		
		if st.button("Logistic Regression Classifer"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/logreg.pkl"),"rb"))
			prediction = predictor.predict(vect_text)
			if prediction == 0:
				result = 'Neutral: the tweet neither supports nor refutes the belief of man-made climate change'
			elif prediction == 1:
				result = 'Pro: the tweet supports the belief of man-made climate change'
			elif prediction == 2:
				result = 'News: the tweet links to factual news about climate change'	
			else:
				result = 'Anti: the tweet does not believe in man-made climate change'		
			st.success(result)
			

		if st.button("Random Forest Classifier"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/rfc.pkl"),"rb"))
			prediction = predictor.predict(vect_text)
			if prediction == 0:
				result = 'Neutral: the tweet neither supports nor refutes the belief of man-made climate change'
			elif prediction == 1:
				result = 'Pro: the tweet supports the belief of man-made climate change'
			elif prediction == 2:
				result = 'News: the tweet links to factual news about climate change'	
			else:
				result = 'Anti: the tweet does not believe in man-made climate change'		
			st.success(result)

		if st.button("Linear Support Vector Classifier"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/lsvc_model.pkl"),"rb"))
			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			prediction = predictor.predict(vect_text)
			if prediction == 0:
				result = 'Neutral: the tweet neither supports nor refutes the belief of man-made climate change'
			elif prediction == 1:
				result = 'Pro: the tweet supports the belief of man-made climate change'
			elif prediction == 2:
				result = 'News: the tweet links to factual news about climate change'	
			else:
				result = 'Anti: the tweet does not believe in man-made climate change'		
			st.success(result)

		if st.button("Naive Bayes Classifier"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/naive_bayes.pkl"),"rb"))
			prediction = predictor.predict(vect_text)
			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			prediction = predictor.predict(vect_text)
			if prediction == 0:
				result = 'Neutral: the tweet neither supports nor refutes the belief of man-made climate change'
			elif prediction == 1:
				result = 'Pro: the tweet supports the belief of man-made climate change'
			elif prediction == 2:
				result = 'News: the tweet links to factual news about climate change'	
			else:
				result = 'Anti: the tweet does not believe in man-made climate change'		
			st.success(result)
		
		if st.button("Bernoulli Naive Bayes Classifier"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/BNBmodel.pkl"),"rb"))
			prediction = predictor.predict(vect_text)
			if prediction == 0:
				result = 'Neutral: the tweet neither supports nor refutes the belief of man-made climate change'
			elif prediction == 1:
				result = 'Pro: the tweet supports the belief of man-made climate change'
			elif prediction == 2:
				result = 'News: the tweet links to factual news about climate change'	
			else:
				result = 'Anti: the tweet does not believe in man-made climate change'		
			st.success(result)

		if st.button("Passive-Aggressive Classifier"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/passive_tf.pkl"),"rb"))
			prediction = predictor.predict(vect_text)
			if prediction == 0:
				result = 'Neutral: the tweet neither supports nor refutes the belief of man-made climate change'
			elif prediction == 1:
				result = 'Pro: the tweet supports the belief of man-made climate change'
			elif prediction == 2:
				result = 'News: the tweet links to factual news about climate change'	
			else:
				result = 'Anti: the tweet does not believe in man-made climate change'		
			st.success(result)


	# Building out the predication page
	if selection == "Analysis and Visuals":
		st.info("Visual Analysis")
		st.markdown('In this section we take a closer look at our data using visual analysis ')

		st.markdown('(1) 52.3% of the tweets support the belief of man-made climate change ')
		st.markdown('(2) 21.1% of the tweets link to factual news about climate change')
		st.markdown('(0) 17.6% of the tweets neither support nor refute the belief of man-made climate change')
		st.markdown('(-1) 9.1% of the tweets do not believe in man-made climate change.')
		raw['sentiment'] = [['Negative', 'Neutral', 'Positive', 'News'][x+1] for x in raw['sentiment']]# checking the distribution
		st.subheader('Pie chart visual of sentiment distribution')
		values = raw['sentiment'].value_counts()/raw.shape[0]
		labels = (raw['sentiment'].value_counts()/raw.shape[0]).index
		colors = ['olive', 'greenyellow', 'palegreen', 'g']
		plt.pie(x=values, labels=labels, autopct='%1.1f%%', startangle=90, explode= (0.1, 0.1, 0.1, 0.1), colors=colors)
		st.pyplot()

		st.subheader('Wordcloud visual of sentiment distribution')
		st.markdown('The word cloud is a visualisation that represents the most frequently occuring words in the given dataset')
		st.markdown('')
		all_words = " ".join([sentence for sentence in raw["message"]])
		wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)
		#plot graph
		plt.figure(figsize=(15,8))
		plt.imshow(wordcloud, interpolation= 'bilinear')
		plt.axis('off')
		st.pyplot()
		# Visualize frequent words
		st.subheader('Main words used:')
		st.markdown('Climate change')
		st.markdown('Global warming')
		st.markdown('Believe')
		st.markdown('Scientist')
		st.markdown('Hoax')
		st.markdown('Change')
		st.markdown('President / Trump')
		st.markdown('Earth')
		st.markdown('Action')
		st.markdown('Combat')
		st.markdown(' Fight')
		st.markdown('American')

		st.subheader("Raw Twitter data and label")
		st.write(raw[['sentiment', 'message']]) # will write the df to the page

	if selection == 'Contact App Developers':
		
		st.write('Contact details in case you any query or would like to know more of our designs:')
		st.write('Keara: kbarnard625@gmail.com')
		st.write('Ronewa: Mutobvuronewa@gmail.com')
		st.write('Leham: leham.greeves@gmail.com')
		st.write('Cecilia: Cecilianunguiane@gmail.com')
		st.write('Ayanda: ayanda7397@gmail.com')
		st.write('Siya: Siya.xola94@gmail.com')

## Looking at the main words used throughout these tweets, we can see that there are many conflicting beliefs.
## Some belief that climate change is:
## * man-made
## * political agenda being pushed
## * a hoax
## * occuring only in USA
## * other


st.set_option('deprecation.showPyplotGlobalUse', False)

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
