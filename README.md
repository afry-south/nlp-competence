# AFRY NLP Competence Group Meetings
. 
In this repository I'll try to summarize & keep the data from the different meetings. Some meetings are found in the "`jvm`"-folder but most are found in "`python`" as Python is the "de facto" langauge for Machine Learning in the industry.

The different meetings:

1. [Text Classification IMDB - 20190204](#Text-Classification-IMDB-[20190204,-#1])
2. (WIP) [Text Classification Quora Insincere Questions - 20190223](#(WIP)-Text-Classification-Quora-Insincere-Questions-[20190223,-#2])
3. [Quora pt 2 - 20190402](#Text-Classification-Quora-Insincere-Questions-pt-2-[,-#3])
4. [Neural Network 101 - 20190521](#Neural-Networks-Basics-[20190521,-#4])
5. [Text Generation pt 1 - 20190626](#Text-Generation-[20190626,-#5])
6. [Text Generation pt 2 - 20191017](#Text-Generation-pt-2-[20191017,-#6])
7. []
8. []


### Text Classification IMDB [20190204, #1]
**Goal:** To classify IMDB review as positive or negative  
**How:** We started out with explaining basic concepts, then we implemented classic Machine Learning & ended up looking into Word Embeddings  
**Why:** Fun task where we learned a lot of basics in Machine Learning and Natural Language Processing  
**Keywords:** TFIDF, Bag of Words, Preprocessing, Machine Learning, Word2Vec, Word Embeddings, scikit-learn, pandas, 

### (WIP) Text Classification Quora Insincere Questions [20190223, #2]
Currently a mess really. We did this locally & had a lot of dependencies...
I'm gonna find time sometime to convert this into `.ipynb`-files to make it easier to work with.    
**Goal:** To classify Quora posts as insincere or sincere (Kaggle competition). A very hard task as not even humans agree.  
**How:** Implemented a baseline using spaCy which is a widely used tool in the industry. Reached a decent scoring.  
**Why:** This meeting we decided what approaches to use in the future (more practical than theoretical)

### Text Classification Quora Insincere Questions pt 2 [20190402, #3]
**Goal:** To classify Quora posts as insincere or sincere (Kaggle competition). A very hard task as not even humans agree.  
**How:** Two implementations that we can tweak directly on Kaggle provided by me.    
**Why:** To see how Kaggle works, test more complex solutions and have some fun by joining a competition


### Neural Networks Basics [20190521, #4]
**Goal:** To learn the basics of Neural Networks such as the building stones, the flow & some feel of the math.  
**How:** A great Python Notebook that shows theory and then lets you implement a Recurrent Neural Network that performs Machine Translation (Eng -> Swe)  
**Why:** Learn basics in Neural Network so that new layers won't they're recognizable, such as Dropout layer etc.

### Text Generation [20190626, #5]
**Goal:** To get a greater understanding of Language Models & see how the "old-school" approaches might even beat Neural Networks with much simpler & easier to grasp concepts + some clever tricks.  
**How:** A Python Notebook as usual which was mainly code to code yourself (!). We generated words on a word-to-word and char-to-char basis.  
**Why:** Understand the core to later expand with more advanced techniques such as Neural Networks

### Text Generation pt 2 [20191017, #6]
**Goal:** To finally try out Neural Networks & generate some Shakespeare.  
**How:** A Python Notebook as usual which was mainly code to code yourself (!). We generated words on a char-to-char basis.  
**Why:** Work with Neural Networks & learn tensorflow

### Social Media Perception / Data Mining [20191127, #7]
**Goal:** Learn some techniques for Data Understanding and ways to visualize data to learn more about it  
**How:** A Python Notebook as usual, during this workshop you'll first learn how to get data from Twitter, parse it and then show it in a Word Cloud. In the end a classifier for which one who's tweeting a certain tweet will be done, comparing J Sjöstedt to J Åkesson.  
**Why:** Learn how to fetch Twitter data in bulk, how to make word-cloud and have some fun.

### SparkNLP [20200212, #8]
**Goal:** Learn what Spark, SparkML and **SparkNLP** is. What we can do with it and so on. Also learn when not to use it (as it brings overhead).  
**How:** Using a JVM project (hidden in `jvm`-folder) we had tooling to try out a Scala Spark approach and a Kotlin "Local" approach.
**Why:** For fun, and to understand that all tools have their place.