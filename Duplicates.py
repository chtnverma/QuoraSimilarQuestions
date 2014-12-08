import sys
import json
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

def read_questions(): 
    questions={}
    trainids=[]
    trainlabels=[]
    testids=[]
    N_questions = int(sys.stdin.readline() ) 
    for i in range(N_questions): 
        line_obj = json.loads(sys.stdin.readline().strip()) 
        questions[line_obj['question_key']]=line_obj 
        
    N_train = int(sys.stdin.readline() ) 
    for i in range(N_train): 
        temp = sys.stdin.readline().strip().split(' ') 
        trainids.append(temp[:2])
        trainlabels.append(int(temp[-1])) 
    
    N_test = int(sys.stdin.readline() ) 
    for i in range(N_test): 
        temp = sys.stdin.readline().strip().split(' ') 
        testids.append(temp[:2])        
    
    return(questions, trainids, trainlabels, testids )

def get_features():
                           
    global allques, doc_similarity, doc_similarity_cosine, n_common_tokens, same_main_topic, n_common_topics, feature_vector, NFeatures 
    NFeatures=0
    allkeys = sorted(allques.keys()) 
    documents = [allques[i]['question_text'] for i in allkeys ]
    vect = TfidfVectorizer(min_df=2)
    tfidf = vect.fit_transform(documents)
    
    doc_similarity = {} # feature #1
    doc_similarity_cosine = {} # feature #2
    NFeatures+=2
    for i in range(len(allques.keys())): 
        for j in range(i+1, len(allques.keys())): 
            doc_similarity[allkeys[i]+'|'+allkeys[j]] = tfidf[i,:]*(tfidf[j,:].T) 
            doc_similarity_cosine[allkeys[i]+'|'+allkeys[j]] = doc_similarity[allkeys[i]+'|'+allkeys[j]]/(np.linalg.norm(tfidf[i,:])*np.linalg.norm(tfidf[j,:]))
            doc_similarity[allkeys[j]+'|'+allkeys[i]] = doc_similarity[allkeys[i]+'|'+allkeys[j]] 
            doc_similarity_cosine[allkeys[j]+'|'+allkeys[i]] = doc_similarity_cosine[allkeys[i]+'|'+allkeys[j]] 
    
    n_common_tokens = {} # feature #3 . normalized wrt num tokens 
    NFeatures+=1
    for i in range(len(allques.keys())): 
        for j in range(i+1, len(allques.keys())): 
            n_common_tokens[allkeys[i]+'|'+allkeys[j]] = len(set(np.where(tfidf[i,:]>0)).intersection(set(np.where(tfidf[j,:]>0)) ) )/ tfidf.shape[1]
            n_common_tokens[allkeys[j]+'|'+allkeys[i]] = n_common_tokens[allkeys[i]+'|'+allkeys[j]] 
    
    same_main_topic = {} # feature #4 
    NFeatures+=1
    for i in allques.keys(): 
        for j in  allques.keys(): 
            if i==j: continue
            if allques[i]['context_topic']==null or  allques[j]['context_topic']==null: 
                same_main_topic[i+'|'+j] = 0 # can be improved by not treating null like this. more informative features can be created here 
            elif allques[i]['context_topic']['name']==allques[j]['context_topic']['name']: 
                same_main_topic[i+'|'+j] = 1 
            else same_main_topic[i+'|'+j] = 0 


    n_common_topics={} # feature #5 . Norm. with total # topics
    NFeatures+=1
    set_alltopics=set() 
    for i in allques.keys():         
        for j in  allques.keys(): 
            if i==j: continue
            if allques[i]['topics']==null or  allques[j]['topics']==null: 
                n_common_topics[i+'|'+j] = 0 # can be improved similarly 
            else: 
                set1 = set([ allques[i]['topics'][k]['name']  for k in range(len(allques[i]['topics']))  ]) 
                set2 = set([ allques[j]['topics'][k]['name']  for k in range(len(allques[j]['topics']))  ]) 
                n_common_topics[i+'|'+j] = set1.intersection(set2) 
        set_alltopics.add([ allques[i]['topics'][k]['name']  for k in range(len(allques[i]['topics']))  ]) 
    for i in  n_common_topics.keys():
        n_common_topics[i] = n_common_topics[i]/len(set_alltopics) 
        
    # can also add features using: 
    ## k-means
    ## LDA
    ## RBM 
    
    
    
    # Creating feature vectors: 
    feature_vector={} 
    for i in allques.keys():         
        for j in allques.keys():         
            if i==j: continue 
            feature_vector[i+'|'+j] = [doc_similarity[i+'|'+j], doc_similarity_cosine[i+'|'+j],n_common_tokens[i+'|'+j],same_main_topic[i+'|'+j],n_common_topics[i+'|'+j]].toarray() 
            
    
                           
    def train_model(trainids, trainlabels): 
        clf = SVC(C=1.0, kernel='rbf') 
        train_vectors=np.zeros((len(trainids),NFeatures)) 
        for i in len(trainids): 
            train_vectors[i,:]=feature_vector[trainids[i][0]+'|'+trainids[i][1]]
        clf.fit(train_vectors, trainlabels.toarray() )  
        return(clf) 
                           
                           
    def test_model(testids, clf): 
        test_vectors=np.zeros((len(testids),NFeatures)) 
        for i in len(testids): 
            if testids[i][0]==testids[i][1]: 
                print(1) 
            else: 
                test_vectors[i,:]=feature_vector[testids[i][0]+'|'+testids[i][1]]
                print(clf.predict(test_vectors)  ) 
                           
def main(): 
    allques, trainids, trainlabels, testids = read_questions() 
    get_features()    
    clf = train_model(trainids, trainlabels, )
    test_model(testids, clf) # add case when ID is same. 
    

if __name__=="__main__":
    main()
    
