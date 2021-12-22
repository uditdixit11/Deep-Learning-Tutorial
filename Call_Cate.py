import pandas as pd
import numpy as np
import seaborn as  sns
import re
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras import optimizers
import sys
sys.executable

############## Step 2

df = pd.read_csv('C:/Users/e5610521/Udit_bkp/Udit/Dataset/googleplaystore/complaints.csv')
df = df[pd.notnull(df['Consumer complaint narrative'])]
df.shape

############## Step 3
df.loc[((df['Product'] == "Credit card or prepaid card") & (df['Sub-product'] == "General-purpose prepaid card")),'Product'] = "Prepaid card"
df.loc[((df['Product'] == "Credit card or prepaid card") & (df['Sub-product'] == "Payroll card")),'Product'] = "Prepaid card"
df.loc[((df['Product'] == "Credit card or prepaid card") & (df['Sub-product'] == "Gift card")),'Product'] = "Prepaid card"
df.loc[((df['Product'] == "Credit card or prepaid card") & (df['Sub-product'] == "Student prepaid card")),'Product'] = "Prepaid card"
df.loc[((df['Product'] == "Credit card or prepaid card") & (df['Sub-product'] == "Government benefit card")),'Product'] = "Prepaid card"


############### Step 4

df_cat = pd.read_csv('C:/Users/e5610521/Udit_bkp/Udit/Dataset/googleplaystore/testcategories.csv')
df_cat.shape
df_none = pd.read_csv('C:/Users/e5610521/Udit_bkp/Udit/Dataset/IMDBDataset12.csv')
df_none['Product'] = 'None'
df_news = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')
df_news = df_news.drop(['target'], axis = 1)
df_news.rename(columns={'content': 'Consumer complaint narrative', 'target_names': 'Product'}, inplace=True)
df_news['Product'] = 'None'
df_none_all = pd.concat([df_news,df_none], ignore_index=True)
print(df_none_all.shape)
df_none_all.head(2)

################## Step 5

df_none_all[df_none_all['Consumer complaint narrative'].str.len() == 1]
df_dic = {'Consumer complaint narrative': '', 'Product': 'None'}
df_none_all = df_none_all.append(df_dic, ignore_index = True)
df_none_all[df_none_all['Consumer complaint narrative'].str.len() < 2]
df_none_all.shape
df_none_all.loc[df_none_all['Consumer complaint narrative'].str.len() == 1, 'Consumer complaint narrative'] == ""
df_none_all[df_none_all['Consumer complaint narrative'].str.len() < 2]

################# Step 6

df.loc[df['Product'] == 'Credit reporting', 'Product'] = 'Credit reporting, credit repair services, or other personal consumer reports'
df.loc[df['Product'] == 'Credit card', 'Product'] = 'Credit card or prepaid card'
df.loc[df['Product'] == 'Payday loan', 'Product'] = 'Payday loan, title loan, or personal loan'
df.loc[df['Product'] == 'Virtual currency', 'Product'] = 'Money transfer, virtual currency, or money service'
df = df[df.Product != 'Other financial service']
df.loc[df['Product'] == 'Student loan', 'Product'] = 'Loan or lease'
df.loc[df['Product'] == 'Vehicle loan or lease', 'Product'] = 'Loan or lease'
df.loc[df['Product'] == 'Consumer Loan', 'Product'] = 'Loan or lease'
df.loc[df['Product'] == 'Payday loan, title loan, or personal loan', 'Product'] = 'Loan or lease'
df.loc[df['Product'] == 'Checking or savings account', 'Product'] = 'Bank account or service'
df.loc[df['Product'] == 'Money transfers', 'Product'] = 'Money transfer, virtual currency, or money service'
print(df.shape)
df["Product"].value_counts()

############### Steop 7
df_subclass = df
CATEGORY_DICT = { 0: 'Banking Services',
                  1: 'Credit Card',
                  2: 'Credit Reporting',
                  3: 'Debt Collection',
                  4: 'Loan or Lease',
                  5: 'Money Services',
                  6: 'Mortgage',
                  7: 'Network Support',
                  8: 'None',
                  9: 'Prepaid Card'}
df.loc[df['Product'] == "Money transfer, virtual currency, or money service"]['Issue'].value_counts()
df_class_debt = df[df['Product'] == "Debt collection"]
df_class_mortage = df[df['Product'] == "Mortgage"]
df_class_reporting = df[df['Product'] == "Credit reporting, credit repair services, or other personal consumer reports"]
df_class_prepaid = df[df['Product'] == "Prepaid card"]

count_class_reporting = df.Product.value_counts()[0]
count_class_debt = df.Product.value_counts()[1]
count_class_mortgage = df.Product.value_counts()[2]
df_class = df[df['Product'].isin(["Credit card or prepaid card",'Loan or lease','Bank account or service','Money transfer, virtual currency, or money service'])]
df_class.Product.value_counts()


############################# Step 8 

df_class_debt_under = df_class_debt.sample(count_class_mortgage)
df_class_reporting_under = df_class_reporting.sample(80000)
df_class_prepaid_over = df_class_prepaid.sample(45000,replace=True)
df_class_network_over = df_cat.sample(45000,replace=True)
print(df_class_debt_under.shape,df_class_reporting_under.shape,df_class_network_over.shape,df_class_prepaid_over.shape)
df_none_all_over = df_none_all.sample(82000,replace=True)

################################# Step 9

df1 = df
df_under = pd.concat([df_class_reporting_under, df_class_debt_under,df_class_mortage,df_class,df_class_prepaid_over,df_class_network_over], axis=0)
df = df_under
df = df.reset_index()

############################### Step 10
df_n = pd.concat([df,df_none_all_over], axis=0)
df_n.shape
df = df_n
df = df.reset_index()
df.shape

################################### STep 11
df_export1 =  df.loc[:,['Consumer complaint narrative','Product']]
df_export1.rename(columns={'Consumer complaint narrative': 'Transcript', 'Product': 'Category'}, inplace=True)
df_export1.shape

########################### Step 12
def print_plot(index):
    example = df[df.index == index][['Consumer complaint narrative', 'Product']].values[0]
    if len(example) > 0:
        print(example[0])
        print('Product:', example[1])

######################### Step 13
print_plot(100)
df = df.reset_index(drop=True)
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]><!#?^')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
custom_stop_words = ['.', ',', '"', "'", ':', ';','bye', 'hey','ive','sir','sorry','wh', 'Wh','im','nt',"good",'morning','evening','hi','hello','from', 'subject', 're', 'edu', 'use','speak','are',"yes","could","n","go","asks","u","got","ask","wen","ta","tell","someone","say","says","tommorow",'ill','make','told','1st','yeah',"guy","might","think","gone","tells","weren't","us","sent","get","see","let","dats",'one',"theyre","didnt","dont","also","tru","theres","end","didn't","cant","son","mom","thats","because","went","childern","come","back","comes","set","belive","believe","break","broke","anyways","anyway","moms","alright","since","everything","even","taes","other","others","tha",'thanks','thank','calling','goodbye','care','hope','please','help','hear','really','maam','mam','ridiculous','name','theyhv','idiot','stupid','ridiculously','ridiculous','date','birth','last','still','theyve','may','would','way','know','mine','instead','birthday','ahead','wrong','damn','guys','first','doesnt','have','been','hello','frustated','actually','thankyou','sound','nice','day','ssn','great','anything','else', 'lot']
# custom_stop_words = ['.', ',', '"', "'", ':', ';','bye', 'hey','ive','sir','sorry','wh', 'Wh','im','nt',"good",'morning','evening','hi','hello','from', 'subject', 're', 'edu', 'use','speak','are',"yes","could","n","go","asks","u","got","ask","wen","ta","tell","someone","say","says","tommorow",'ill','make','told','1st','yeah',"guy","might","think","gone","tells","weren't","us","sent","get","see","let","dats",'one',"theyre","didnt","dont","also","tru","theres","end","didn't","cant","son","mom","thats","because","went","childern","come","back","comes","set","belive","believe","break","broke","anyways","anyway","moms","alright","since","everything","even","taes","other","others","tha",'thanks','thank','calling','goodbye','care','hope','please','help','hear','really','maam','mam','ridiculous','name','theyhv','idiot','stupid']
# custom_stop_words = ['.', ',', '"', "'", ':', ';', 'wh', 'Wh','im','nt',"good",'morning','evening','hi','hello','from', 'subject', 're', 'edu', 'use','speak','are',"yes","could","n","go","asks","u","got","ask","wen","ta","tell","someone","say","says","tommorow",'ill','make','told','1st','yeah',"guy","might","think","gone","tells","weren't","us","sent","get","see","let","dats",'one',"theyre","didnt","dont","also","tru","theres","end","didn't","cant","son","mom","thats","because","went","childern","come","back","comes","set","belive","believe","break","broke","anyways","anyway","moms","alright","since","everything","even","taes","other","others","tha"]
# custom_stop_words = ['.', ',', '"', "'", ':', ';', 'wh', 'Wh','im','nt',"good",'morning','evening','hi','hello','from', 'subject', 're', 'edu', 'use','speak','are',"yes","could"]
STOPWORDS.update(custom_stop_words)
def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = str(text).lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = text.replace('x', '')
    text = text.replace('X', '')
    text = text.split()
    text = [word for word in text if word.isalpha() and len(word) > 2 and word not in STOPWORDS]
#     cleaned_text = [word for word in text if word not in STOPWORDS]
    cleaned_text = " ".join(text)
#    text = re.sub(r'\W+', '', text)
#     text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    return cleaned_text

###################### Step 14
df['Consumer complaint narrative'] = df['Consumer complaint narrative'].apply(clean_text)
df['Consumer complaint narrative'] = df['Consumer complaint narrative'].str.replace('\d+', '')


###################### Step 15
raw_docs_train = df['Consumer complaint narrative'].values
product_train1 = df['Product'].values
product_train1[120:130]

##################### Step 16

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
product_train = label_encoder.fit_transform(product_train1)

##################### Step 18
print(product_train1[np.where(product_train == 9)[0][0]])
product_train1[214477]

##################### Step 19
maxLen = 150

####################  Step 20

sns.countplot(x='Product',data=df)

#################### Step 21

X_train, X_test, Y_train, Y_test = train_test_split(raw_docs_train, product_train, stratify=product_train, random_state=42, test_size=0.1, shuffle=True)

################# Step 22

from keras.utils import np_utils
num_labels = len(np.unique(product_train))
print(num_labels)
Y_oh_train = np_utils.to_categorical(Y_train, num_labels)
Y_oh_test = np_utils.to_categorical(Y_test, num_labels)

#################### Step 23 

glove_file = 'C:/Users/e5610521/Udit_bkp/Udit/Dataset/glove_6B_100d.txt'
def read_glove_vecs(glove_file):
    with open(glove_file, encoding='utf8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map
  
  
  ######################### Step 24
  
  word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(glove_file)
  
  def lstmm_model(input_shape, word_to_vec_map, word_to_index):
    sentence_indices = Input(shape = input_shape, dtype='int32')

    # create the embedding layer pretrained with Glove vectorsl i9mkuj 0-0 n 9
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

    embeddings = embedding_layer(sentence_indices)

    # propagate the embeddings through a Lstm layer with 128-dim hidden state
    # return will be batch of sequence
    X = LSTM(128, return_sequences = True)(embeddings)
    # adding dropout with probability of 0.5
    X = Dropout(0.5)(X)

    # using another Lstm layer with 16-dim hidden state
    # return output will be a single hidden state
    X = LSTM(16, return_sequences = False)(X)
    # Add dropout with probability of 0.5
    X = Dropout(0.5)(X)

    # adding dense layer with softmax activation to get batch of 5-dim vectors
    X = Dense(10, activation=None)(X)
    # adding softmax activation function
    X = Activation('softmax')(X)

    # creating model. it will convert sentence_indices into X
    model = Model(inputs=[sentence_indices], outputs=X)

    return model
  
  ###################### Step 25
  
  def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    vocab_len = len(word_to_index) + 1 #added 1 to fit embedding(requirement)
    emb_dim = 100 #defining dimensionality of your Glove word vector(=50)

    emb_matrix = np.zeros((vocab_len, emb_dim))

    # setting each row "index" of the embedding matrix to be word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    # defining the embedding layer with the correct output/ input sizes, make it trainable
    embedding_layer = Embedding(vocab_len, emb_dim, trainable = False)

    # Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".
    embedding_layer.build((None,))

    # set the weights of the embedding layer to the embedding matrix. your layer is now pretrained
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer
  
  ########################### Step 26
  
 model = lstmm_model((maxLen, ), word_to_vec_map, word_to_index)
print(model.summary())

########################### Step 27

def sentences_to_indices(X, word_to_index, max_len):
    m = X.shape[0]
    # initializing X_indices as a numpy matrix of zero and the correct shape
    X_indices = np.zeros((m, max_len))

    for i in range(m):#loop of training examples
        # Converting the ith training sentence in lower case and split it into words.
        sentence_words = [word.lower().replace('\t', '') for word in X[i].split(' ') if word.replace('\t', '') != '']
        j = 0
        # loop of words of sentence_words
        for w in sentence_words:
            # setting the (i,j)th entry of X_indices to the index of the correct word.
            try:
                X_indices[i, j] = word_to_index[w]

            except : 0
            # Increment j to j + 1
            j = j+1
    return X_indices
  
############ Step 8

X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
print(X_train_indices. shape)
adam = optimizers.Adam()
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])

########################## Step 29

from keras.callbacks import EarlyStopping
earlystop = EarlyStopping(monitor = 'val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
model.fit(X_train_indices, y=Y_oh_train, batch_size=256, epochs=18, verbose=1, validation_data=(X_test_indices, Y_oh_test), callbacks = [earlystop])

########################### Step 20

from keras.callbacks import EarlyStopping
earlystop = EarlyStopping(monitor = 'val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
model.fit(X_train_indices, y=Y_oh_train, batch_size=256, epochs=18, verbose=1, validation_data=(X_test_indices, Y_oh_test), callbacks = [earlystop])

###################### Step 31
# data = clean_text("Hi this is samantha, how may i help you. This is george, on last friday I made a payment transfer using account number 87654321987. My account has been debited with a amount of 5000 dollar but after sometime I got failed transaction message. I want to recheck whether this transaction is failed or not. Off course Sir we will help you out with that, Can I have your ssn Its 9876543234 and your date of birth. Its 31 december 1980. Yes Friday one transfer of 5000 dollar has been made from your account")
# data = clean_text("hi thank you for calling cardholder services my name is jay may have your 16 digit card number please 5173060010362519 thank you so much and may i have your first name and your last name spears o'neal thank you so much so let me go ahead and try to pull up your records in our system first kindly on the line please OK so I already pull up your records and how may i help you today yes i'm trying to activate this gift for it OK so regarding with this let me verify some information first may i have your the original load balance for this gift card i'm sorry what was that how much is the initial balance for this gift card  yes i don't know it was a gift given to me so that's why i'm trying to activate it how much is the load or the initial balance for this gift card i don't have a balance sorry i don't know about i don't know regarding with this i do apologize but the original balance is part of verification if you cannot be able to provide them with that i cannot be able to give you the information of the card but if you but all you need to do is to set up your pin code to the automated system or you can just simply sit visit the website there's about website at the back of your card ")
# data = clean_text("hello hello can't hear you")
# data = clean_text("hi thank you for calling cardholder services my name is jade may i have your sixteen digit card number please just a second four seven five four two three zero one zero eight nine one five four four one thank you so much and may have your first name and your last name please gary fredrickson thank you so much and how may i help you today well i went to use my travel visa card and it told me that it was expired so when i went to triple A they told me i had to call this number and request a new card OK sure don't worry let us replace your card but let me verify some information first OK may i have your last four digits of your social security number two one six nine and your date of birth please one thirty one forty seven thank you so much hell gary let me go ahead and replace your call but i just need to verify your address can you provide to me your address please eight three one perceval street southwest perceval is P E R C I V A L OK thank you so much so don't worry let me let me replace your card and you will receive your card within seven to ten business days OK OK and did you did you have the rest of the address olympia washington nine eight five oh two i'm yeah kent so it'll come to this address in seven to ten days seven to ten business days not included there saturday and sunday OK thank you you're welcome is there anything else that i can help you with that'll do it thank you you're welcome and thank you for calling cardholder services have a great day bye for now bye bye")
# data = clean_text("thank you for getting mad at banking system and the how can i help hi my name is alex masuk i have a reward card it's from samsung refund and unfortunately it expired so i need another card i see OK sounds like sir let me try to connect you to the right department for the best system and thank you very wait now i i had another person trying to transfer me and then the system just hung up on me i'm really sorry to hear that sir but i don't have access here on my end but you don't have to worry i'll try to connect you again OK OK alright thank you you're welcome thank you for calling me back have a great day")
# data = clean_text("I opened new accounts at Chase bank on 12 of april 2015 and the next day, I closed my account with XXXX XXXX XXXX and made cash deposits of {$6000.00} ( {$3000.00} into my Chase Checking Account and {$3000.00} into my Chase Liquid Account ). ")
# data = clean_text("Hi you are speaking with Samantha Can I have your first name It's jay and last name is rahul. Can I have your SSN please Its 987654321 ok great Your account or card number. its  987654321. I have a bank account with pnc bank. I had a over draft of 2000 dollars. they charged me 500 dollar over draft fee. then a administration fee and 20 dollars every week over drawn reacurring fee. Please check. We are really sorry for this. i'll be transferring you over to the right department for you to be assisted today alright.May I have contact number in case call is disconnected yes its 99 ok 6754 ok 987. Thank you have a nice day")
# data = clean_text("Hi this is samantha. How may I help you? I meant that this is george. On last friday, I made a payment transfer using account number *********** OK, and my account has been debited wi the amount of **** dollars, but after some time I got failed transaction message. I want to re check whether this transaction is failed or not, of course, then he will help you out wi that can I have your assistant, it's *********** the date of bir, it's 31 december **** OK. Let me pull out your records. Yes, friday. 1 transfer of **** dollars has been made from your accounts. Ok thanks a lot.")
data = clean_text("Hi you are speaking with. How may I help you ? This is jay, my debit card is lost or has been stolen by someone, and I want to block my card as soon as possible. Yeah sure, sir, before it, I want to verify some information. Can I have your assessment? It's ************* Thank you. Your 1 name. It's john, your last name, bender, your date of bir. It's ** ******** ***** thanks for verifying this information, and we have blocked your card and you will be. We are reissuing a new part, and you will be getting this card within a 7 to 10 days. Anything else that I can help you wi that uh. No thanks a lot.")
print(data)
# data = clean_text("Hi this is samantha. How may I help you samantha? This is ravi. my debit card is lost and I want to block this card as soon as possible Yeah sure, sir, before it, I want to verify some information. Can I have your assessment? It's ************* Thank you. Your 1 name. It's john, your last name, bender, your date of bir. It's ** ******** ***** thanks for verifying this information, and we have blocked your card and you will be. We are reissuing a new part, and you will be getting debit card within a 7 to 10 days. Anything else that I can help you wi that uh. No thanks a lot.")

print(data)
maxLen = 150
X_mmm = np.array([data])
X_mmm = sentences_to_indices(X_mmm, load_word_to_index, maxLen)
prediction = loaded_model.predict(X_mmm)
topic_list = []
topic_percen = []
lst = prediction[0]
category_dict = {0: 'Bank account or service', 1: 'Credit card or prepaid card', 2: 'Credit reporting or other personal consumer reports', 3: 'Debt collection', 4: 'Loan or lease', 5: 'Money transfer, virtual currency, or money service', 6: 'Mortgage', 7: 'Network Support', 8: 'None', 9: 'Prepaid card'}
category_index = np.argmax(prediction)
call_categorize = category_dict[category_index]
list_cat_tuple = sorted( [(x,i) for (i,x) in enumerate(lst)], reverse=True )[:3]
for tup in list_cat_tuple:
    topic_list.append(category_dict[tup[1]])
    topic_percen.append(tup[0])
    
print(prediction)
print(prediction[0][7])
print(np.argmax(prediction))
print(product_train1[np.where(product_train == np.argmax(prediction))[0][0]])
print(topic_list)
# "yes","could","n","go","asks","u","got","ask","wen","ta","tell","someone","say","says","tommorow",'ill','make','told','1st','yeah',"guy","might","think","gone","tells","weren't","us","sent"
# "theyre","didnt","dont","also","tru","theres","end","didn't","cant","son","mom","thats","because","went","childern","come","back"
# "belive","believe","break","broke","anyways","anyway","moms","alright","since","everything","even","taes","other","others","tha"

################################ Step 31

from keras.models import save_model 
model.save('call_cat_model_4.h5')

from keras.models import save_model,load_model
loaded_model = load_model('call_cat_model_4.h5')

import pickle
vec_file = 'word_index_dict_4.pkl'
pickle.dump(word_to_index, open(vec_file, 'wb'))
  


























