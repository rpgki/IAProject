# Libraries used
import random
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# The dataset is created within the main program
# Creation of dataframe with input features
raw_X = {
    'Character': ['Ajani','Ajani','Ajani','Angrath','Angrath','Arlinn','Arlinn','Arlinn','Arlinn','Ashiok','Chandra','Chandra','Chandra','Chandra','Chandra','Davriel','Domri','Domri','Dovin','Dovin','Gideon','Gideon','Gideon','Gideon','Huatli','Huatli','Huatli','Huatli','Huatli','Jace','Jace','Jace','Jace','Jace','Jace','Jace','Jace','Jace','Jaya','Jaya','Jaya','Jaya','Jiang','Jiang','Jiang','Karn','Karn','Karn','Karn','Kasmina','Kaya','Kaya','Kaya','Kiora','Liliana','Liliana','Nahiri','Nahiri','Narset','NicolBolas','NicolBolas','NicolBolas','NicolBolas','NicolBolas','Nissa','Nissa','ObNixilis','ObNixilis','Ral','Ral','Ral','Saheeli','Saheeli','Samut','Samut','Samut','Sarkhan','Sarkhan','Sorin','Sorin','Tamiyo','Teferi','Teferi','Teferi','Teferi','Teyo','Teyo','Teyo','Teyo','Teyo','Tezzeret','Tezzeret','Tezzeret','Tezzeret','Tibalt','Ugin','Ugin','Ugin','Vivien','Vivien','Vivien','Vraska','Vraska','Vraska','Vraska']
    ,'Act': ['Acto 3','Acto 3','Acto 3','Acto 3','Acto 3','Acto 3','Acto 3','Acto 3','Acto 3','Acto 3','Acto 2','Acto 2','Acto 3','Acto 3','Acto 3','Acto 3','Acto 2','Acto 2','Acto 2','Acto 2','Acto 3','Acto 3','Acto 3','Acto 3','Acto 3','Acto 3','Acto 3','Acto 3','Acto 3','Acto 2','Acto 2','Acto 2','Acto 3','Acto 3','Acto 3','Acto 3','Acto 3','Acto 3','Acto 2','Acto 2','Acto 2','Acto 3','Acto 3','Acto 3','Acto 3','Acto 3','Acto 3','Acto 3','Acto 3','Acto 3','Acto 3','Acto 3','Acto 3','Acto 3','Acto 1','Acto 3','Acto 2','Acto 3','Acto 3','Acto 2','Acto 3','Acto 3','Acto 3','Acto 3','Acto 2','Acto 2','Acto 3','Acto 3','Acto 1','Acto 1','Acto 2','Acto 2','Acto 2','Acto 3','Acto 3','Acto 3','Acto 3','Acto 3','Acto 2','Acto 3','Acto 2','Acto 3','Acto 3','Acto 3','Acto 3','Acto 2','Acto 2','Acto 3','Acto 3','Acto 3','Acto 1','Acto 3','Acto 3','Acto 3','Acto 3','Acto 3','Acto 3','Acto 3','Acto 2','Acto 2','Acto 2','Acto 1','Acto 1','Acto 1','Acto 3']
    ,'Act 1': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0]
    ,'Act 2': [0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,1,1,0,0,1,1,1,1,1,0,0,0,0,0,1,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0]
    ,'Act 3': [1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,1,0,1,1,1,1,0,0,1,1,0,0,0,0,0,1,1,1,1,1,0,1,0,1,1,1,1,0,0,1,1,1,0,1,1,1,1,1,1,1,0,0,0,0,0,0,1]																																																																																																							
    ,'Allied': [1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,1,0,0,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,1,1,0,0,0,0,1,1,0,0,1,1,1,1,0,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,0,1,1,1,1,1,1,1]
    ,'Red': [0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,1,1,1,1,0,0,1,0,1,0,0,0,1,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,1,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,1,1,0,0,1,0,0,0,1,0,1,0,0,0,0,1,0,1,0,1,0,1,0,1,0,0,0]
    ,'White': [1,0,0,1,1,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,0,1,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0,1,1,1,0,0,1,0,0,0,1,1,0,0,1,1,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0]
    ,'Black': [0,0,1,0,0,1,1,0,0,1,0,0,0,0,1,0,1,1,0,1,1,0,1,1,0,1,0,0,1,0,0,0,1,1,0,0,0,1,0,0,1,0,0,0,1,1,0,1,0,0,0,0,1,0,1,0,1,1,0,1,0,1,0,0,1,0,0,0,0,1,1,0,0,0,1,1,0,1,0,0,0,0,0,0,1,1,0,1,0,0,1,1,0,0,1,0,0,1,0,0,0,0,1,0,0]
    ,'Green': [1,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,1,0,0,1,1,0,0,0,0,0,0,0,1,1,1,1,1,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
    ,'Blue': [0,0,0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,1,0,1,1,1,0,0,0,0,0,0,0,1,0,1,0,1,0,1,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0,1,1,0,0,0,0,1,0,1,0,1,0,0,0,0,0,1,1,1,0,1,1,0,0,1]
}
X = pd.DataFrame(data=raw_X)

# Creation of dataframe with output feature
raw_y = {
    'Relation': [1,2,3,4,1,5,6,4,7,8,9,10,4,11,12,13,14,15,16,15,5,1,12,15,4,5,17,2,3,18,19,20,3,12,16,9,21,15,11,19,20,22,17,1,3,23,22,24,25,26,27,28,3,29,15,4,30,30,31,12,4,12,21,11,14,27,32,22,27,14,3,16,10,32,23,24,21,15,33,33,34,11,18,20,12,14,28,3,28,27,24,23,32,22,6,25,11,15,11,18,19,28,14,27,11]
}
Y = pd.DataFrame(data=raw_y)

# The user is asked to input the main character and story
character = input('Input the main character: ')
story = input('Input the color of the story: ')

# The colors are set when user inputs the story
if story == 'white':
    red = 0
    white = 1
    blue = 0
    black = 0
    green = 0
elif story == 'red':
    red = 1
    white = 0
    blue = 0
    black = 0
    green = 0
elif story == 'blue':
    red = 0
    white = 0
    blue = 1
    black = 0
    green = 0
elif story == 'black':
    red = 0
    white = 0
    blue = 0
    black = 1
    green = 0
elif story == 'green':
    red = 0
    white = 0
    blue = 0
    black = 0
    green = 1

# The acts are chosen as random
act1 = random.randint(0,1)
act2 = random.randint(0,1)
act3 = random.randint(0,1)

# A function is defined to determine the ally in different acts
def Ally():
    if story == 'white' & (act == 'Acto 1' | act == 'Acto 2' | act == 'Acto 3'):
        ally = 1
    elif story == 'blue' & (act == 'Acto 1' | act == 'Acto 2' | act == 'Acto 3'):
        ally = 1
    elif story == 'black' & (act == 'Acto 1' | act == 'Acto 2' | act == 'Acto 3'):
        ally = 0
    elif story == 'red' & (act == 'Acto 1' | act == 'Acto 3'):
        ally = 1
    elif story == 'red' & act == 'Acto 2 ':
        ally = 0
    elif story == 'green' & act == 'Acto 1':
        ally = 1
    elif story == 'green' & (act == 'Acto 2' | act == 'Acto 3'):
        ally = 0
    return ally

# Setting colors based on story by user
if story == 'red':
    red = 1; white = 0; black = 0; green = 0; blue = 0
elif story == 'white':
    red = 0; white = 1; black = 0; green = 0; blue = 0
elif story == 'black':
    red = 0; white = 0; black = 1; green = 0; blue = 0
elif story == 'green':
    red = 0; white = 0; black = 0; green = 1; blue = 0
elif story == 'blue':
    red = 0; white = 0; black = 0; green = 0; blue = 1

# Training the model
clf = LogisticRegression(solver='sag', max_iter=100, random_state=42,
                         multi_class='multinomial').fit(X,y)

# The values to predict are saved
Act1_X = {
    'Character': character
    ,'Act': 'Acto 1'
    ,'Act 1': act1
    ,'Act 2': act2
    ,'Act 3': act3
    ,'Allied': Ally()
    ,'Red': red
    ,'White': white
    ,'Black': black
    ,'Green': green
    ,'Blue': blue
}

Act2_X = {
    'Character': character
    ,'Act': 'Acto 2'
    ,'Act 1': act1
    ,'Act 2': act2
    ,'Act 3': act3
    ,'Allied': Ally()
    ,'Red': red
    ,'White': white
    ,'Black': black
    ,'Green': green
    ,'Blue': blue
}

Act3_X = {
    'Character': character
    ,'Act': 'Acto 3'
    ,'Act 1': act1
    ,'Act 2': act2
    ,'Act 3': act3
    ,'Allied': Ally()
    ,'Red': red
    ,'White': white
    ,'Black': black
    ,'Green': green
    ,'Blue': blue
}

Act1 = pd.DataFrame(data=Act1_X)
Act2 = pd.DataFrame(data=Act2_X)
Act3 = pd.DataFrame(data=Act3_X)

# The planewalker to assist the main character is predicted
predAct1 = clf.predict(Act1)
predAct2 = clf.predict(Act2)
predAct3 = clf.predict(Act3)

# Then the story is told
# The next part is pseudo code
if story == 'red':
    print("Red Story With Model")
elif story == 'black':
    print("Black Story With Model")
elif story == 'white':
    print("White Story With Model")
elif story == 'green':
    print("Green Story With Model")
elif story == 'blue':
    print("Blue Story With Model")