# Libraries used
import random
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# The dataset is created within the main program
# Creation of dataframe with input features
raw_X = {
    'Character': [17,17,17,5,5,35,35,35,35,8,16,16,16,16,16,6,36,36,10,10,4,4,4,4,1,1,1,1,1,11,11,11,11,11,11,11,11,11,18,18,18,18,2,2,2,32,32,32,32,26,14,14,14,29,12,12,33,33,31,15,15,15,15,15,7,7,23,23,28,28,28,9,9,22,22,22,25,25,30,30,34,19,19,19,19,27,27,27,27,27,24,24,24,24,13,21,21,21,20,20,20,3,3,3,3]
    ,'Act': [2,2,2,2,2,2,2,2,2,2,1,1,2,2,2,2,1,1,1,1,2,2,2,2,2,2,2,2,2,1,1,1,2,2,2,2,2,2,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,0,2,1,2,2,1,2,2,2,2,1,1,2,2,0,0,1,1,1,2,2,2,2,2,1,2,1,2,2,2,2,1,1,2,2,2,0,2,2,2,2,2,2,2,1,1,1,0,0,0,2]
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

# The character name and story are lower case
character = character.lower()
story = story.lower()

# A function is defined to convert strings to numbers
def StringToNumber(character):
    if character == 'huatli':
        character = 1
    elif character == 'jiang':
        character = 2
    elif character == 'vraska':
        character = 3
    elif character == 'gideon':
        character = 4
    elif character == 'angrath':
        character = 5
    elif character == 'davriel':
        character = 6
    elif character == 'nissa':
        character = 7
    elif character == 'ashiok':
        character = 8
    elif character == 'saheeli':
        character = 9
    elif character == 'dovin':
        character = 10
    elif character == 'jace':
        character = 11
    elif character == 'liliana':
        character = 12
    elif character == 'tibalt':
        character = 13
    elif character == 'kaya':
        character = 14
    elif character == 'nicolbolas':
        character = 15
    elif character == 'chandra':
        character = 16
    elif character == 'ajani':
        character = 17
    elif character == 'jaya':
        character = 18
    elif character == 'teferi':
        character = 19
    elif character == 'vivien':
        character = 20
    elif character == 'ugin':
        character = 21
    elif character == 'samut':
        character = 22
    elif character == 'obnixilis':
        character = 23
    elif character == 'tezzeret':
        character = 24
    elif character == 'sarkhan':
        character = 25
    elif character == 'kasmina':
        character = 26
    elif character == 'teyo':
        character = 27
    elif character == 'ral':
        character = 28
    elif character == 'kiora':
        character = 29
    elif character == 'sorin':
        character = 30
    elif character == 'narset':
        character = 31
    elif character == 'karn':
        character = 32
    elif character == 'nahiri':
        character = 33
    elif character == 'tamiyo':
        character = 34
    elif character == 'arlinn':
        character = 35
    elif character == 'domri':
        character = 36
    return character

# A function is created to transform the result to string
def NumberToString(number):
    if number == 1:
        rsl = 'huatli'
    elif number == 2:
        rsl = 'jiang'
    elif number == 3:
        rsl = 'vraska'
    elif number == 4:
        rsl = 'gideon'
    elif number == 5:
        rsl = 'angrath'
    elif number == 6:
        rsl = 'davriel'
    elif number == 7:
        rsl = 'nissa'
    elif number == 8:
        rsl = 'ashiok'
    elif number == 9:
        rsl = 'saheeli'
    elif number == 10:
        rsl = 'dovin'
    elif number == 11:
        rsl = 'jace'
    elif number == 12:
        rsl = 'liliana'
    elif number == 13:
        rsl = 'tibalt'
    elif number == 14:
        rsl = 'kaya'
    elif number == 15:
        rsl = 'nicolbolas'
    elif number == 16:
        rsl = 'chandra'
    elif number == 17:
        rsl = 'ajani'
    elif number == 18:
        rsl = 'jaya'
    elif number == 19:
        rsl = 'teferi'
    elif number == 20:
        rsl = 'vivien'
    elif number == 21:
        rsl = 'ugin'
    elif number == 22:
        rsl = 'samut'
    elif number == 23:
        rsl = 'obnixilis'
    elif number == 24:
        rsl = 'tezzeret'
    elif number == 25:
        rsl = 'sarkhan'
    elif number == 26:
        rsl = 'kasmina'
    elif number == 27:
        rsl = 'teyo'
    elif number == 28:
        rsl = 'ral'
    elif number == 29:
        rsl = 'kiora'
    elif number == 30:
        rsl = 'sorin'
    elif number == 31:
        rsl = 'narset'
    elif number == 32:
        rsl = 'karn'
    elif number == 33:
        rsl = 'nahiri'
    elif number == 34:
        rsl = 'tamiyo'
    return rsl.capitalize()

# The acts are chosen as random
act1 = random.randint(0,1)
act2 = random.randint(0,1)
act3 = random.randint(0,1)

# A function is defined to determine the ally in different acts
def Ally(act=''):
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
    'Character': StringToNumber(character)
    ,'Act': 0
    ,'Act 1': act1
    ,'Act 2': act2
    ,'Act 3': act3
    ,'Allied': Ally('Acto 1')
    ,'Red': red
    ,'White': white
    ,'Black': black
    ,'Green': green
    ,'Blue': blue
}

Act2_X = {
    'Character': StringToNumber(character)
    ,'Act': 1
    ,'Act 1': act1
    ,'Act 2': act2
    ,'Act 3': act3
    ,'Allied': Ally('Acto 2')
    ,'Red': red
    ,'White': white
    ,'Black': black
    ,'Green': green
    ,'Blue': blue
}

Act3_X = {
    'Character': StringToNumber(character)
    ,'Act': 2
    ,'Act 1': act1
    ,'Act 2': act2
    ,'Act 3': act3
    ,'Allied': Ally('Acto 3')
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

# The number predicted is passed to a string

# Then the story is told
# The next part is pseudo code
if story == 'red':
    print("Story: " + story + ", Character: " + character.capitalize() + ", Predicted: " + NumberToString(predAct1))
elif story == 'black':
    print("Black Story With Model")
elif story == 'white':
    print("White Story With Model")
elif story == 'green':
    print("Green Story With Model")
elif story == 'blue':
    print("Blue Story With Model")