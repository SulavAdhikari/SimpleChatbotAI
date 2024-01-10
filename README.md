# this is just a simple AI chat bot

This chat bot has limited intends. it basically answers the question depending on what the questions and answers are in the dataset.
When you run the chatbot.py file and provide an input, the input is seperated in to bag of words and the AI model is used to predict the result. The result contains the intents from inents.json file and the probability of matching. After that the most probable intent is selected and one asnwer is randomly selected.

to build follow this:
```
pip install pyttsx3
pip install --user -U nltk
pip install tensorflow
# Edit the intents before training
python training.py
python chatbot.py
```
![image](https://github.com/SulavAdhikari/SimpleChatbotAI/assets/65087106/c52538ee-071c-42cc-ab54-7e3515b7dedd)
