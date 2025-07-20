# personal_assistant.py

# Step 1: Data Collection and Preparation
import pandas as pd
from sklearn.model_selection import train_test_split
data = {
    'text': [
        'Remind me to call mom at 2 PM',
        'Start the pop playlist',
        'Dim the bedroom lights',
        'How is the weather in New York today?',
        'Turn on the lights in the kitchen',
        'Play some classical music',
        'Do I need an umbrella in Tokyo?',
        'Switch off the living room light',
        'Tell me the forecast for Melbourne',
        'Set a reminder to attend meeting at 6 PM'
    ],
    'intent': [
        'set_reminder',
        'play_music',
        'control_lights',
        'get_weather',
        'control_lights',
        'play_music',
        'get_weather',
        'control_lights',
        'get_weather',
        'set_reminder'
    ]
}



df = pd.DataFrame(data)
df['text'] = df['text'].apply(lambda x: x.lower())
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['intent'], test_size=0.2, random_state=42)

# Step 2: NLP Model Development
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
model = BertForSequenceClassification.from_pretrained('./results')  # or reuse the in-memory trained model
model.eval()

train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=128)

train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': [0 if intent == 'set_reminder' else 1 if intent == 'get_weather' else 2 if intent == 'play_music' else 3 for intent in y_train]
})

test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': [0 if intent == 'set_reminder' else 1 if intent == 'get_weather' else 2 if intent == 'play_music' else 3 for intent in y_test]
})

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

# Step 3: Integration with External Services (Weather API)
import requests

def get_weather(city):
    api_key = '060df0fe0f7003ea196574ae7c06fb40'
    base_url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}'
    response = requests.get(base_url)
    data = response.json()
    if data['cod'] == 200:
        temperature = data['main']['temp']
        weather_description = data['weather'][0]['description']
        return f"The temperature in {city} is {temperature}K with {weather_description}."
    else:
        return "City not found."
    

def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    # Force to CPU
    model.to("cpu")
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()

    id2label = {0: 'set_reminder', 1: 'get_weather', 2: 'play_music', 3: 'control_lights'}
    return id2label.get(predicted_class_id, 'unknown')



# Step 4: User Interface Design and Development using Flask
from flask import Flask, request, jsonify
import logging

app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask():
    try:
        user_input = request.json.get('query')
        predicted_intent = predict_intent(user_input)
        response = f"The predicted intent is: {predicted_intent}"
        return jsonify({'response': response})
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return jsonify({'response': 'An error occurred'}), 500

    


@app.route('/ask_web', methods=['GET'])
def ask_web():
    try:
        user_input = request.args.get('query')
        if not user_input:
            return jsonify({'error': 'Missing query parameter'}), 400

        predicted_intent = predict_intent(user_input)

        # Action based on intent
        if predicted_intent == 'get_weather':
            # extract location (very basic way)
            import re
            city_match = re.search(r'in ([a-zA-Z\s]+)', user_input.lower())
            city = city_match.group(1).strip() if city_match else 'New York'
            response = get_weather(city)

        elif predicted_intent == 'play_music':
            response = "🎵 Playing your favorite song!"

        elif predicted_intent == 'set_reminder':
            response = "⏰ Reminder set!"

        elif predicted_intent == 'control_lights':
            response = "💡 Lights adjusted!"

        else:
            response = f"The predicted intent is: {predicted_intent}"

        return jsonify({'response': response})

    except Exception as e:
        import traceback
        traceback.print_exc()
        logging.error(f"Error occurred: {e}")
        return jsonify({'response': 'An error occurred'}), 500




# Step 5: Testing and Deployment
def test_preprocess():
    assert "Hello, World!".lower() == "hello, world!"


def test_get_weather():
    result = get_weather("New York")
    assert "temperature in New York" in result

# Run tests
test_preprocess()
test_get_weather()  # Uncomment after adding valid API key

# Step 6: Iteration and Improvement
# You can call trainer.train() again with more data if needed

# Step 7: Maintenance and Monitoring
logging.basicConfig(filename='assistant.log', level=logging.DEBUG)

if __name__ == '__main__':
    app.run(debug=True)
