# Import necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer  # For converting text to a matrix of TF-IDF features
from sklearn.naive_bayes import MultinomialNB  # Naive Bayes classifier for multi-class classification
from sklearn.pipeline import make_pipeline  # For creating a pipeline to streamline the process

# Define training data
# These are example sentences that the chatbot will learn to recognize.
training_sentences = [
    "hello", "hi", "hey", "how are you",  # Greeting phrases
    "goodbye", "bye", "see you later",  # Farewell phrases
    "what's the weather like", "tell me the weather", 
    "how's the weather today", "weather forecast",  # Weather-related inquiries
    "what time is it", "what's your name", 
    "how can you help me", "thank you", "thanks"  # General inquiries and expressions of gratitude
]

# Define the corresponding labels for the training sentences
# Each sentence is associated with a label indicating its intent.
training_labels = [
    "greeting", "greeting", "greeting", "greeting",  # All greetings labeled as "greeting"
    "farewell", "farewell", "farewell",  # All farewells labeled as "farewell"
    "weather", "weather", "weather", "weather",  # All weather inquiries labeled as "weather"
    "time", "name",  # Additional intents for time and name inquiries
    "help", "thanks", "thanks"  # Help and thanks labels for general inquiries
]

# Create a machine learning model pipeline
# This pipeline will combine the TF-IDF vectorizer and the Naive Bayes classifier.
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model using the training data
# The model learns to classify the sentences based on their corresponding labels.
model.fit(training_sentences, training_labels)

# Function to predict the intent of the user input
def predict_intent(text):
    # Use the trained model to predict the intent of the given text
    return model.predict([text])[0]  # Return the predicted label

# Function to generate a chatbot response based on user input
def chatbot_response(user_input):
    # Predict the intent of the user input
    intent = predict_intent(user_input)  
    # Dictionary mapping intents to responses
    responses = {
        "greeting": "Hi! How can I assist you today?",  # Response for greetings
        "farewell": "Goodbye! Have a great day.",  # Response for farewells
        "weather": "It's sunny today! The weather is great!",  # Response for weather inquiries
        "time": "I can't tell the time, but you can check your device.",  # Response for time inquiries
        "name": "I'm your friendly chatbot!",  # Response for name inquiries
        "help": "I can assist with weather inquiries, greetings, and more.",  # Response for help inquiries
        "thanks": "You're welcome! Let me know if you need anything else."  # Response for thanks
    }
    # Return the response based on the predicted intent; if not recognized, return a default message
    return responses.get(intent, "Sorry, I don't understand that.")  

# Main execution block
if __name__ == "__main__":
    # Print a startup message indicating the chatbot is ready
    print("Chatbot is running! Type 'exit' to stop.")
    # Start an infinite loop to interact with the user
    while True:
        # Get user input from the terminal
        user_input = input("You: ")  
        # Check if the user wants to exit the loop
        if user_input.lower() == 'exit':  
            print("Chatbot is shutting down.")  # Inform the user the chatbot is shutting down
            break  # Exit the loop
        # Get the chatbot's response based on user input
        response = chatbot_response(user_input)  
        # Print the chatbot's response to the terminal
        print(f"Bot: {response}")  
