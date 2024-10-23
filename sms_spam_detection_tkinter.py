import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import tkinter as tk
from tkinter import ttk

# Load and preprocess the dataset
data_file = 'SMSSpamCollection'  # Replace with your file path
with open(data_file, 'r', encoding='latin-1') as f:
    data = f.readlines()

messages = []
labels = []
for line in data:
    label, message = line.split('\t', 1)
    labels.append(label.strip())
    messages.append(message.strip())

# Create a DataFrame
df = pd.DataFrame({
    'label': labels,
    'message': messages
})

# Convert labels to binary (ham = 0, spam = 1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split the dataset
X = df['message']
y = df['label']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train Models
models = {
    "Naive Bayes": MultinomialNB(),
    "SVM": SVC(kernel='linear', probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Fit models
for model_name, model in models.items():
    model.fit(X_train_tfidf, y_train)


# Function to make predictions based on selected model
def classify_message(model_name, message):
    if not message.strip():
        return "Please enter a valid message"

    model = models[model_name]
    message_tfidf = tfidf.transform([message])
    prediction = model.predict(message_tfidf)

    print(f"Model: {model_name}, Prediction: {prediction}")  # Debugging print statement
    return "Spam" if prediction[0] == 1 else "Not Spam"


# Tkinter UI
def start_ui():
    def on_classify():
        sms_content = sms_entry.get("1.0", "end-1c")  # Get input from text box
        selected_model = model_combobox.get()  # Get selected model from dropdown

        print(f"User input: {sms_content}")  # Debugging print statement
        print(f"Selected model: {selected_model}")  # Debugging print statement

        if sms_content and selected_model:
            result = classify_message(selected_model, sms_content)
            result_label.config(text=f"Prediction: {result}", foreground="green" if result == "Not Spam" else "red")
        else:
            result_label.config(text="Please enter an SMS and select a model.", foreground="orange")

    # Create the main window
    root = tk.Tk()
    root.title("SMS Spam Detection")

    # Increase window size for better layout
    root.geometry("500x500")  # Increased height to 500px
    root.configure(bg="#F0F8FF")  # Light background color

    # Create frames to organize the layout
    frame_top = tk.Frame(root, bg="#F0F8FF")
    frame_top.pack(pady=20)

    frame_middle = tk.Frame(root, bg="#F0F8FF")
    frame_middle.pack(pady=10)

    frame_bottom = tk.Frame(root, bg="#F0F8FF")
    frame_bottom.pack(pady=20)

    # Title Label
    title_label = ttk.Label(frame_top, text="SMS Spam Classifier", font=("Helvetica", 18, "bold"), background="#F0F8FF")
    title_label.pack()

    # SMS Text Label and Entry Box
    sms_label = ttk.Label(frame_middle, text="Enter SMS message:", font=("Helvetica", 12), background="#F0F8FF")
    sms_label.pack(pady=5)

    sms_entry = tk.Text(frame_middle, height=6, width=50, font=("Arial", 10))
    sms_entry.pack(pady=5)

    # Dropdown for model selection
    model_label = ttk.Label(frame_middle, text="Select Model:", font=("Helvetica", 12), background="#F0F8FF")
    model_label.pack(pady=10)

    model_combobox = ttk.Combobox(frame_middle, values=list(models.keys()), state="readonly", font=("Arial", 10))
    model_combobox.pack(pady=5)
    model_combobox.set("Naive Bayes")  # Default model selection

    # Classify Button
    classify_button = ttk.Button(frame_bottom, text="Classify", command=on_classify, style="W.TButton")
    classify_button.pack(pady=10)

    # Result Label
    result_label = ttk.Label(frame_bottom, text="Prediction: ", font=("Helvetica", 14, "bold"), background="#F0F8FF")
    result_label.pack(pady=10)

    # Styling for Buttons
    style = ttk.Style()
    style.configure("W.TButton", font=("Helvetica", 12), padding=6, relief="raised")

    # Start the Tkinter main loop
    root.mainloop()


# Start the UI
if __name__ == "__main__":
    start_ui()
