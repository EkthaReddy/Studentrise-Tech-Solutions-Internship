# Studentrise Tech Solutions Internship

This intership provides tasks realted to AIs. In those tasks we need to create the model for each assigned tasks.

<details>
  
<summary><h3>Spam Email Classifier</h3></summary>

Creating an AI model to classify emails as spam or not spam involves several steps, including data collection, preprocessing, feature extraction, and model training. Here's a step-by-step guide to building such a classifier:

**Step 1:** Data Collection
You can use the Enron email dataset, which is publicly available and commonly used for email classification tasks.

**Step 2:** Data Preprocessing
Preprocessing involves cleaning the text data to make it suitable for feature extraction and model training. Common preprocessing steps include:

Removing HTML tags: Emails often contain HTML content.

Removing special characters and numbers: These are usually not helpful for text classification.

Converting text to lowercase: This ensures uniformity.

Removing stop words: Words like "and", "the", etc., which are not useful for classification.

Stemming or lemmatization: Reducing words to their base forms.

**Step 3:** Feature Extraction
Feature extraction involves converting text data into numerical features that can be used by machine learning algorithms. Common techniques include:

Bag of Words (BoW): Representing text as a set of word frequencies.

Term Frequency-Inverse Document Frequency (TF-IDF): A more refined version of BoW that accounts for the importance of words.

Word Embeddings: Using pre-trained models like Word2Vec or GloVe to convert words into dense vectors.

**Step 4:** Model Training
Several machine learning algorithms can be used to train the classifier, including:

Logistic Regression: A simple and effective baseline model.
Naive Bayes: Particularly suited for text classification tasks.
Support Vector Machines (SVM): Effective for high-dimensional spaces.
Random Forest: An ensemble method that is robust and powerful.
Neural Networks: Including simple feedforward networks or more complex architectures like LSTM or Transformers for advanced performance.
**Step 5:** Model Evaluation
Evaluating the performance of your model using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.

**Step 6:** Deployment
Deploying the model in a production environment where it can classify incoming emails in real-time.

Here’s a simplified implementation using Python and common libraries like Scikit-learn, NLTK, and Pandas:

python
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

# Load the dataset (assuming it's already downloaded and extracted)
emails = pd.read_csv('enron_emails.csv')

# Preprocessing function
def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and numbers
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# Apply preprocessing
emails['clean_text'] = emails['message'].apply(preprocess_text)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(emails['clean_text']).toarray()
y = emails['label']  # Assuming 'label' is the column with spam/ham labels

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='spam')
recall = recall_score(y_test, y_pred, pos_label='spam')
f1 = f1_score(y_test, y_pred, pos_label='spam')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
```

</details>

---

<details>
<summary><h3>Handwritten Digit Recognizer</h3></summary>
To develop a handwritten digit recognizer using the MNIST dataset, you can follow these steps:

1. **Set Up the Environment:**
   Ensure you have the necessary libraries installed. You will need TensorFlow/Keras for the neural network, and Tkinter or another GUI library for the interface.
   ```bash
   pip install tensorflow numpy matplotlib
   ```

2. **Load and Preprocess the MNIST Dataset:**
   Load the dataset, normalize the pixel values, and split the data into training and testing sets.
   ```python
   import tensorflow as tf
   from tensorflow.keras.datasets import mnist

   (x_train, y_train), (x_test, y_test) = mnist.load_data()
   x_train, x_test = x_train / 255.0, x_test / 255.0

   x_train = x_train.reshape(-1, 28, 28, 1)
   x_test = x_test.reshape(-1, 28, 28, 1)
   ```

3. **Build the CNN Model:**
   Define a convolutional neural network using Keras.
   ```python
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

   model = Sequential([
       Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
       MaxPooling2D(pool_size=(2, 2)),
       Conv2D(64, kernel_size=(3, 3), activation='relu'),
       MaxPooling2D(pool_size=(2, 2)),
       Flatten(),
       Dense(128, activation='relu'),
       Dense(10, activation='softmax')
   ])

   model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
   ```

4. **Train the Model:**
   Train the model with the training data.
   ```python
   model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
   ```

5. **Evaluate the Model:**
   Evaluate the model's performance on the test data.
   ```python
   test_loss, test_acc = model.evaluate(x_test, y_test)
   print(f'Test accuracy: {test_acc}')
   ```

6. **Create the GUI:**
   Use Tkinter to create a simple GUI where users can draw digits.
   ```python
   import tkinter as tk
   from PIL import Image, ImageDraw, ImageOps
   import numpy as np

   class DigitRecognizerApp(tk.Tk):
       def __init__(self, model):
           super().__init__()
           self.title("Handwritten Digit Recognizer")
           self.canvas = tk.Canvas(self, width=200, height=200, bg='white')
           self.canvas.pack()
           self.button_predict = tk.Button(self, text="Predict", command=self.predict_digit)
           self.button_predict.pack()
           self.button_clear = tk.Button(self, text="Clear", command=self.clear_canvas)
           self.button_clear.pack()
           self.model = model
           self.image = Image.new('L', (200, 200), 255)
           self.draw = ImageDraw.Draw(self.image)
           self.canvas.bind('<B1-Motion>', self.paint)

       def paint(self, event):
           x1, y1 = (event.x - 5), (event.y - 5)
           x2, y2 = (event.x + 5), (event.y + 5)
           self.canvas.create_oval(x1, y1, x2, y2, fill='black')
           self.draw.ellipse([x1, y1, x2, y2], fill='black')

       def predict_digit(self):
           image = self.image.resize((28, 28)).convert('L')
           image = ImageOps.invert(image)
           image = np.array(image) / 255.0
           image = image.reshape(1, 28, 28, 1)
           prediction = self.model.predict(image)
           digit = np.argmax(prediction)
           tk.messagebox.showinfo("Prediction", f"The digit is: {digit}")

       def clear_canvas(self):
           self.canvas.delete("all")
           self.draw.rectangle([0, 0, 200, 200], fill='white')

   if __name__ == "__main__":
       app = DigitRecognizerApp(model)
       app.mainloop()
   ```
</details>

------
<details>
<summary><h3>Chatbot for Customer Support</h3></summary>

  To build an AI-powered chatbot for customer support, follow these steps:

1. **Set Up Environment:**
   Ensure you have the necessary tools and libraries. You will need a pre-trained language model API (like OpenAI GPT-3 or Dialogflow), a backend framework (like Flask or Django), and a frontend framework (like React or a mobile framework).

2. **Choose and Set Up the Language Model:**
   - **OpenAI GPT-3:**
     - Sign up and get API access from OpenAI.
     - Install the OpenAI Python client.
       ```bash
       pip install openai
       ```
     - Create a script to handle requests to the GPT-3 API.
       ```python
       import openai

       openai.api_key = 'your_openai_api_key'

       def get_gpt3_response(prompt):
           response = openai.Completion.create(
               engine="text-davinci-003",
               prompt=prompt,
               max_tokens=150
           )
           return response.choices[0].text.strip()
       ```

   - **Google Dialogflow:**
     - Set up a Dialogflow project and create an agent.
     - Integrate Dialogflow with your backend using its API.

3. **Build the Backend:**
   Use a backend framework like Flask to handle user queries and interact with the language model API.
   ```python
   from flask import Flask, request, jsonify
   from your_gpt3_module import get_gpt3_response  # Import the function you created

   app = Flask(__name__)

   @app.route('/chat', methods=['POST'])
   def chat():
       user_message = request.json.get('message')
       response = get_gpt3_response(user_message)
       return jsonify({'response': response})

   if __name__ == '__main__':
       app.run(debug=True)
   ```

4. **Build the Frontend:**
   Use a frontend framework like React to create a user interface for the chatbot.
   ```jsx
   // App.js (React)
   import React, { useState } from 'react';
   import axios from 'axios';

   function App() {
       const [message, setMessage] = useState('');
       const [response, setResponse] = useState('');

       const handleSend = async () => {
           const result = await axios.post('http://localhost:5000/chat', { message });
           setResponse(result.data.response);
       };

       return (
           <div className="App">
               <h1>Customer Support Chatbot</h1>
               <input
                   type="text"
                   value={message}
                   onChange={(e) => setMessage(e.target.value)}
               />
               <button onClick={handleSend}>Send</button>
               <p>{response}</p>
           </div>
       );
   }

   export default App;
   ```

5. **Deploy the Application:**
   - **Backend:** Deploy the Flask backend on a platform like Heroku, AWS, or any other cloud service.
   - **Frontend:** Deploy the React frontend on a platform like Netlify, Vercel, or any web hosting service.

6. **Integrate and Test:**
   Ensure the frontend and backend are properly integrated and test the chatbot with various customer support queries.

By following these steps, you can build and deploy an AI-powered chatbot for customer support that uses a pre-trained language model to handle and respond to user queries effectively.

</details>

------

<details>

<summary><h3>Movie Recommendation System</h3></summary>

To develop a movie recommendation system using collaborative filtering or content-based filtering, follow these steps:

1. **Set Up the Environment:**
   Ensure you have the necessary libraries installed. You will need pandas, NumPy, and scikit-learn for data processing and modeling.
   ```bash
   pip install pandas numpy scikit-learn
   ```

2. **Collect and Preprocess the Data:**
   Use the MovieLens dataset, which is a widely used dataset for movie recommendations. Download the dataset and preprocess it.
   ```python
   import pandas as pd

   # Load datasets
   ratings = pd.read_csv('path/to/ratings.csv')
   movies = pd.read_csv('path/to/movies.csv')

   # Merge datasets
   data = pd.merge(ratings, movies, on='movieId')
   ```

3. **Implement Collaborative Filtering:**
   - **User-Based Collaborative Filtering:**
     Calculate the similarity between users and recommend movies based on similar users' preferences.
     ```python
     from sklearn.metrics.pairwise import cosine_similarity

     # Create user-item matrix
     user_item_matrix = data.pivot_table(index='userId', columns='title', values='rating')

     # Fill NaN with 0
     user_item_matrix = user_item_matrix.fillna(0)

     # Calculate cosine similarity between users
     user_similarity = cosine_similarity(user_item_matrix)

     # Convert to DataFrame
     user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

     def get_user_based_recommendations(user_id, n_recommendations=5):
         # Get similar users
         similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:]
         # Get movies rated by similar users
         similar_users_ratings = user_item_matrix.loc[similar_users]
         # Calculate weighted average of ratings
         recommended_movies = similar_users_ratings.mean(axis=0).sort_values(ascending=False).head(n_recommendations)
         return recommended_movies.index.tolist()

     # Example usage
     print(get_user_based_recommendations(user_id=1))
     ```

   - **Item-Based Collaborative Filtering:**
     Calculate the similarity between items (movies) and recommend movies similar to ones the user has liked.
     ```python
     # Transpose user-item matrix to item-user matrix
     item_user_matrix = user_item_matrix.T

     # Calculate cosine similarity between movies
     item_similarity = cosine_similarity(item_user_matrix)

     # Convert to DataFrame
     item_similarity_df = pd.DataFrame(item_similarity, index=item_user_matrix.index, columns=item_user_matrix.index)

     def get_item_based_recommendations(movie_title, n_recommendations=5):
         # Get similar movies
         similar_movies = item_similarity_df[movie_title].sort_values(ascending=False).index[1:n_recommendations+1]
         return similar_movies.tolist()

     # Example usage
     print(get_item_based_recommendations(movie_title='Toy Story (1995)'))
     ```

4. **Implement Content-Based Filtering:**
   Calculate the similarity between movies based on their content features (e.g., genres, directors, actors).
   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer
   from sklearn.metrics.pairwise import linear_kernel

   # Create TF-IDF matrix for movie genres
   tfidf = TfidfVectorizer(stop_words='english')
   tfidf_matrix = tfidf.fit_transform(movies['genres'])

   # Calculate cosine similarity between movies
   cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

   # Create a series with movie titles as index
   indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

   def get_content_based_recommendations(movie_title, n_recommendations=5):
       idx = indices[movie_title]
       sim_scores = list(enumerate(cosine_sim[idx]))
       sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
       sim_scores = sim_scores[1:n_recommendations+1]
       movie_indices = [i[0] for i in sim_scores]
       return movies['title'].iloc[movie_indices].tolist()

   # Example usage
   print(get_content_based_recommendations(movie_title='Toy Story (1995)'))
   ```

5. **Evaluation and Deployment:**
   Evaluate the performance of your recommendation system using metrics such as precision, recall, or RMSE. Once satisfied with the performance, you can deploy the system using a web framework like Flask or Django, or integrate it into an existing application.

By following these steps, you can create a movie recommendation system using collaborative filtering or content-based filtering. Adjust the algorithms and parameters as needed to optimize the recommendations.

</details>

------

<details>

<summary><h3>Face Recognition System</h3></summary>

To build an AI-based face recognition system that can detect and recognize faces in real-time, follow these steps:

1. **Set Up the Environment:**
   Ensure you have the necessary libraries installed. You will need OpenCV, dlib, and face recognition libraries.
   ```bash
   pip install opencv-python dlib face_recognition
   ```

2. **Face Detection:**
   Use a pre-trained face detection model such as MTCNN or Haar Cascades.
   ```python
   import cv2

   # Load Haar Cascade for face detection
   face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

   def detect_faces(image):
       gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
       return faces
   ```

3. **Face Recognition:**
   Use a pre-trained face recognition model like FaceNet or VGGFace. The `face_recognition` library uses dlib's facial recognition capabilities.
   ```python
   import face_recognition
   import numpy as np

   def encode_faces(image):
       face_encodings = face_recognition.face_encodings(image)
       return face_encodings

   # Load known faces and their encodings
   known_face_encodings = []
   known_face_names = []

   # Example: Load an image and encode the face
   image = face_recognition.load_image_file("path/to/known_person.jpg")
   encoding = face_recognition.face_encodings(image)[0]
   known_face_encodings.append(encoding)
   known_face_names.append("Person Name")
   ```

4. **Recognize Faces in Real-Time:**
   Use OpenCV to capture video and recognize faces in real-time.
   ```python
   video_capture = cv2.VideoCapture(0)

   while True:
       ret, frame = video_capture.read()
       rgb_frame = frame[:, :, ::-1]

       # Detect faces
       faces = detect_faces(frame)

       # Recognize faces
       face_encodings = encode_faces(rgb_frame)
       face_names = []

       for face_encoding in face_encodings:
           matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
           name = "Unknown"

           face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
           best_match_index = np.argmin(face_distances)
           if matches[best_match_index]:
               name = known_face_names[best_match_index]

           face_names.append(name)

       # Display the results
       for (x, y, w, h), name in zip(faces, face_names):
           cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
           cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

       cv2.imshow('Video', frame)

       if cv2.waitKey(1) & 0xFF == ord('q'):
           break

   video_capture.release()
   cv2.destroyAllWindows()
   ```

5. **Training Your Own Face Recognition Model:**
   If you want to train your own model, you can use a deep learning framework like TensorFlow or PyTorch. Here's an example using FaceNet with TensorFlow:
   ```python
   import tensorflow as tf
   from tensorflow.keras.models import load_model

   # Load pre-trained FaceNet model
   model = load_model('path/to/facenet_model.h5')

   def get_face_embedding(model, face_pixels):
       face_pixels = face_pixels.astype('float32')
       mean, std = face_pixels.mean(), face_pixels.std()
       face_pixels = (face_pixels - mean) / std
       samples = np.expand_dims(face_pixels, axis=0)
       yhat = model.predict(samples)
       return yhat[0]
   ```

By following these steps, you can build an AI-based face recognition system that detects and recognizes faces in real-time. Adjust the model and parameters according to your specific requirements.
</details>

-------
