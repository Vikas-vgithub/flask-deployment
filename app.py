from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

# Load and prepare your data outside the route to avoid reloading it on each request
data = pd.read_csv('templates/Crop_production.csv')
data.drop('Crop', axis=1, inplace=True)

# Encode labels
label_encoder_crop = LabelEncoder()
data['Recommended_Crop'] = label_encoder_crop.fit_transform(data['Recommended_Crop'])
label_encoder_fertilizer = LabelEncoder()
data['Fertilizer'] = label_encoder_fertilizer.fit_transform(data['Fertilizer'])

# Train your model here similarly to your Tkinter app
conditions = {
    'apple': (500, 1000),
    'rice': (115, 500),
    'arecanut': (950, 4500),
    'banana': (1700, 2300),
    'maize': (50, 100),
    'barley': (180, 400),
    'sugarcane': (300, 800),
    'cotton': (500, 1000),
    'jute': (1000, 1500),
    'potato': (350, 550)
}

filtered_data = pd.concat([data[(data['rainfall'] >= min_rainfall) & (data['rainfall'] <= max_rainfall)]
                           for crop, (min_rainfall, max_rainfall) in conditions.items()])

X = filtered_data.drop(['Recommended_Crop', 'Fertilizer'], axis=1)
y_crop = filtered_data['Recommended_Crop']
y_fertilizer = filtered_data['Fertilizer']

crop_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
crop_classifier.fit(X, y_crop)

fertilizer_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
fertilizer_classifier.fit(X, y_fertilizer)

@app.route('/', methods=['GET', 'POST'])
def index():
    # Default values to display no prediction
    predicted_crop = None
    predicted_fertilizer = None
    
    if request.method == 'POST':
        # Extract data from the form
        user_input = request.form
        nitrogen = float(user_input['nitrogen'])
        phosphorus = float(user_input['phosphorus'])
        potassium = float(user_input['potassium'])
        moisture = float(user_input['moisture'])
        rainfall = float(user_input['rainfall'])
        area = float(user_input['area'])
        
        predicted_crop, predicted_fertilizer = predict({
            'N': nitrogen, 'P': phosphorus, 'K': potassium,
            'M': moisture, 'rainfall': rainfall, 'Area_in_hectares': area
        })
        
    # Render the template with the prediction results (or None if not yet predicted)
    return render_template('index.html', predicted_crop=predicted_crop, predicted_fertilizer=predicted_fertilizer)


def predict(user_input):
    user_df = pd.DataFrame([user_input])
    predicted_crop = crop_classifier.predict(user_df)
    predicted_fertilizer = fertilizer_classifier.predict(user_df)
    predicted_crop_label = label_encoder_crop.inverse_transform(predicted_crop)
    predicted_fertilizer_label = label_encoder_fertilizer.inverse_transform(predicted_fertilizer)
    return predicted_crop_label[0], predicted_fertilizer_label[0]

if __name__ == '__main__':
    app.run(debug=True)
