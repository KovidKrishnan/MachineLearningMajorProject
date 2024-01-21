import json
import subprocess
from django.http import JsonResponse
from django.shortcuts import render
from django.views import View
import joblib

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os
import matplotlib
from django.core.files.storage import FileSystemStorage

def random_forest(request):
    
    matplotlib.use('Agg')
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(current_dir, 'twomer-random-forest-metrics.json')

    if os.path.exists(json_file_path):
        # If the JSON file exists, read metrics from the file
        with open(json_file_path, 'r') as json_file:
            results = json.load(json_file)
        return render(request, 'twomer-random-forest.html', results)

    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path to the CSV file
    csv_file_path = os.path.join(current_dir, 'Twomer Dataset.csv')

    # Load the CSV file
    df = pd.read_csv(csv_file_path)

    class_mapping = {'B': 0, 'S': 1}
    df['class'] = df['class'].map(class_mapping)
    
    # Split the dataset into features (X) and target variable (y)
    X = df.drop('class', axis=1)
    y = df['class']

    # Initialize the Random Forest classifier
    rf_model = RandomForestClassifier(random_state=42)

    # Train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf_model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, thresholds_pr = precision_recall_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
    fpr, tpr, thresholds_roc = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
    class_report = classification_report(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)

    # Plot precision-recall curve
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.plot(recall, precision, label='Precision-Recall Curve', color='blue')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()

    # Plot ROC curve
    plt.subplot(1, 3, 2)
    plt.plot(fpr, tpr, label='ROC Curve', color='red')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    # Plot Confusion Matrix with annotations
    plt.subplot(1, 3, 3)
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            plt.text(j, i, f"{conf_mat[i, j]:,}", ha='center', va='center', color='white')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Convert plot to base64 for embedding in HTML
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    
    results = {
        'accuracy': accuracy,
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'class_report': class_report,
        'conf_mat': conf_mat.tolist(),
        'plot_base64': plot_base64,
    }
    
    # Save metrics to a JSON file
    json_file_path = os.path.join(current_dir, 'twomer-random-forest-metrics.json')
    with open(json_file_path, 'w') as json_file:
        json.dump(results, json_file)

    # Render the HTML template
    return render(request, 'twomer-random-forest.html', results)


class UploadApkView(View):
    def get(self, request):
        return render(request, 'upload_apk.html')

class ProcessApkView(View):
    def post(self, request):
        # Handle APK file processing and prediction here
        # Retrieve the uploaded file using request.FILES['apk_file']

        if 'apk_file' not in request.FILES:
            predictions = {'message': 'Error', 'prediction': 'Unable to prdict'}
            return render(request, 'display_predictions.html',predictions)

        apk_file = request.FILES['apk_file']

        # Step 1: Save the APK file temporarily
        fs = FileSystemStorage()
        saved_path = fs.save(apk_file.name, apk_file)

        # Step 2: Process the APK file and get predictions (replace this with your processing logic)
        predictions = process_apk(apk_file.name)

        # Step 3: Delete the temporarily saved APK file
        os.remove(saved_path)

        # Render predictions in the response
        return render(request, 'display_predictions.html', {'predictions': predictions})

def process_apk(file_path):
    # Add your logic to process the APK file and return predictions
    # For example, you might want to use external tools like jadx for APK analysis

    # Placeholder: Example logic to read content from the APK file
    
    
    current_dir = os.path.dirname(os.path.abspath(__file__))

    print('step 1 start:')
    
    permissions = []
    with open(current_dir + '\permissions-list.txt', 'r') as file:
        permissions = list(file.read().split(','))
    
    
    print('step 2 start')
    def run_androguard_commands(commands):
        try:
            # Concatenate commands into a single string
            command_string = '\n'.join(commands)

            # Run the command in a subprocess, passing it to the shell
            process = subprocess.Popen(['androguard', 'analyze'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            output, error = process.communicate(input=command_string)

            return output, error
        except Exception as e:
            return None, str(e)

    # Example usage:
    androguard_commands = [
        "a, d, dx = AnalyzeAPK('" + file_path + "')\n'",
        'a.get_permissions()',
        # Add more commands as needed
    ]
    

    output, error = run_androguard_commands(androguard_commands)

    if output is None:
        predictions = {'message':error,'prediction': 'Unable to process the application'}
        return predictions
    # Find the substring between "['" and "']"
    start_marker = "['"
    end_marker = "']"

    print('step 3 start')
    start_index = output.find(start_marker)
    end_index = output.find(end_marker, start_index + len(start_marker))

    if start_index != -1 and end_index != -1:
        extracted_content = output[start_index + len(start_marker):end_index]
        sentences = [sentence.strip() for sentence in extracted_content.split(',')]
        
        # Write each sentence to a new line in a new file
        with open('uploaded_app_permissions.txt', 'w') as output_file:
            for sentence in sentences:
                last_word = sentence.split('.')[-1]
                output_file.write(f'"{last_word}"\n')
    else: 
        predictions = {'message':'Error','prediction': 'Unable to process the application'}
        return predictions
    
    print('step 4 start')
    permissions_list = []
    with open('uploaded_app_permissions.txt', 'r') as input_file:
        for line in input_file:
            permissions_list.append(line.replace("'","").replace('"','').strip())
            
    print(permissions_list)
    
    df = []

    print('step 5 start')
    # Iterate over pairs of permissions in app_perms
    for i in range(len(permissions_list)):
        for j in range(i + 1, len(permissions_list)):
            # Check if the combination exists in permissions_list
            a = permissions_list[i] + '-' + permissions_list[j]
            b = permissions_list[j] + '-' + permissions_list[i]
            df.append(a.strip())
            df.append(b.strip())

    X_new = pd.DataFrame([dict(zip(permissions, [1 if feature in df else 0 for feature in permissions]))])
            
                
    print('step 6 start')
    model_filename = 'D:\Academics\Major Project\Web\TwomerModels\Twomer-Random-Forest.joblib'
  # Replace with your actual model file path
    loaded_model = joblib.load(model_filename)

    # Assuming you have a DataFrame 'df' with the required permissions_list columns
    # Modify 'permissions_list' to match the column names in your DataFrame

    # Make predictions on the DataFrame
    class_label = loaded_model.predict(X_new)[0]
                
    # Placeholder: Example prediction result
    predictions = {'message': 'Predicted Class for the uploaded App',
        'prediction': 'Benign' if class_label == 0 else 'Malign'}
    
    return predictions