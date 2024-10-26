import csv
import subprocess
import json

# Path to input CSV and output JSON file
input_file = 'training_data.csv'
output_file = 'enhanced_training_data.json'

# Function to call LLaMa model for each row with the required format


def call_llama_for_prompt(text):
    # Define the prompt
    prompt = f"""
    Transform the following text into structured format (add any additional fields if needed):

    Text: {text}

    Required Output:
    {{
        'text': '{text}',
        'polarity': 'Positive/Negative/Neutral',
        'extracted_concern': 'Key concern phrase',
        'category': 'Primary category',
        'intensity': int,  # Scale of 1-10
        'secondary_category': 'Secondary category',
        'keywords': ['key', 'phrases']
    }}

    Here is an example of the expected output:
    {{
        'text': 'Panic attacks are becoming more frequent and intense',
        'polarity': 'Negative',
        'extracted_concern': 'panic attacks',
        'category': 'Panic Attacks',
        'secondary_category': 'Anxiety (General)',
        'intensity': 9,
        'keywords': ['panic attacks', 'frequent', 'intense']
    }},

    Here are the available categories (only choose among these):
    Anxiety (General)
    Health Anxiety
    Social Anxiety
    Depression
    Eating Disorders
    Insomnia/Sleep Issues
    Career Confusion
    Academic Stress
    Relationship Issues
    Self-esteem Issues
    Financial Stress
    Work-related Stress
    Family Issues
    Addiction
    PTSD/Trauma
    Positive Outlook (for positive experiences)
    Burnout
    Grief/Loss
    Identity Issues
    Panic Attacks
    """
    try:
        # Call the LLaMa model via subprocess (assuming llama.cpp or similar CLI-based)
        result = subprocess.run(['ollama', 'run', 'llama3'],
                                input=prompt, text=True, capture_output=True, check=True)
        # Parse the JSON output from LLaMa response
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error calling LLaMa:", e)
        return None


# Read the CSV file and process each row
training_data = []
with open(input_file, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        text = row['text']
        enhanced_data = call_llama_for_prompt(text)
        if enhanced_data:
            training_data.append(enhanced_data)

# Save the enhanced training data to a new JSON file
with open(output_file, 'w') as jsonfile:
    json.dump(training_data, jsonfile, indent=4)

print(f"Enhanced training data has been saved to {output_file}")
