import pandas as pd

# Read the CSV file
df = pd.read_csv('DataMinor/sampleDataset.csv')

# Select only the first two columns
newData = df.iloc[:, :2]

# Define the classify_input function
def classify_input(text):
    text = text.lower()
    if 'anxious' in text or 'anxiety' in text or 'worried' in text:
        return 'Anxiety'
    elif 'happy' in text or 'excited' in text or 'joy' in text or 'hopeful' in text:
        return 'None'
    elif 'sleep' in text or 'tired' in text or 'fatigue' in text:
        return 'Insomnia'
    elif 'stress' in text or 'stressed' in text or 'pressure' in text or 'tension' in text or 'work' in text or 'job' in text:
        return 'Stress'
    elif 'feeling low' in text or 'sad' in text or 'feeling down' in text or 'feeling very low' in text:
        return 'Sad'
    elif 'feeling good' in text or 'feeling great' in text or 'feeling much better' in text or 'feeling better' in text:
        return 'Happy'
    elif 'worse' in text or 'depressed' in text or 'depression' in text:
        return 'Depression'
    elif 'not eating' in text:
        return 'Low Self-Esteem'
    return None

# Apply the classify_input function to each element in the UserInput column
print(newData.columns)
if 'UserInput' in newData.columns:
    print("Hi")
    newData['Class'] = newData['UserInput'].apply(classify_input)

# Display the result
print(newData)

# Total NaN values in Class [ print first 5 such rows ]
print(newData[newData['Class'].isnull()].head())
print("Total NaN values in Class: ", newData['Class'].isnull().sum())

# Save the result to a new CSV file
newData.to_csv('DataMinor/finalDataSet.csv', index=False)