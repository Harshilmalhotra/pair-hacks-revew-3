import ast
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import XLMRobertaTokenizerFast, XLMRobertaForTokenClassification, Trainer, TrainingArguments

# Define your label list here
label_list = ["PERSON", "PHONE_NUMBER", "EMAIL", "URL", "ORGANIZATION", "LOCATION", "DATE", "JOB_TITLE", "PRODUCT", "WORK_OF_ART"]

# Load your dataset
data = {
    "train": [
        {"text": "Aditya Kulshrestha Ó +91 78408 69129 R hello@adikul.dev", "entities": "[[0, 17, 'PERSON'], [22, 37, 'PHONE_NUMBER'], [39, 54, 'EMAIL']]"},
        {"text": "Neel Bhalla NeelBhallaBos@gmail.com (978) 831 8115 Linkedin ⇗ NeelBhalla2021 GitHub ⇗ ashdawngary", "entities": "[[0, 10, 'PERSON'], [12, 30, 'EMAIL'], [32, 45, 'PHONE_NUMBER'], [56, 69, 'URL'], [72, 87, 'URL']]"},
        {"text": "Chewy, Boston, MA Feb 2023 - Present Software Engineer 1", "entities": "[[0, 6, 'ORGANIZATION'], [8, 14, 'LOCATION'], [17, 19, 'DATE'], [22, 31, 'DATE'], [33, 47, 'JOB_TITLE']]"},
        {"text": "SRM Institute of Science and Technology Kattankulathur, IN", "entities": "[[0, 34, 'ORGANIZATION'], [36, 52, 'LOCATION']]"},
        {"text": "I work at Google, an American multinational technology company.", "entities": "[[11, 17, 'ORGANIZATION']]"},
        {"text": "John Doe is the CEO of XYZ Corp., a major player in the tech industry.", "entities": "[[0, 8, 'PERSON'], [16, 25, 'ORGANIZATION']]"},
        {"text": "The university, MIT, is located in Cambridge, MA.", "entities": "[[14, 17, 'ORGANIZATION'], [30, 39, 'LOCATION']]"},
        {"text": "Please visit our website at http://example.com for more details.", "entities": "[[29, 45, 'URL']]"},
        {"text": "Dr. Jane Smith received her degree from Harvard University in 2010.", "entities": "[[0, 12, 'PERSON'], [31, 49, 'ORGANIZATION']]"},
        {"text": "I had a meeting with Sarah Connor yesterday at 2 PM.", "entities": "[[16, 28, 'PERSON']]"},
        {"text": "The new office is in San Francisco, CA.", "entities": "[[20, 36, 'LOCATION']]"},
        {"text": "Apple Inc. was founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne.", "entities": "[[0, 11, 'ORGANIZATION'], [37, 47, 'PERSON'], [49, 60, 'PERSON'], [62, 74, 'PERSON']]"},
        {"text": "My favorite restaurant is The French Laundry, located in Yountville, CA.", "entities": "[[20, 38, 'ORGANIZATION'], [48, 59, 'LOCATION']]"},
        {"text": "Amazon is an e-commerce and cloud computing giant.", "entities": "[[0, 6, 'ORGANIZATION']]"},
        {"text": "The film 'Inception' was directed by Christopher Nolan.", "entities": "[[9, 18, 'WORK_OF_ART'], [32, 47, 'PERSON']]"},
        {"text": "The event will take place at the Central Park in New York.", "entities": "[[22, 32, 'LOCATION'], [37, 45, 'LOCATION']]"},
        {"text": "Elon Musk is the founder of SpaceX and Tesla, Inc.", "entities": "[[0, 10, 'PERSON'], [23, 28, 'ORGANIZATION'], [33, 39, 'ORGANIZATION']]"},
        {"text": "The Eiffel Tower is located in Paris, France.", "entities": "[[4, 16, 'LOCATION'], [23, 28, 'LOCATION']]"},
        {"text": "The iPhone 14 is a popular smartphone model by Apple.", "entities": "[[4, 14, 'PRODUCT'], [30, 35, 'ORGANIZATION']]"},
        {"text": "SRM Institute of Science & Technology, B.Tech CSE Secured 9.05/10.00 CGPA in First Semester", "entities": "[[0, 34, 'ORGANIZATION']]"},
        {"text": "Kreative Tech, Flutter Developer Intern Hands-on experience in mobile app development, with a keen focus on seamlessly integrating APIs and crafting robust, production-grade applications for optimal user experience.", "entities": "[[0, 13, 'ORGANIZATION'], [15, 30, 'JOB_TITLE']]"},
        {"text": "Grocery App Developed a dynamic grocery store mobile application using Flutter, providing users with an intuitive platform to browse and purchase groceries effortlessly.", "entities": "[[0, 12, 'PRODUCT']]"},
        {"text": "Human Detector Developed a versatile Python-based tool for real-time human detection and counting using the Faster R-CNN Inception v2 model.", "entities": "[[0, 13, 'PRODUCT']]"},
        {"text": "Stick & Dot, AI/ML Developer December 2023 – present | Bengaluru, IN", "entities": "[[0, 10, 'ORGANIZATION'], [12, 24, 'JOB_TITLE'], [27, 34, 'DATE'], [37, 39, 'DATE'], [42, 54, 'LOCATION']]"},
        {"text": "Oneso Technology, Burlington, MA June 2020 - March 2021 Software Developer", "entities": "[[0, 15, 'ORGANIZATION'], [17, 24, 'LOCATION'], [27, 33, 'DATE'], [36, 49, 'DATE'], [51, 67, 'JOB_TITLE']]"},
        {"text": "Fujitsu Research of India Private Limited Bengaluru Assistant Software Developer - QA team, Fujitsu Network Communications December 2022 - Present", "entities": "[[0, 37, 'ORGANIZATION'], [39, 52, 'LOCATION'], [55, 82, 'JOB_TITLE'], [85, 94, 'ORGANIZATION'], [97, 111, 'DATE'], [114, 122, 'DATE']]"},
        {"text": "The Deeva, Full Stack App Developer Involved in developing their new project's app version. Leading a team of 3 development interns.", "entities": "[[0, 9, 'ORGANIZATION'], [11, 33, 'JOB_TITLE']]"},
        {"text": "SRM Institute of Science & Technology, B.Tech CSE CGPA - 9.34", "entities": "[[0, 34, 'ORGANIZATION']]"},
        {"text": "SignWave, SwiftUI | Flutter | Firebase A sign language translator that takes video as input and uses machine learning to recognize signs, converting them into text.", "entities": "[[0, 9, 'PRODUCT']]"},
        {"text": "Nityam Sharma Flutter Developer nityamsharma02@gmail.com +919056282368 linkedin.com/in/nityamsharma-cse https://github.com/SharmaNityam", "entities": "[[0, 12, 'PERSON'], [13, 29, 'JOB_TITLE'], [31, 50, 'EMAIL'], [52, 66, 'PHONE_NUMBER'], [68, 91, 'URL'], [93, 110, 'URL']]"},
        {"text": "Shaurya Singh Srinet Software Development Engineer shauryasrinet@gmail.com 9999847323 Chennai, IN", "entities": "[[0, 18, 'PERSON'], [20, 48, 'JOB_TITLE'], [50, 68, 'EMAIL'], [70, 82, 'PHONE_NUMBER'], [84, 90, 'LOCATION']]"},
        {"text": "Surya Mallikarjuna Shivaji Balijepalli Email: bsmshivaji94@gmail.com Contact Number: +91-9618187761", "entities": "[[0, 26, 'PERSON'], [31, 51, 'EMAIL'], [56, 72, 'PHONE_NUMBER']]"},
        {"text": "Shinjan Patra Software Development Engineer Chennai, IN", "entities": "[[0, 13, 'PERSON'], [15, 44, 'JOB_TITLE'], [47, 53, 'LOCATION']]"},
        {"text": "Ranjan Singh Senior Software Engineer ranjansingh@xyz.com 9109999100", "entities": "[[0, 11, 'PERSON'], [13, 34, 'JOB_TITLE'], [36, 50, 'EMAIL'], [52, 62, 'PHONE_NUMBER']]"},
        {"text": "Krishna Vignesh S Mobile App Developer krishnavignesh.s@gmail.com +91-9999999999", "entities": "[[0, 16, 'PERSON'], [18, 38, 'JOB_TITLE'], [40, 63, 'EMAIL'], [65, 79, 'PHONE_NUMBER']]"},
        {"text": "Ravi Shankar Senior Data Scientist ravishankar@datainsights.com 9801234567 Bangalore, IN", "entities": "[[0, 12, 'PERSON'], [14, 35, 'JOB_TITLE'], [37, 65, 'EMAIL'], [67, 77, 'PHONE_NUMBER'], [79, 90, 'LOCATION']]"},
        {"text": "Aaditya Verma Full Stack Developer aadityaverma@webtech.com 8009876543 Pune, IN", "entities": "[[0, 13, 'PERSON'], [15, 35, 'JOB_TITLE'], [37, 60, 'EMAIL'], [62, 72, 'PHONE_NUMBER'], [74, 81, 'LOCATION']]"}
    ],
    "validation": [
        {"text": "John Doe is a software engineer at Google.", "entities": "[[0, 8, 'PERSON'], [23, 29, 'JOB_TITLE'], [35, 41, 'ORGANIZATION']]"},
        {"text": "Elon Musk founded SpaceX in 2002.", "entities": "[[0, 9, 'PERSON'], [17, 23, 'ORGANIZATION'], [27, 31, 'DATE']]"},
        {"text": "MIT is located in Cambridge, MA.", "entities": "[[0, 3, 'ORGANIZATION'], [15, 26, 'LOCATION']]"},
        {"text": "The Eiffel Tower is in Paris.", "entities": "[[4, 16, 'LOCATION'], [20, 25, 'LOCATION']]"},
        {"text": "Apple Inc. released the iPhone 14 in 2022.", "entities": "[[0, 11, 'ORGANIZATION'], [19, 30, 'PRODUCT'], [34, 38, 'DATE']]"}
    ]
}

# Convert to DataFrames
train_df = pd.DataFrame(data['train'])
validation_df = pd.DataFrame(data['validation'])

# Convert DataFrames to Dataset objects
dataset = DatasetDict({
    'train': Dataset.from_pandas(train_df),
    'validation': Dataset.from_pandas(validation_df)
})

# Initialize tokenizer and model
tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-large')
model = XLMRobertaForTokenClassification.from_pretrained('xlm-roberta-large', num_labels=len(label_list))

# Function to tokenize and align labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['text'], truncation=True, is_split_into_words=False, padding='max_length', max_length=128)

    labels = []
    for i, entity_str in enumerate(examples['entities']):
        entities = ast.literal_eval(entity_str)  # Convert string back to list
        word_ids = tokenized_inputs.word_ids(batch_index=i)

        # Ensure labels list matches the length of the tokenized input
        label_ids = [-100] * len(tokenized_inputs['input_ids'])

        for start, end, label_type in entities:
            start_word_id = word_ids[start] if start < len(word_ids) else None
            end_word_id = word_ids[end] if end < len(word_ids) else None

            if start_word_id is not None and end_word_id is not None:
                for idx in range(start, end + 1):
                    word_id = word_ids[idx]
                    if word_id is not None:
                        if word_id < len(label_ids):  # Ensure index is within bounds
                            label_ids[word_id] = label_list.index(label_type)

        labels.append(label_ids)

    tokenized_inputs['labels'] = labels
    return tokenized_inputs

# Tokenize and align labels
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=30,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("path_to_save_model")
tokenizer.save_pretrained("path_to_save_tokenizer")
