# 🩺 **Emergency Project**

This project was inspired by a real event:
when my grandfather needed urgent medical care, I noticed that the paramedic had to ask him many critical questions before starting treatment - questions that could have been answered in advance if his medical information were summarized and available.

Emergency Project aims to make that process faster and smarter.
It takes a patient's medical record, sends it to an AI model, and automatically generates a structured summary that the paramedic can review on the way to the patient, helping them provide more accurate and efficient treatment.

# ⚙️ **How to Run**

1.Run the main script:
python main.py

2.Select the GPT model you want to use.

3.Enter a patient ID number.
Example file: examples/123456789.txt → type 123456789

4.The program will process the file and create a summary inside the output folder:
output/summary_123456789.txt

# 🚀 **Future Improvements**

In the future, I would like to enhance the project by adding:

1.PDF support – the ability to read and summarize medical records directly from PDF files.

2.More advanced and medically tuned AI models – to improve accuracy and medical relevance.

3.Better readability for paramedics – optimizing the summary format for quick understanding in tough conditions (e.g., inside an ambulance).

4.A patient database – to securely store and update each patient’s medical history over time.

