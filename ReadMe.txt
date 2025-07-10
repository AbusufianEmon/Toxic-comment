
Toxic Comment Classification with TensorFlow & Gradio
=====================================================

This project implements a multi-label text classification model that identifies toxic comments.
It uses TensorFlow for model building and training, and Gradio to create an interactive web interface.

-----------------------------------------------------
Dataset
-----------------------------------------------------
- Path: /content/drive/MyDrive/Deep Learning/train.csv
- The dataset contains a 'comment_text' column and 6 label columns:
  - toxic
  - severe_toxic
  - obscene
  - threat
  - insult
  - identity_hate

Each comment may have one or more labels.



**** Dataset and Model Downlad link are given in dataset&model_download_link.txt. Must be download those files. Otherwise not work

-----------------------------------------------------
Dependencies
-----------------------------------------------------
Install the following Python libraries:

pip install tensorflow pandas numpy matplotlib gradio

-----------------------------------------------------
Project Steps
-----------------------------------------------------

1. Import Libraries:
   Load TensorFlow, Pandas, Numpy, etc.

2. Load and Preprocess Dataset:
   Read the CSV file and extract features (X) and labels (y).

3. Text Vectorization:
   Convert comments into sequences of integers using TextVectorization.

4. Create TensorFlow Dataset Pipeline (MCSHBAP):
   - Map
   - Cache
   - Shuffle
   - Batch
   - Prefetch

   Split dataset:
   - train: 70%
   - validation: 20%
   - test: 10%

5. Build the Model:
   - Embedding layer
   - Bidirectional LSTM
   - Dense layers with ReLU
   - Final output with sigmoid activation

6. Compile and Train:
   Compile with BinaryCrossentropy and Adam optimizer.
   Train the model for 5 epochs.

7. Evaluate:
   Use Precision, Recall, and BinaryAccuracy to evaluate on test set.

8. Save & Load the Model:
   Save with: model.save('my_model.keras')
   Load with: tf.keras.models.load_model('my_model.keras')

9. Make Predictions:
   Vectorize new comment and classify using threshold 0.2.

10. Gradio Web App:
    - Define score_comment() function
    - Create interface with gr.Interface()
    - Launch it to get a web UI

-----------------------------------------------------
Usage
-----------------------------------------------------
- Run the notebook or script in Colab or local environment.
- Use Gradio interface to input a comment and get predictions.

-----------------------------------------------------
Model Overview
-----------------------------------------------------
- Embedding Layer: Word vector representation
- Bidirectional LSTM: Capture context from both directions
- Dense Layers: Feature processing
- Output: 6 sigmoid outputs for multi-label classification

-----------------------------------------------------
Credits
-----------------------------------------------------
- Dataset: Jigsaw Toxic Comment Classification Challenge (Kaggle)
- Frameworks: TensorFlow, Gradio


