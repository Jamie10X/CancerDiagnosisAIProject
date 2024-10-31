# CancerDiagnosisAIProject
Cancer Diagnosis Project Using Feedforward Neural Network (FNN)

1. Project Objective
This project aimed to build a Feedforward Neural Network (FNN) model to classify cancer types based on gene expression data from four different cancer types:
	•	BRCA (Breast Cancer)
	•	COAD (Colon Adenocarcinoma)
	•	LUAD (Lung Adenocarcinoma)
	•	THCA (Thyroid Cancer)
The FNN model was trained on gene activity levels across thousands of genes, with the goal of achieving high classification accuracy for each cancer type.

2. Data Preparation
Data Loading and Combination
	•	Objective: Load individual datasets for each cancer type and combine them into a single dataset with labeled classes.
	•	Process:
	•	Loaded datasets TCGA_BRCA.csv, TCGA_COAD.csv, TCGA_LUAD.csv, and TCGA_THCA.csv.
	•	Added a CancerType column to each dataset to denote the cancer type.
	•	Combined all datasets into a single DataFrame to standardize the data processing pipeline.
	•	Result: Successfully created a single dataset containing all samples with cancer type labels, ready for preprocessing.


Handling Missing Values
	•	Issue: The dataset contained approximately 146 million NaN values.
	•	Solution: Replaced NaN values with the mean of each feature column.
	•	Outcome: Successfully removed all NaN values, ensuring numerical stability during model training.


Label Encoding and One-Hot Encoding
	•	Objective: Prepare target labels in a format suitable for multi-class classification.
	•	Process:
	•	Used Label Encoding to convert categorical labels into integer values.
	•	Applied One-Hot Encoding to convert integer labels into a binary matrix format for each class.
	•	Outcome: Labels were successfully one-hot encoded, creating four binary columns, one for each cancer type.

Feature Scaling
	•	Objective: Normalize the data to improve the model’s convergence during training.
	•	Process: Applied MinMaxScaler to scale the gene expression values between 0 and 1.
	•	Outcome: Normalized features, ensuring consistent data ranges that improve neural network performance.

3. Model Architecture and Configuration

Model Definition
	•	Architecture:
	•	Input Layer: Defined to accept the large number of gene features.
	•	Hidden Layers:
	•	First hidden layer with 1024 neurons and ReLU activation.
	•	Dropout layer with 0.3 dropout rate to prevent overfitting.
	•	Second hidden layer with 512 neurons and ReLU activation.
	•	Dropout layer with 0.3 dropout rate for additional regularization.
	•	Output Layer: Softmax layer with 4 neurons to represent each cancer type.
	•	Result: Created a multi-layer FNN capable of capturing complex patterns in high-dimensional gene expression data.

Model Compilation
	•	Objective: Configure the model for training.
	•	Settings:
	•	Optimizer: Adam with a low learning rate of 1e-5 to ensure stable learning.
	•	Loss Function: Categorical Crossentropy, suited for multi-class classification.
	•	Metrics: Accuracy to monitor classification performance.
	•	Outcome: Successfully compiled the model, ready for training.

4. Model Training and Early Stopping
	•	Objective: Train the model on the dataset and validate its performance.
	•	Settings:
	•	Epochs: 20 (Early Stopping applied to prevent overfitting).
	•	Batch Size: 32.
	•	Validation Split: 20% of training data.
	•	Early Stopping: Monitored val_loss with patience of 3 epochs to halt training when performance plateaued.
	•	Training Results:
	•	The model showed continuous improvement in both training and validation accuracy with decreasing loss.
	•	Final training accuracy: 100%, validation accuracy: 99.96%.
	•	Final validation loss was minimal, indicating effective learning and generalization.

5. Model Evaluation

Test Set Performance
	•	Objective: Evaluate the model on an unseen test set to determine its generalization ability.
	•	Results:
	•	Test Accuracy: Achieved an impressive 99.96% accuracy on the test set.
	•	Loss: Maintained a low test loss, indicating the model’s robustness.

Confusion Matrix and Classification Report

 •	Confusion Matrix:

[[3927    1    2    1]
 [   1 4046    1    0]
 [   0    0 4019    1]
 [   0    0    0 4001]]

 •	Almost perfect classification for all cancer types, with minimal misclassifications.
	•	Classification Report:
	•	Precision, Recall, F1-score for each class was 1.00, confirming that the model achieved near-perfect performance across all metrics.
	•	Overall Accuracy: 100%, with weighted averages for precision, recall, and F1-score also reaching 1.00, demonstrating the model’s effectiveness.

6. Training and Validation Plots
	•	Objective: Visualize training dynamics to ensure stable learning and identify any signs of overfitting.
	•	Accuracy Plot:
	•	Training and validation accuracy both increased steadily, with minimal overfitting observed.
	•	Loss Plot:
	•	Both training and validation loss decreased over epochs, with validation loss stabilizing near zero.
	•	Outcome: Visualizations confirmed that the model was well-tuned and did not suffer from overfitting, thanks to effective regularization and early stopping.

7. Conclusion

The FNN model demonstrated exceptional performance on the cancer classification task, achieving nearly perfect accuracy on both training and test data. Key contributing factors to this success included careful handling of NaN values, effective feature scaling, dropout regularization, and low learning rate for stable training.

Summary of Key Metrics:
	•	Final Test Accuracy: 99.96%
	•	Precision, Recall, F1-score: 1.00 across all classes.

This project successfully showcases the capability of FNNs to classify complex biological data, providing a promising model for cancer diagnosis based on gene expression profiles.
