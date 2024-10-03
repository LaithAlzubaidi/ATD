# ATD
AI-To-Data: A Secure Decentralized Learning Approach for Multi-Scenario Violence Detection Using Blockchain
The code is didvded into parts 

A- How to run ATD

This guideline walks you through how to run the provided code across several models (CNN + SE + BiLSTM, External Fusion, Internal Fusion, Grad-CAM, and t-SNE). It includes steps for environment setup, dataset preparation, model execution, and evaluation.


1. Environment Setup:
Ensure you have the necessary libraries installed. You can use the following command to install the required packages:

'pip install tensorflow keras scikit-learn matplotlib numpy seaborn xgboost'

2. Data Preparation:
Prepare your datasets for training and testing:

The code expects image datasets organized in directories (train/, test/) under folders for each class.
You will need to have three datasets:
RWF Dataset
RLV Dataset
Hockey Dataset


3. Running the CNN Models with SE + BiLSTM (from file CNN models+SE+BiLSTM.txt):
This script trains a CNN model based on NASNetMobile with an added SE (Squeeze-and-Excitation) block, followed by a BiLSTM layer.
Modify the paths to your dataset (train_set and valid_set) to point to your local directories.
To run this script, execute:

'python CNN_models+SE+BiLSTM.py'


After the model is trained, it saves the weights as Nasnet.h5 and evaluates the performance. Visualization for loss, accuracy, and confusion matrix is provided.

4. Running Internal Fusion (from file Internal Fusion.txt):
This script performs internal fusion of features extracted from different models, like NASNet, Xception, and MobileNet, combining them at a feature level using the add function.
It applies majority voting with several classifiers like Logistic Regression, SVC, KNN, XGBoost, and GaussianNB.

'python Internal_Fusion.py'


6. Running the External Fusion (from file External Fusion.txt):
This script combines the outputs from multiple pre-trained models (NASNet, Xception, and MobileNet) using a feature fusion technique.
Models are loaded using their pre-trained weights, and the predictions from different models are aggregated.
Majority voting classifiers are implemented using Logistic Regression, SVM, KNN, and XGBoost.
To run the External Fusion, execute:

'python External_Fusion.py'

6. Applying Grad-CAM (from file Grad-CAM and t-SNE.txt):
Grad-CAM visualizes which parts of the image influence the model's decisions.

The script processes images and creates heatmaps using the Grad-CAM technique.
It requires you to provide an image path for testing (img_path = 'Grad-CAM/original/10.jpg'), which you can change to any test image.
The script also implements t-SNE to visualize feature embeddings for both training and validation datasets.

'python Grad-CAM_and_t-SNE.py'


7. Model Evaluation:
The performance of the models is evaluated using several metrics, such as:

Accuracy
Precision
Recall
F1 Score
Cohen's Kappa
AUC (Area Under the ROC Curve)
The scripts also generate confusion matrices and visual plots (Grad-CAM heatmaps, t-SNE embeddings, etc.).

8. Troubleshooting:
Ensure you have the right file paths to the datasets and the pre-trained model weights (.h5 files).
Modify the paths in the code to match your local file structure.
If using custom layers like SEBlock, remember to load models with custom_objects={'SEBlock': SEBlock} as shown in the code.
By following these steps, you should be able to execute the provided models and generate the visual outputs.



B-Eye on AI
Run Each Script:

Step 1: Run secret_sharing.py to generate the shares.
bash
Copy code
python secret_sharing.py
Step 2: Use the shares generated from the secret sharing script to initialize and run the blockchain.
bash
Copy code
python blockchain.py
Step 3: Validate the ledger using the validate_ledger.py script.
bash
Copy code
python validate_ledger.py
Step 4: Run the proof of share consensus algorithm by using proof_of_share_blockchain.py.
bash
Copy code
python proof_of_share_blockchain.py
Step 5: Run the script to automatically generate blocks from an Excel file (node_block_from_excel.py).
bash
Copy code
python node_block_from_excel.py


Trained Models Due to the limited size of the repository, the trained models are not included. However, they can be requested from l.alzubaidi@qut.edu.au.

Citation If you find this work useful and use it in your research, please consider citing our paper:

Contact For any questions, issues, or requests regarding the code or trained models, please contact l.alzubaidi@qut.edu.au. We welcome feedback and collaboration opportunities.
