# Brain Tumor Classifier

This repository contains a Brain Tumor Classifier built using a Convolutional Neural Network (CNN) architecture. The classifier utilizes brain MRI scans to categorize tumors into the following four categories:

1. Glioma
2. Meningioma
3. No Tumor
4. Pituitary

## Project Overview
The classifier is based on transfer learning, using the **Xception** model as a pretrained backbone. The Adamax optimizer and categorical cross-entropy loss function are used to achieve high classification accuracy.

### Key Features:
- Multi-class classification of brain tumors.
- Transfer learning with Xception for robust feature extraction.
- Interactive and reproducible implementation using Jupyter Notebook.

## Dataset
The dataset used for this project is sourced from Kaggle and contains labeled MRI scans for the four tumor categories.

- **Dataset Source:** [Brain Tumor MRI (Kaggle)](https://www.kaggle.com/code/yousefmohamed20/brain-tumor-mri-accuracy-99/input)
- **Alternative Dataset Link (Google Drive):** [(https://drive.google.com/drive/folders/1ghc0L100ZcnXlWkR2z3q-alAiD8sXlRH?usp=drive_link)](#)

## Repository Contents
- `brain-tumor-mri-using-cnn.ipynb`: Jupyter Notebook containing the entire pipeline, including:
  - Dataset loading and preprocessing
  - Model training and evaluation
  - Visualization of results

## Prerequisites
### General Requirements:
- Python 3.7 or later
- Jupyter Notebook

### Python Libraries:
- `tensorflow`
- `numpy`
- `matplotlib`
- `opencv-python`
- `scikit-learn`

You can install the required libraries using pip:
```bash
pip install tensorflow numpy matplotlib opencv-python scikit-learn
```

## Getting Started
1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/your-username/brain-tumor-classifier.git
   cd brain-tumor-classifier
   ```

2. Download the dataset from Kaggle or the Google Drive link provided above.

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Open and run the `brain-tumor-mri-using-cnn.ipynb` notebook.

5. Follow the instructions within the notebook to preprocess the data, train the model, and evaluate its performance.

## Usage Instructions
- Ensure the dataset is correctly placed in the appropriate directory as specified in the notebook.
- Run the notebook cells sequentially to execute the pipeline.
- Visualize the classification results and metrics.

## Future Improvements
- Explore other state-of-the-art pretrained models for enhanced accuracy.
- Implement real-time tumor classification using a web or mobile application.
- Incorporate more diverse datasets to improve model generalizability.

## Contributing
Contributions are welcome! Feel free to fork the repository and submit a pull request.


Enjoy experimenting with the brain tumor classifier! 🚀

