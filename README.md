# Human Activity Recognition with Smartphones (UCI HAR v2)

This project uses neural networks to perform **human activity classification** based on data collected from smartphone sensors (accelerometer and gyroscope). The dataset used is version 2 of the well-known **UCI HAR Dataset**, available on Kaggle.

---

## 📦 Dataset

The dataset can be found here:  
🔗 [Human Activity Recognition with Smartphones (UCI HAR v2) – Kaggle](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones)

### Dataset details:

- Each instance represents a 2.56-second window (128 readings)
- **561 features** extracted from sensor signals (time and frequency domains)
- Data collected from 30 volunteers performing activities with a smartphone worn at the waist
- Target activity classes:
  - `WALKING`
  - `WALKING_UPSTAIRS`
  - `WALKING_DOWNSTAIRS`
  - `SITTING`
  - `STANDING`
  - `LAYING`

---

## 🎯 Project Goal

The goal is to develop and train neural network models that can accurately predict human activity based on sensor data. The focus is on applying **neural networks with PyTorch** to a supervised multi-class classification problem, exploring different model architectures and learning their trade-offs.

---

## 🧠 Models to Be Implemented

This project explores multiple neural network architectures:

### 1. **MLP (Multilayer Perceptron)**
- A fully connected neural network with multiple hidden layers
- Ideal for high-dimensional tabular datasets like UCI HAR v2
- Will serve as a baseline

### 2. **1D CNN (Convolutional Neural Network)**
- 1D convolutional layers applied to the tabular data, treated as a sequence
- Useful for capturing spatial patterns and local interactions between features

### 3. *(Optional)* **LSTM (Long Short-Term Memory)**
- Recurrent network suitable for sequential data
- May be used if raw signal sequences are reintroduced

---

## 🛠 Tech Stack

- [Python](https://www.python.org/)
- [PyTorch](https://pytorch.org/)
- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [Jupyter Notebooks](https://jupyter.org/)
- [VS Code](https://code.visualstudio.com/) + Jupyter extension

---

## 📁 Project Structure

```

human-activity-recognition/
├── data/             # Raw or preprocessed datasets
├── notebooks/        # Jupyter Notebooks for EDA, training, and evaluation
├── src/              # Modular Python scripts (data loaders, models, training, etc.)
├── outputs/          # Saved models, plots, confusion matrices
├── README.md         # Project overview and documentation
├── requirements.txt  # Project dependencies
└── .gitignore        # Files/folders to exclude from version control

```

---

## 📈 Expected Results

- Accuracy > 90% using neural network classifiers
- Evaluation using confusion matrix and learning curves
- Comparison between MLP and CNN performance on this dataset

---

## 📌 Project Status

☑️ Initial setup  
☑️ Data exploration  
☑️ MLP implementation  
☑️ Evaluation and hyperparameter tuning  
☐ CNN experimentation  
☐ Final documentation  

---

## 👨‍💻 Author

Murilo Zangari  
[Portfolio](https://murilozangari.com) • [LinkedIn](https://www.linkedin.com/in/murilozangari)


