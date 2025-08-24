# ✍️ Signature Verification System

A **Multilayer Perceptron (MLP)-based** signature verification system trained on **800 authentic and forged samples** from 40 individuals.

- 🧮 **Features:** Extracts 9 statistical and geometric features (eccentricity, skewness, centroid, etc.) for classification.
- 🖼️ **Preprocessing:** Grayscale conversion, Otsu’s thresholding, and Gaussian filtering for robust feature enhancement.
- 🧠 **Model:** MLP with 3 layers (7, 10, and 30 neurons), optimized using Adam and Softmax classifier.
- 🏆 **Performance:** Achieves up to **97% training accuracy**.

---

## 📁 Project Structure

```
📄 Code_sign.py           # Main script: feature extraction, training, evaluation
📓 Main_Code.ipynb        # Jupyter notebook for experimentation
📂 TestFeatures/
    └── testcsv.csv       # Features of the test signature
📂 Testing/
    └── testing_001.csv   # Test signature data (CSV)
    ...
📂 Training/
    ...                   # Training signature data (CSV)
```

---

## ⚙️ How It Works

1. **Preprocessing** 🧹  
   Converts signature images to grayscale, applies Gaussian filtering, and binarizes using Otsu’s threshold. Crops the signature region.

2. **Feature Extraction** 🏷️  
   Extracts 9 features: ratio, centroid (x, y), eccentricity, solidity, skewness (x, y), kurtosis (x, y).

3. **Dataset Preparation** 📊  
   Generates CSVs for training and testing from image folders.

4. **Model Training** 🤖  
   - MLP with 3 hidden layers: 7, 10, and 30 neurons.
   - Adam optimizer and Softmax for classification.
   - Trains on features from the `Training/` folder.

5. **Testing** 🔍  
   - Extracts features from a test signature.
   - Predicts using the trained model.
   - Outputs **Genuine** or **Forged**.

---

## 🚀 Usage

1. **Install dependencies:**
   ```sh
   pip install numpy pandas matplotlib scikit-image tensorflow keras
   ```

2. **Prepare your data:**
   - Place training signature CSVs in `Training/`
   - Place test signature CSVs in `Testing/`

3. **Run the main script:**
   ```sh
   python Code_sign.py
   ```
   - Enter the person’s ID and the path to the test signature image when prompted.

4. **Results:**
   - The script prints training and test accuracy, and classifies the test signature.

---

## 🛠️ Main Functions

- `rgbgrey`: RGB to grayscale conversion
- `greybin`: Grayscale to binary using Otsu’s threshold
- `preproc`: Preprocessing and cropping
- `getCSVFeatures`: Extracts 9 features from an image
- `makeCSV`: Generates training/testing CSVs
- `evaluate`: Trains and evaluates the MLP
- `trainAndTest`: Runs cross-validation for multiple users

---

## 💡 Example

```python
# Extract features and classify a signature
makeCSV()
train_person_id = input("Enter person's id : ")
test_image_path = input("Enter path of signature image : ")
testing(test_image_path)
evaluate(train_path, test_path, type2=True)
```


