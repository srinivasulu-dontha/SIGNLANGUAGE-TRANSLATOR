# SIGNLANGUAGE-TRANSLATOR


A real-time sign language translator built using Python, machine learning, and computer vision. It allows users to interpret sign language gestures and convert them into readable text using a trained model.

---

## 📁 Project Structure

```

📦 sign-language-translator/
├── app.py                         # Main application file
├── model.ipynb                    # Jupyter notebook for training and analysis
├── gesture\_recognition\_model.pkl  # Trained ML model
├── run.bat                        # Windows batch file to run the app
├── sign launguage data set/
│   ├── amer\_sign2.png
│   ├── amer\_sign3.png
│   ├── american\_sign\_language.PNG
│   ├── sign\_mnist\_train.csv
│   ├── sign\_mnist\_test.csv
│   └── ...                        # More data files

````

---

## 💡 Features

- Real-time hand gesture recognition
- Converts signs into English text
- Pre-trained ML model using **Sign Language MNIST**
- GUI/CLI based execution (depending on `app.py` content)
- Image and CSV dataset processing

---

## 🛠 Technologies Used

- Python 3
- OpenCV
- NumPy & Pandas
- Scikit-learn
- Jupyter Notebook
- Git & GitHub

---

## 📦 Dataset Source

- [Sign Language MNIST Dataset by Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)

---

## 🚀 How to Run

### Prerequisites:
- Python 3.10+
- pip packages (see `requirements.txt` or use below)

### Step-by-step:
```bash
pip install -r requirements.txt
python app.py
````

Or run the Windows batch file:

```bash
run.bat
```

---

## 📸 Screenshots

*Add screenshots here later to demonstrate the app interface, training accuracy, etc.*

---

## 🙋‍♂️ Author

**Dontha Srinivasulu**
Email: [donthasrinivasulu83@gmail.com](mailto:donthasrinivasulu83@gmail.com)
GitHub: [@srinivasulu-dontha](https://github.com/srinivasulu-dontha)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

````

---

### ✅ How to Add This to GitHub

1. In your project folder:
   - Create a new file named `README.md`
2. Paste the content above into it
3. Save the file
4. Then run these commands:

```bash
git add README.md
git commit -m "Add project README"
git push

