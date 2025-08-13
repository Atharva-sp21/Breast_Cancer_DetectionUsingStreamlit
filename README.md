# Logistic Regression Web App

A simple and interactive **Machine Learning web application** built with **Streamlit** that uses a **Logistic Regression** model to make predictions.  
The application also uses **Plotly** for interactive visualizations and **Pickle** for model storage and loading.

---

## 🚀 Features

- **Logistic Regression Model** for binary/multiclass classification
- **Streamlit Web Interface** for easy user interaction
- **Plotly Graphs** for dynamic, interactive data visualization
- **Pickle** for saving and loading the trained model
- Real-time prediction based on user input
- Responsive and clean UI

---

## 📂 Project Structure
app.py # Streamlit app code
- model.pkl # Saved Logistic Regression model (Pickle)
- requirements.txt # Python dependencies
- data.csv # Dataset (optional, if included)
- README.md # Project documentation
The application provides:
- Input forms for user data
- A **predict** button that uses the logistic regression model to output the prediction
- **Plotly** visualizations for:
  - Data distribution
  - Decision boundaries (if applicable)
  - Prediction probability plots

---

## 🛠️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/logistic-regression-streamlit.git
   cd logistic-regression-streamlit
