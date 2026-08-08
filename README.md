# Financial Sentiment & Stock Prediction Dashboard

This repository contains a professional Streamlit dashboard that serves as the front-end interface for a machine learning pipeline focused on financial sentiment analysis and stock price prediction.

## Features

1. **💬 Sentiment Analysis**
   - Uses a fine-tuned **RoBERTa-Large** model (355M parameters).
   - Classifies financial text into three categories: **Positive, Neutral, and Negative**.
   - Achieves a high validation accuracy of **98.43%**.
   - Includes single-text analysis with quick examples and batch analysis capabilities.

2. **📈 Stock Price Prediction**
   - Uses a **Temporal Long Short-Term Memory (T-LSTM)** network.
   - Specifically trained to predict **Reliance Industries** close prices.
   - Utilizes 15-day rolling windows with 9 features (8 technical stock indicators + 1 sentiment score).
   - Achieves an Root Mean Square Error (RMSE) of **₹43.26**.
   - Visualizes actual vs. predicted prices alongside comprehensive error metrics (MAE, MAPE, Error Distribution).

## Tech Stack
- **Frontend Framework:** Streamlit
- **Machine Learning (NLP):** HuggingFace Transformers (PyTorch)
- **Machine Learning (Time Series):** Scikit-learn (Joblib for persistence), PyTorch / Keras (depending on TLSTM backend)
- **Data Visualization:** Plotly Graph Objects, Pandas
- **Data Processing:** NumPy, Pandas

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Kumar18-GI/ml_frontend.git
   cd ml_frontend
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Important Note on Models**: 
   The application requires the trained model checkpoints to run:
   - RoBERTa weights must be located at `../roberta_final_checkpoint-20260808T120900Z-1-001/roberta_final_checkpoint`
   - TLSTM weights and scalers must be located at `../tlstm/`
   
   *(Note: Ensure paths match your local environment setup in `pipeline/model_loader.py`)*

## Running the Dashboard

Launch the Streamlit app locally:
```bash
streamlit run app.py
```

The app will be accessible at `http://localhost:8501`.

## Architecture Note
This application was refactored into a **single-page native Streamlit app** to ensure maximum stability, preventing UI re-rendering glitches and multi-page routing conflicts. Heavy machine learning models (like RoBERTa) are lazily loaded and cached in memory using `@st.cache_resource` for optimal performance.
