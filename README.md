## Predictive Modeling for Boost Converter Component Sizing

Welcome to the **Boost Converter Component Sizing** project!  
This project builds a predictive model that recommends optimal passive component values (Resistance, Inductance, and Capacitance) for DC-DC Boost Converters based on input and output voltages.

---

##  Project Overview

This notebook uses **Multi-Output Regression** techniques to predict:

- Resistance (ohms)
- Inductance (henries)
- Capacitance (farads)

based on the input features:

- Input Voltage (Vin)
- Output Voltage (Vout)

The pipeline includes:

- Data cleaning and preprocessing
- Exploratory data analysis (EDA) and visualization
- Feature selection and pattern recognition
- Training a supervised regression model
- Model evaluation and inference
- Component recommendation based on user-provided voltages

---

##  Data Source

- **Dataset**: Boost Converter experimental/simulated data
- **Format**: CSV file (`boost_converter_data.csv`)

---

##  How It Works

1. **Data Cleaning**: 
   - Verified and handled missing or inconsistent values.
   - Ensured correct data types.

2. **Exploratory Data Analysis**: 
   - Visualized input voltage, output voltage, and component relationships.
   - Generated a correlation heatmap.

3. **Feature Selection**: 
   - Selected `Vin` and `Vout` as input features.
   - Targeted `Resistance1`, `Inductance`, and `Capacitance` as outputs.

4. **Model Training**: 
   - Trained a **Multi-Output Regression model** using a Random Forest Regressor.

5. **Model Evaluation**: 
   - Achieved near-perfect results:
     - Mean Squared Error (MSE): ~1e-7
     - R² Score: ~0.99999999

6. **Inference**: 
   - Users can input any Vin and Vout values.
   - The trained model predicts the recommended component values instantly.

---

##  Selected Features

| Feature | Description |
|:--------|:------------|
| Vin | Input voltage to the converter |
| Vout | Desired output voltage from the converter |

---

##  Model Performance

- **Mean Squared Error (MSE)**: `~1.04e-7`
- **R-squared (R²) Score**: `~0.99999999`


---

## Requirements

Install the following libraries:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

(You can install them all at once using: `pip install -r requirements.txt`)

---

##  Usage Example

```python
# Recommend passive components for given Vin and Vout
recommend_components(18, 14)

# Output:
# Recommended Resistance1: 17.998305 ohms
# Recommended Inductance: 0.003799 H
# Recommended Capacitance: 0.000170 F
```

---

##  License

This project is licensed under the MIT License.
