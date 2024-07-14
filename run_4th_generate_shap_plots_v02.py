import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the predictions with SHAP values
data = pd.read_csv('predictions_with_shap.csv')

# Set the number of top features to display
top_features = 10

# Create a directory to save the plots
import os
if not os.path.exists('plots'):
    os.makedirs('plots')

# Iterate over each prediction
for i in range(len(data)):
    row = data.iloc[i]
    shap_values = row.filter(like='SHAP_')
    shap_values = shap_values.abs().sort_values(ascending=False).head(top_features)
    
    # Prepare data for plotting
    plot_data = pd.DataFrame({
        'Feature': shap_values.index.str.replace('SHAP_', ''),
        'SHAP Value': shap_values.values
    })
    
    # Create a bar plot for the SHAP values
    plt.figure(figsize=(20, 6))
    sns.barplot(x='SHAP Value', y='Feature', data=plot_data, palette='viridis', hue='Feature', dodge=False, legend=False)
    
    plt.title(f'Feature Impact on Churn Prediction for Customer {i + 1}')
    plt.xlabel('SHAP Value (Impact on Model Output)')
    plt.ylabel('Feature')
    
    # Save the plot
    plt.savefig(f'plots/churn_prediction_{i + 1}.png')
    plt.close()

print("Plots generated and saved in the 'plots' directory.")