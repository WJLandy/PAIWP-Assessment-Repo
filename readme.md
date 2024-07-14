Repo for PAIWP assessment

Required libraries detailed in the "requirements.txt"
    
    To automatically install use the command "pip install -r requirements.txt"


Included .csv file details

    Customer_Churn.csv

    churn data used for model training


    new_data.csv

    30 records separated from the initial training set, this is intended to mimic live data inputs that predictions are required for


Included .py file details

    run_1st_data_cleaning.py

    First file to run to clean the input data contained in "Customer_Churn.csv" and output as a new file "cleaned_data.csv"


    run_2nd_modeling.py

    Second file to run that takes the data inside "cleaned_data.csv" and builds the "model.pkl" and "scaler.pkl" files


    run_3rd_predict_V02.py

    Third file to run that takes the "new_data.csv" file cleans the data and makes predictions using the built model, it then integrates shap in order to explain what features on the input data had the biggest impact on the prediction it then outputs the results as 'predictions_with_shap.csv'


    run_4th_generate_shap_plots_v02.py

    Fourth file to run that takes the 'predictions_with_shap.csv' file and creates bar charts for all the generated predictions so that the results can be easily interperated by a non-technical audience - in a deployment scenario this would be visible in the companies CRM system to assist sales, support and retention teams to understand both the likleyhood of churn and the contributing factors
    