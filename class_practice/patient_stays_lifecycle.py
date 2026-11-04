"""
Title: Hospital Patient Stay Analytics Lifecycle
Description: Fill in the TODO sections across the 5 lifecycle phases.
Dataset: patient_stays.csv  (80 rows)
Columns: Patient_ID, Department, Age, BMI, Num_Procedures, Length_of_Stay_Days
"""

# Step 0: Import necessary libraries
# TODO: Import pandas, numpy, scikit-learn LinearRegression, and metrics


def phase_1_discovery_and_setup():
    """
    [Phase 1] Discovery & Data Setup (5 Marks)
    - Goal: Load 'patient_stays.csv' and inspect structure.
    - Check for null values and handle them (fill numeric NaNs with the median).
    """
    print("\n[PHASE 1] Discovery & Data Setup")
    print("-" * 65)
    # TODO: Load CSV file using pandas
    # TODO: Print first 5 rows, shape, and df.info()
    # TODO: Count missing values per column and impute numeric NaNs with the median
    pass


def phase_2_descriptive_statistics(df):
    """
    [Phase 2] Descriptive Statistics & Data Measures (10 Marks)
    - Goal: Compute central tendency and dispersion for 'Length_of_Stay_Days'.
    - Detect outliers using the IQR method.
    """
    print("\n[PHASE 2] Descriptive Statistics")
    print("-" * 65)
    # TODO: Calculate Mean, Median, Std Dev, and IQR for Length_of_Stay_Days
    # TODO: Compute lower/upper bounds (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
    # TODO: Print the patients (Patient_ID) whose Length_of_Stay_Days falls outside the bounds
    # TODO: Print average Length_of_Stay_Days per Department (groupby)
    pass


def phase_3_bivariate_association(df):
    """
    [Phase 3] Bivariate Association & Correlation (10 Marks)
    - Goal: Examine relationships among Age, BMI, Num_Procedures and Length_of_Stay_Days.
    """
    print("\n[PHASE 3] Bivariate Association")
    print("-" * 65)
    # TODO: Compute Pearson correlation matrix of the numeric columns
    # TODO: Highlight corr(Num_Procedures, Length_of_Stay_Days) and interpret its strength
    # TODO: Name the feature most strongly correlated with Length_of_Stay_Days
    pass


def phase_4_predictive_modeling(df):
    """
    [Phase 4] Predictive Analytics & Regression Modeling (10 Marks)
    - Goal: Predict 'Length_of_Stay_Days' from 'Num_Procedures'.
    - Train a Simple Linear Regression model.
    """
    print("\n[PHASE 4] Predictive Modeling")
    print("-" * 65)
    # TODO: Set X = df[['Num_Procedures']] and y = df['Length_of_Stay_Days']
    # TODO: Fit LinearRegression; print slope and intercept
    # TODO: Predict Length_of_Stay_Days when Num_Procedures = 4 procedures and print the result
    # TODO: return model, X, y
    pass


def phase_5_evaluation_and_reporting(model, X, y, df):
    """
    [Phase 5] Evaluation & Prescriptive Recommendation (5 Marks)
    - Goal: Evaluate the model using R-squared and MAE.
    - Provide operational recommendations for hospital bed-management.
    """
    print("\n[PHASE 5] Evaluation & Recommendations")
    print("-" * 65)
    # TODO: Calculate R2 score and Mean Absolute Error (MAE)
    # TODO: Print a prescriptive action item: estimate extra bed-days per additional procedure and recommend how many beds to reserve for high-procedure departments
    pass


def run_analytics_lifecycle():
    """Facade Orchestrator function."""
    print("=" * 65)
    print("STARTING HOSPITAL ANALYTICS LIFECYCLE (Out of 40 Marks)")
    print("=" * 65)

    df = phase_1_discovery_and_setup()
    # phase_2_descriptive_statistics(df)
    # phase_3_bivariate_association(df)
    # model, X, y = phase_4_predictive_modeling(df)
    # phase_5_evaluation_and_reporting(model, X, y, df)


if __name__ == "__main__":
    run_analytics_lifecycle()
