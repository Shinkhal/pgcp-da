"""
Title: Student Academic Performance Analytics Lifecycle
Description: Fill in the TODO sections across the 5 lifecycle phases.
Dataset: student_performance.csv  (70 rows)
Columns: Student_ID, Stream, Study_Hours (per day), Attendance_Pct, Sleep_Hours, Exam_Score (out of 100)
"""

# Step 0: Import necessary libraries
# TODO: Import pandas, numpy, scikit-learn LinearRegression, and metrics


def phase_1_discovery_and_setup():
    """
    [Phase 1] Discovery & Data Setup (5 Marks)
    - Goal: Load 'student_performance.csv' and inspect structure.
    - Check for null values and handle them (fill numeric NaNs with the mean).
    """
    print("\n[PHASE 1] Discovery & Data Setup")
    print("-" * 65)
    # TODO: Load CSV file using pandas
    # TODO: Print first 5 rows, shape, and df.info()
    # TODO: Count missing values per column and impute numeric NaNs with the column mean
    pass


def phase_2_descriptive_statistics(df):
    """
    [Phase 2] Descriptive Statistics & Data Measures (10 Marks)
    - Goal: Compute central tendency and dispersion for 'Exam_Score'.
    - Detect outliers using the IQR method.
    """
    print("\n[PHASE 2] Descriptive Statistics")
    print("-" * 65)
    # TODO: Calculate Mean, Median, Std Dev, and IQR for Exam_Score
    # TODO: Compute lower/upper bounds (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
    # TODO: Print the students (Student_ID) whose Exam_Score falls outside the bounds
    # TODO: Print average Exam_Score per Stream (groupby)
    pass


def phase_3_bivariate_association(df):
    """
    [Phase 3] Bivariate Association & Correlation (10 Marks)
    - Goal: Examine relationships among Study_Hours, Attendance_Pct, Sleep_Hours and Exam_Score.
    """
    print("\n[PHASE 3] Bivariate Association")
    print("-" * 65)
    # TODO: Compute Pearson correlation matrix of the numeric columns
    # TODO: Highlight corr(Study_Hours, Exam_Score) and interpret its strength
    # TODO: Name the feature most strongly correlated with Exam_Score
    pass


def phase_4_predictive_modeling(df):
    """
    [Phase 4] Predictive Analytics & Regression Modeling (10 Marks)
    - Goal: Predict 'Exam_Score' from 'Study_Hours'.
    - Train a Simple Linear Regression model.
    """
    print("\n[PHASE 4] Predictive Modeling")
    print("-" * 65)
    # TODO: Set X = df[['Study_Hours']] and y = df['Exam_Score']
    # TODO: Fit LinearRegression; print slope and intercept
    # TODO: Predict Exam_Score when Study_Hours = 6 hours and print the result
    # TODO: return model, X, y
    pass


def phase_5_evaluation_and_reporting(model, X, y, df):
    """
    [Phase 5] Evaluation & Prescriptive Recommendation (5 Marks)
    - Goal: Evaluate the model using R-squared and MAE.
    - Provide operational recommendations for the academic counselling team.
    """
    print("\n[PHASE 5] Evaluation & Recommendations")
    print("-" * 65)
    # TODO: Calculate R2 score and Mean Absolute Error (MAE)
    # TODO: Print a prescriptive action item: estimate the marks gained per extra daily study hour and recommend whether a mandatory study-hall program is justified
    pass


def run_analytics_lifecycle():
    """Facade Orchestrator function."""
    print("=" * 65)
    print("STARTING STUDENT ANALYTICS LIFECYCLE (Out of 40 Marks)")
    print("=" * 65)

    df = phase_1_discovery_and_setup()
    # phase_2_descriptive_statistics(df)
    # phase_3_bivariate_association(df)
    # model, X, y = phase_4_predictive_modeling(df)
    # phase_5_evaluation_and_reporting(model, X, y, df)


if __name__ == "__main__":
    run_analytics_lifecycle()
