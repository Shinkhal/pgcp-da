"""
Title: Student Academic Performance Analytics Lifecycle
Description: Complete the TODO sections across the 5 lifecycle phases.
Dataset: student_performance.csv (70 rows)
"""

# Step 0: Import necessary libraries
# TODO: Import pandas, numpy, LogisticRegression,
#       train_test_split, and StandardScaler.
# TODO: Import accuracy_score, precision_score, recall_score,
#       f1_score, and confusion_matrix.


def phase_1_discovery_and_setup():
    """
    [Phase 1] Discovery & Data Setup (5 Marks)
    - Load and inspect the dataset.
    - Handle missing numerical values.
    """
    print("\n[PHASE 1] Discovery & Data Setup")
    print("-" * 65)

    # TODO: Load CSV
    # TODO: Inspect dataset
    # TODO: Handle missing values

    return df


def phase_2_descriptive_statistics(df):
    """
    [Phase 2] Descriptive Statistics (10 Marks)
    - Analyze Exam_Score.
    - Calculate statistical measures.
    - Detect outliers using IQR.
    - Compare performance across streams.
    """
    print("\n[PHASE 2] Descriptive Statistics")
    print("-" * 65)

    # TODO: Calculate statistical measures
    # TODO: Apply IQR method
    # TODO: Identify outliers
    # TODO: Analyze streams

    return df


def phase_3_bivariate_association(df):
    """
    [Phase 3] Bivariate Association (10 Marks)
    - Examine relationships among student attributes.
    - Analyze correlation with Exam_Score.
    """
    print("\n[PHASE 3] Bivariate Association")
    print("-" * 65)

    # TODO: Compute correlation matrix
    # TODO: Analyze selected correlations
    # TODO: Identify strongest feature

    return df


def phase_4_predictive_modeling(df):
    """
    [Phase 4] Predictive Modeling (10 Marks)
    - Create a Pass/Fail target from Exam_Score.
    - Predict student performance using Logistic Regression.
    """
    print("\n[PHASE 4] Predictive Modeling")
    print("-" * 65)

    # TODO: Create binary target (Pass/Fail)
    # TODO: Prepare features
    # TODO: Encode categorical data
    # TODO: Split dataset (80:20)
    # TODO: Scale features
    # TODO: Train Logistic Regression
    # TODO: Generate predictions

    return model, X_test, y_test, y_pred


def phase_5_evaluation_and_reporting(model, X_test, y_test, y_pred, df):
    """
    [Phase 5] Evaluation & Reporting (5 Marks)
    - Evaluate classification performance.
    - Recommend academic interventions.
    """
    print("\n[PHASE 5] Evaluation & Recommendations")
    print("-" * 65)

    # TODO: Calculate classification metrics
    # TODO: Display confusion matrix
    # TODO: Interpret model performance
    # TODO: Generate recommendations


def run_analytics_lifecycle():
    """Facade Orchestrator function."""

    print("=" * 65)
    print("STUDENT PERFORMANCE CLASSIFICATION (Out of 40 Marks)")
    print("=" * 65)

    df = phase_1_discovery_and_setup()
    df = phase_2_descriptive_statistics(df)
    df = phase_3_bivariate_association(df)
    model, X_test, y_test, y_pred = phase_4_predictive_modeling(df)
    phase_5_evaluation_and_reporting(
        model, X_test, y_test, y_pred, df
    )


if __name__ == "__main__":
    run_analytics_lifecycle()