import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt=" %Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class DataCleaning:

    def __init__(self, df: pd.DataFrame):
        # Store a copy to prevent mutating the original dataframe unexpectedly
        self.df = df.copy()

    def snake_case(self) -> "DataCleaning":
        """Convert column names to snake_case."""
        self.df.columns = (
            self.df.columns.str.strip().str.lower().str.replace(" ", "_")
        )
        logger.info("Converted columns to snake_case")
        return self

    def sanitize_text(self) -> "DataCleaning":
        """Strip whitespace and lowercase all string/object columns."""
        string_cols = self.df.select_dtypes(include=["object", "string"]).columns
        self.df[string_cols] = self.df[string_cols].apply(
            lambda x: x.str.strip().str.lower()
        )
        logger.info("Sanitized text columns")
        return self

    def clean_salary(self) -> "DataCleaning":
        """Remove commas and convert salary to numeric."""
        if "salary" in self.df.columns:
            self.df["salary"] = (
                self.df["salary"]
                .astype(str)
                .str.replace(",", "", regex=False)
            )
            self.df["salary"] = pd.to_numeric(
                self.df["salary"], errors="coerce"
            )
            logger.info("Cleaned and converted salary column to numeric")
        return self

    def standardize_dates(self) -> "DataCleaning":
        """Parse joining dates into a standard datetime format."""
        if "joining_date" in self.df.columns:
            self.df["joining_date"] = pd.to_datetime(
                self.df["joining_date"], errors="coerce"
            )
            logger.info("Standardized joining_date column")
        return self

    def remove_duplicates(self) -> "DataCleaning":
        """Drop duplicate rows."""
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        dropped_count = initial_count - len(self.df)
        logger.info(f"Removed {dropped_count} duplicate rows")
        return self

    def fill_missing_values(self) -> "DataCleaning":
        """Fill missing values using suitable strategies."""

        # Numerical columns
        if "salary" in self.df.columns:
            self.df["salary"] = self.df["salary"].fillna(
                self.df["salary"].median()
            )

        if "performance_score" in self.df.columns:
            self.df["performance_score"] = (
                self.df["performance_score"].fillna(
                    self.df["performance_score"].mean()
                )
            )

        # Categorical columns
        for col in ["first_name", "last_name"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna("unknown")

        if "department" in self.df.columns:
            mode_values = self.df["department"].mode()
            if not mode_values.empty:
                self.df["department"] = self.df["department"].fillna(
                    mode_values.iloc[0]
                )

        logger.info("Filled missing values")
        return self

    def get_result(self) -> pd.DataFrame:
        """Return the cleaned dataframe."""
        return self.df


if __name__ == "__main__":
    data = {
        "Employee ID": [101, 102, 103, 104, 105, 106, 102, 108],
        "First Name": ["John", "Alice", "BOB", "   charlie", None, "David", "Alice", "Eva"],
        "Last Name": ["Smith", "Johnson", "Brown", "Davis", "Wilson", None, "Johnson", "Taylor"],
        "Joining Date": [
            "2021-06-15",
            "2020/01/10",
            "15-08-2019",
            "2022-03-01",
            None,
            "2021-11-20",
            "2020/01/10",
            "2023-01-05",
        ],
        "Salary": ["55,000", "72000", None, "65000", "48000", "81,000", "72000", "60,000"],
        "Department": ["IT", "HR", "it", "Finance", "HR", "Finance", "HR", "Marketing"],
        "Performance Score": [4.5, 3.8, None, 4.2, 2.9, 4.9, 3.8, 4.1],
    }

    df = pd.DataFrame(data)
    print("--- Original DataFrame ---")
    print(df)
    logger.info("Created a messy sample dataframe")

    # Method chaining pattern enabled by returning self
    cleaner = DataCleaning(df)
    df_clean = (
        cleaner.snake_case()
        .sanitize_text()
        .clean_salary()
        .standardize_dates()
        .remove_duplicates()
        .fill_missing_values()
        .get_result()
    )

    sns.set_theme(style="whitegrid")

    # Chart 1: Employee salary
    plt.figure(figsize=(12, 6))

    sns.barplot(
        x=df_clean["employee_id"].astype(str),
        y=df_clean["salary"],
        color="skyblue",
        edgecolor="black"
    )

    plt.xlabel("Employee ID")
    plt.ylabel("Salary")
    plt.title("Employee Salary by Employee ID")
    plt.tight_layout()
    plt.show()

    # Chart 2: Employee performance score
    plt.figure(figsize=(12, 6))

    sns.barplot(
        x=df_clean["employee_id"].astype(str),
        y=df_clean["performance_score"],
        color="lightgreen",
        edgecolor="black"
    )

    plt.xlabel("Employee ID")
    plt.ylabel("Performance Score")
    plt.title("Employee Performance Score by Employee ID")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df_clean,
        x="department",
        y="performance_score"
    )

    plt.xlabel("Department")
    plt.ylabel("Performance Score")
    plt.title("Performance Score Distribution by Department")
    plt.tight_layout()
    plt.show()

    print("\n--- Fully Cleaned DataFrame ---")
    print(df_clean)