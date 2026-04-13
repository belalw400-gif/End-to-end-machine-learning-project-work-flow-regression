"""
Data quality checks and validation for the ML project.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from src.logger import get_logger

logger = get_logger(__name__)


class DataQualityChecker:
    """Comprehensive data quality validation."""
    
    def __init__(self, df: pd.DataFrame, name: str = "Dataset"):
        self.df = df
        self.name = name
        self.issues = []
        self.warnings = []
        
    def check_missing_values(self, threshold: float = 0.5) -> Dict[str, float]:
        """Check for missing values above threshold."""
        missing_pct = (self.df.isnull().sum() / len(self.df))
        high_missing = missing_pct[missing_pct > threshold]
        
        if not high_missing.empty:
            for col, pct in high_missing.items():
                self.issues.append(f"Column '{col}' has {pct:.1%} missing values")
        
        return missing_pct.to_dict()
    
    def check_data_types(self) -> Dict[str, str]:
        """Verify expected data types."""
        dtypes = self.df.dtypes.to_dict()
        
        # Check for object columns that should be numeric
        numeric_cols = ['price', 'median_list_price', 'lat', 'lng']
        for col in numeric_cols:
            if col in self.df.columns and self.df[col].dtype == 'object':
                self.warnings.append(f"Column '{col}' is object type, should be numeric")
        
        return {str(k): str(v) for k, v in dtypes.items()}
    
    def check_duplicates(self) -> int:
        """Check for duplicate rows."""
        n_duplicates = self.df.duplicated().sum()
        if n_duplicates > 0:
            self.warnings.append(f"{n_duplicates} duplicate rows found")
        return n_duplicates
    
    def check_outliers(self) -> Dict[str, int]:
        """Check for potential outliers using IQR method."""
        outliers = {}
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            n_outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            outliers[col] = n_outliers
            
            if n_outliers > len(self.df) * 0.05:
                self.warnings.append(f"Column '{col}' has {n_outliers} outliers ({n_outliers/len(self.df)*100:.1f}%)")
        
        return outliers
    
    def check_price_column(self) -> Tuple[bool, List[str]]:
        """Validate price column."""
        problems = []
        
        if 'price' not in self.df.columns:
            problems.append("Price column not found")
            return False, problems
        
        if self.df['price'].dtype not in ['float64', 'int64']:
            problems.append(f"Price column has invalid type: {self.df['price'].dtype}")
        
        if (self.df['price'] <= 0).any():
            n_invalid = (self.df['price'] <= 0).sum()
            problems.append(f"{n_invalid} rows with price <= 0")
        
        if self.df['price'].isnull().any():
            n_null = self.df['price'].isnull().sum()
            problems.append(f"{n_null} missing price values")
        
        return len(problems) == 0, problems
    
    def check_date_column(self) -> Tuple[bool, List[str]]:
        """Validate date column."""
        problems = []
        
        if 'date' not in self.df.columns:
            problems.append("Date column not found")
            return False, problems
        
        try:
            dates = pd.to_datetime(self.df['date'])
            if dates.isnull().any():
                n_null = dates.isnull().sum()
                problems.append(f"{n_null} invalid date values")
        except Exception as e:
            problems.append(f"Date conversion error: {str(e)}")
        
        return len(problems) == 0, problems
    
    def check_feature_columns(self, expected_cols: List[str] = None) -> Tuple[bool, List[str]]:
        """Validate expected feature columns."""
        problems = []
        
        if expected_cols is None:
            expected_cols = ['price', 'date', 'zipcode', 'city_full', 'median_list_price']
        
        for col in expected_cols:
            if col not in self.df.columns:
                problems.append(f"Expected column not found: {col}")
        
        return len(problems) == 0, problems
    
    def run_all_checks(self) -> Dict:
        """Run all quality checks and return summary."""
        logger.info(f"🔍 Running quality checks on {self.name}...")
        
        summary = {
            "name": self.name,
            "shape": self.df.shape,
            "missing_values": self.check_missing_values(),
            "data_types": self.check_data_types(),
            "duplicates": self.check_duplicates(),
            "outliers": self.check_outliers(),
            "price_valid": self.check_price_column(),
            "date_valid": self.check_date_column(),
            "features_valid": self.check_feature_columns(),
            "issues": self.issues,
            "warnings": self.warnings,
        }
        
        # Log results
        if self.issues:
            logger.error(f"❌ {self.name} - {len(self.issues)} issues found:")
            for issue in self.issues:
                logger.error(f"   - {issue}")
        
        if self.warnings:
            logger.warning(f"⚠️  {self.name} - {len(self.warnings)} warnings:")
            for warning in self.warnings:
                logger.warning(f"   - {warning}")
        
        if not self.issues:
            logger.info(f"✅ {self.name} - All checks passed!")
        
        return summary


def validate_dataset(df: pd.DataFrame, split_name: str = "Dataset") -> bool:
    """Quick validation function."""
    checker = DataQualityChecker(df, split_name)
    summary = checker.run_all_checks()
    return len(checker.issues) == 0


# Example usage
if __name__ == "__main__":
    # Load and check processed data
    train_df = pd.read_csv("data/processed/train_fe.csv")
    checker = DataQualityChecker(train_df, "Training Data")
    summary = checker.run_all_checks()
    print("\n📊 Quality Check Summary:")
    print(f"  Issues: {len(checker.issues)}")
    print(f"  Warnings: {len(checker.warnings)}")
