# Windowed Cross-Correlation Analysis

This project provides a generalized Python workflow for performing **windowed cross-correlation (CCF) analysis** on time series data. It is designed to be dataset-agnostic—any collection of CSV files containing timestamped measurements can be dropped in for processing.

## Overview
The workflow automatically:
- Splits each dataset into user-defined windows from a configured column or timestamp bounds.  
- Computes cross-correlation functions between two selected variables.  
- Identifies the strongest correlation, lag, and directionality within each window.  
- Generates delta summaries comparing changes in correlation strength and lag between consecutive windows.  
- Batches results across all CSV files in a directory and exports a consolidated summary table.

## Key Features
- **Flexible column detection** – Auto-identifies timestamp and value columns, or allows custom regex patterns.  
- **Configurable filters** – Optional exclusion of outliers via numeric thresholds.  
- **Generalized input** – Works with any CSV containing a timestamp column and two numeric value columns.  
- **Batch processing** – Handles all files in a directory with a single command.  
- **Detailed summaries** – Outputs per-segment statistics and delta comparisons in a single CSV file.  
- **Robust error handling** – Provides clear status messages when data cannot be analyzed.  

## Configuration
Edit the configuration block at the top of `windowed-ccf-analysis.py` before running:
- Set `TIMESTAMP_COL` to the timestamp column name in the input CSV files.
- Set `WINDOW_COLUMN` when the input data already contains a column whose values define each analysis window.
- Leave `WINDOW_COLUMN = None` and edit `WINDOW_DEFINITIONS` when windows should be defined by timestamp bounds.
- Set `Y_COL_REGEX` and `X_COL_REGEX` to choose specific variables, or leave them as `None` to use the first two numeric columns.

## Output
The script produces a CSV file (`ccf_window_summary.csv`) containing:  
- File identifiers (and optional parsed IDs from filenames).  
- Per-segment correlation metrics (strength, lag, direction).  
- Delta metrics capturing how relationships shift between windows.  
- Metadata on data quality and sample counts.

## Use Cases
This workflow is well-suited for exploratory analysis of time-dependent relationships, such as:  
- Detecting changes in relationships between paired measurements.  
- Comparing behavior across user-defined intervals.  
- Screening datasets for shifts in dynamic correlations.
