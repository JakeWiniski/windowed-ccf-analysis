# Windowed Cross-Correlation Analysis

This project provides a generalized Python workflow for performing **windowed cross-correlation (CCF) analysis** on time series data. It is designed to be dataset-agnostic—any collection of CSV files containing timestamped measurements can be dropped in for processing.

## Overview
The workflow automatically:
- Splits each dataset into two predefined time windows (default: **10–72 hours** and **post-72 hours** relative to the first timestamp).  
- Computes cross-correlation functions between two selected variables.  
- Identifies the strongest correlation, lag, and directionality within each window.  
- Generates a “delta” summary comparing changes in correlation strength and lag between the two windows.  
- Batches results across all CSV files in a directory and exports a consolidated summary table.

## Key Features
- **Flexible column detection** – Auto-identifies timestamp and value columns, or allows custom regex patterns.  
- **Configurable filters** – Optional exclusion of outliers via numeric thresholds.  
- **Generalized input** – Works with any CSV containing a `t_stamp` column and two numeric value columns.  
- **Batch processing** – Handles all files in a directory with a single command.  
- **Detailed summaries** – Outputs per-segment statistics and delta comparisons in a single CSV file.  
- **Robust error handling** – Provides clear status messages when data cannot be analyzed.  

## Output
The script produces a CSV file (`ccf_two_window_summary.csv`) containing:  
- File identifiers (and optional parsed IDs from filenames).  
- Per-segment correlation metrics (strength, lag, direction).  
- Delta metrics capturing how relationships shift between windows.  
- Metadata on data quality and sample counts.

## Use Cases
This workflow is well-suited for exploratory analysis of time-dependent relationships, such as:  
- Detecting variable coupling before and after an event.  
- Comparing early vs. late behavior in experiments.  
- Screening datasets for shifts in dynamic correlations.
