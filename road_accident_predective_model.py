import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
# Define the file path prefix
FILE_PATH_PREFIX = 'files/' 

# --- Data Loading & Cleaning Functions ---

def clean_state_ut_column(df, state_col_name):
    """
    Standardizes the state/UT column in a DataFrame.
    Renames to 'State/UT', strips whitespace, and removes summary rows.
    Also handles known state name variations.
    """
    df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True).str.replace('"', '')
    normalized_state_col = ' '.join(state_col_name.split())

    state_col_real_name = None
    for col in df.columns:
        # Use case-insensitive matching
        if col.lower() == normalized_state_col.lower():
            state_col_real_name = col
            break

    if state_col_real_name:
        df.rename(columns={state_col_real_name: 'State/UT'}, inplace=True)
    else:
        print(f"Warning: Could not find state column '{normalized_state_col}'.")
        return pd.DataFrame()

    if 'State/UT' in df.columns:
        df['State/UT'] = df['State/UT'].str.strip()
        df = df[~df['State/UT'].str.lower().str.contains('total')]
        df = df[~df['State/UT'].str.lower().str.contains('all india')]
        df = df[~df['State/UT'].str.lower().str.contains('laddakh')]
        df = df[~df['State/UT'].str.lower().str.contains('lakshadweep')]
        # Set index *before* renaming based on index values
        df.set_index('State/UT', inplace=True)
        # Handle index renaming after setting index for consistency
        df.index = df.index.str.replace('D & N Haveli', 'Dadra and Nagar Haveli', case=False, regex=False)
        df.index = df.index.str.replace('Daman & Diu', 'Daman and Diu', case=False, regex=False)
        df.index = df.index.str.replace('A & N Islands', 'Andaman and Nicobar Islands', case=False, regex=False)
        df.index = df.index.str.replace('Delhi \(Ut\)', 'Delhi', case=False, regex=True)
        df.index = df.index.str.replace('Delhi', 'Delhi', case=False, regex=False) # Handle variations
        df.index = df.index.str.replace('Jammu & Kashmir', 'Jammu and Kashmir', case=False, regex=False)
        df.index = df.index.str.replace('Puducherry', 'Pondicherry', case=False, regex=False) # Common variation

    else:
        print("Warning: 'State/UT' column not found after renaming.")
        return pd.DataFrame()

    return df


def load_and_clean_df(filepath, state_col_name):
    """
    Loads a CSV file, cleans the state/UT column, and sets it as index.
    """
    full_path = f"{FILE_PATH_PREFIX}{filepath}"
    if not os.path.exists(full_path):
        print(f"Error: File not found at {full_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(full_path)
        df = clean_state_ut_column(df, state_col_name)
        # Group by index AFTER cleaning to handle states like D&N + Daman&Diu if needed
        df = df.groupby(df.index).sum(numeric_only=True)
        return df

    except Exception as e:
        print(f"Error processing file {full_path}: {e}")
        return pd.DataFrame()

def load_2022_actual_data(filepath):
    """
    Loads the new 2022 data file, filters for States/UTs only,
    and extracts the actual Accidents, Killed, and Injured numbers.
    """
    full_path = f"{FILE_PATH_PREFIX}{filepath}"
    if not os.path.exists(full_path):
        print(f"Error: 2022 data file not found at {full_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(full_path)
        df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)

        # Ensure 'Sl. No.' column exists before filtering
        if 'Sl. No.' in df.columns:
             # Filter for states/UTs (Sl. No. 1-36)
             df = df[pd.to_numeric(df['Sl. No.'], errors='coerce').between(1, 36, inclusive='both')]
        else:
             print("Warning: 'Sl. No.' column not found in 2022 actual data file. Attempting to proceed without filtering.")

        df = df.rename(columns={
            'State/UT/City': 'State/UT',
            'Total Traffic Accidents - Cases': 'Actual_Accidents_2022',
            'Total Traffic Accidents - Died': 'Actual_Killed_2022',
            'Total Traffic Accidents - Injured': 'Actual_Injured_2022'
        })

        # Make sure 'State/UT' column exists after rename
        if 'State/UT' not in df.columns:
            print("Error: 'State/UT' column not found after renaming in 2022 actual data.")
            return pd.DataFrame()

        df['State/UT'] = df['State/UT'].str.strip()
        df.set_index('State/UT', inplace=True)

        # Handle index renaming after setting index
        df.index = df.index.str.replace('D & N Haveli', 'Dadra and Nagar Haveli', case=False, regex=False)
        df.index = df.index.str.replace('Daman & Diu', 'Daman and Diu', case=False, regex=False)
        df.index = df.index.str.replace('A & N Islands', 'Andaman and Nicobar Islands', case=False, regex=False)
        df.index = df.index.str.replace('Delhi \(Ut\)', 'Delhi', case=False, regex=True)
        df.index = df.index.str.replace('Delhi', 'Delhi', case=False, regex=False)
        df.index = df.index.str.replace('Jammu & Kashmir', 'Jammu and Kashmir', case=False, regex=False)
        df.index = df.index.str.replace('Puducherry', 'Pondicherry', case=False, regex=False)

        # Select only the relevant columns and convert to numeric
        actual_data = df[['Actual_Accidents_2022', 'Actual_Killed_2022', 'Actual_Injured_2022']]
        actual_data = actual_data.apply(pd.to_numeric, errors='coerce').fillna(0)
        # Group after cleaning
        actual_data = actual_data.groupby(actual_data.index).sum()

        print("Successfully processed 2022 actual data.")
        return actual_data

    except Exception as e:
        print(f"Error processing 2022 data file {full_path}: {e}")
        return pd.DataFrame()

def convert_cols_to_numeric(df, cols):
    """
    Converts a list of columns in a DataFrame to numeric,
    coercing any errors to NaN. Fills resulting NaNs with 0.
    """
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.fillna(0, inplace=True) # Fill NaNs introduced by coercion
    return df


def load_historical_data_2006_2021(files):
    """
    Loads and consolidates all historical data from 2006-2021
    for all three metrics: Accidents, Killed, Injured.
    """
    print("Loading and consolidating historical data (2006-2021)...")
    all_metrics_df = {}
    years_2006_14 = [str(y) for y in range(2006, 2015)]
    years_2015_17 = ['2015', '2016', '2017']
    years_2018_21 = ['2018', '2019', '2020', '2021']

    for metric in ['Accidents', 'Killed', 'Injured']:
        print(f"--- Processing Metric: {metric} ---")
        try:
            # 1. Load 2006-2014 Rural and Urban data
            df_rural = load_and_clean_df(files.get(f'rural_{metric.lower()}_2006_14', ''), 'States/UTs')
            df_urban = load_and_clean_df(files.get(f'urban_{metric.lower()}_2006_14', ''), 'States/UTs')

            if df_rural.empty and df_urban.empty:
                print(f"Warning: Both Rural and Urban 2006-14 files missing for {metric}. Skipping this period.")
                df_total_2006_14 = pd.DataFrame() # Empty df
            else:
                df_rural = convert_cols_to_numeric(df_rural, years_2006_14)
                df_urban = convert_cols_to_numeric(df_urban, years_2006_14)
                cols_r = [y for y in years_2006_14 if y in df_rural.columns]
                cols_u = [y for y in years_2006_14 if y in df_urban.columns]
                # Use outer join and fillna(0)
                df_total_2006_14 = df_rural[cols_r].add(df_urban[cols_u], fill_value=0).fillna(0)
                print(f"Loaded {metric} 2006-14.")

            # 2. Load 2014-2017 Total data
            file_2014_17 = files.get(f'total_{metric.lower()}_2014_17', '')
            df_total_2014_17_raw = load_and_clean_df(file_2014_17, 'States/UTs')

            if df_total_2014_17_raw.empty:
                print(f"Warning: File missing or empty for {metric} 2014-17. Skipping this period.")
                df_total_2015_17 = pd.DataFrame()
            else:
                # Find columns like '...2015', '...2016', '...2017'
                cols_2014_17_rename = {}
                for year in ['2014', '2015', '2016', '2017']:
                    year_col = [col for col in df_total_2014_17_raw.columns if f" {year}" in col and 'Share' not in col and 'Rank' not in col]
                    if not year_col: # Fallback
                         year_col = [col for col in df_total_2014_17_raw.columns if year in col and 'Share' not in col and 'Rank' not in col]

                    if year_col:
                        cols_2014_17_rename[year_col[0]] = year

                df_total_2014_17 = df_total_2014_17_raw.rename(columns=cols_2014_17_rename)
                existing_years_2015_17 = [y for y in years_2015_17 if y in df_total_2014_17.columns]
                if not existing_years_2015_17:
                    print(f"Warning: No valid columns found for 2015-2017 in {file_2014_17}")
                    df_total_2015_17 = pd.DataFrame(index=df_total_2014_17_raw.index)
                else:
                    df_total_2015_17 = convert_cols_to_numeric(df_total_2014_17, existing_years_2015_17)[existing_years_2015_17]
                print(f"Loaded {metric} 2015-17.")


            # 3. Load 2018-2021 Total data
            print(f"Loading 2018-21 data for {metric}...")
            df_total_2018_21 = pd.DataFrame()
            
            try:
                if metric == 'Accidents':
                    # Use the single total accidents file
                    df_acc_raw = load_and_clean_df(files['total_accidents_2018_21'], 'States/UTs')
                    rename_map = {
                        'State/UT-Wise Total Number of Road Accidents during 2018': '2018',
                        'State/UT-Wise Total Number of Road Accidents during 2019': '2019',
                        'State/UT-Wise Total Number of Road Accidents during 2020': '2020',
                        'State/UT-Wise Total Number of Road Accidents during 2021 - Numbers': '2021'
                    }
                    df_total_2018_21 = df_acc_raw.rename(columns=rename_map)

                elif metric == 'Killed':
                    # Load and sum the three breakdown files
                    df_nh = load_and_clean_df(files['nh_killed_2018_21'], 'States/UTs')
                    df_sh = load_and_clean_df(files['sh_killed_2018_21'], 'States/UTs')
                    df_other = load_and_clean_df(files['other_roads_2018_21'], 'State/UT') # Note: 'State/UT'

                    rename_nh = {
                        'Total Number of Persons Killed on NH during 2018': '2018',
                        'Total Number of Persons Killed on NH during 2019': '2019',
                        'Total Number of Persons Killed on NH during 2020': '2020',
                        'Total Number of Persons Killed on NH during 2021 - Number': '2021'
                    }
                    rename_sh = {
                        'Total Number of Persons Killed in Road Accidents on State Highways during 2018': '2018',
                        'Total Number of Persons Killed in Road Accidents on State Highways during 2019': '2019',
                        'Total Number of Persons Killed in Road Accidents on State Highways during 2020': '2020',
                        'Total Number of Persons Killed in Road Accidents on State Highways during 2021 - Number': '2021'
                    }
                    rename_other = {
                        'Killed - 2018': '2018',
                        'Killed - 2019': '2019',
                        'Killed - 2020': '2020',
                        'Killed - 2021 - Number': '2021'
                    }
                    
                    df_nh = df_nh.rename(columns=rename_nh)[years_2018_21].apply(pd.to_numeric, errors='coerce')
                    df_sh = df_sh.rename(columns=rename_sh)[years_2018_21].apply(pd.to_numeric, errors='coerce')
                    df_other = df_other.rename(columns=rename_other)[years_2018_21].apply(pd.to_numeric, errors='coerce')

                    df_total_2018_21 = df_nh.add(df_sh, fill_value=0).add(df_other, fill_value=0).fillna(0)

                elif metric == 'Injured':
                    # Load and sum the three breakdown files
                    df_nh = load_and_clean_df(files['nh_injured_2018_21'], 'States/UTs')
                    df_sh = load_and_clean_df(files['sh_injured_2018_21'], 'States/UTs')
                    df_other = load_and_clean_df(files['other_roads_2018_21'], 'State/UT') # Note: 'State/UT'

                    rename_nh = {
                        'Total Number of Persons Injured on NH during 2018': '2018',
                        'Total Number of Persons Injured on NH during 2019': '2019',
                        'Total Number of Persons Injured on NH during 2020': '2020',
                        'Total Number of Persons Injured on NH during 2021 - Number': '2021'
                    }
                    rename_sh = {
                        'Total Number of Persons Injured in Road Accidents on State Highways during 2018': '2018',
                        'Total Number of Persons Injured in Road Accidents on State Highways during 2019': '2019',
                        'Total Number of Persons Injured in Road Accidents on State Highways during 2020': '2020',
                        'Total Number of Persons Injured in Road Accidents on State Highways during 2021': '2021' # This one is different
                    }
                    rename_other = {
                        'Injured - 2018': '2018',
                        'Injured - 2019': '2019',
                        'Injured - 2020': '2020',
                        'Injured - 2021 - Number': '2021'
                    }

                    df_nh = df_nh.rename(columns=rename_nh)[years_2018_21].apply(pd.to_numeric, errors='coerce')
                    df_sh = df_sh.rename(columns=rename_sh)[years_2018_21].apply(pd.to_numeric, errors='coerce')
                    df_other = df_other.rename(columns=rename_other)[years_2018_21].apply(pd.to_numeric, errors='coerce')
                    
                    df_total_2018_21 = df_nh.add(df_sh, fill_value=0).add(df_other, fill_value=0).fillna(0)
            
            except Exception as e:
                print(f"Error processing 2018-2021 data for {metric}: {e}")
                df_total_2018_21 = pd.DataFrame() # Ensure it's an empty df on error

            if df_total_2018_21.empty:
                 print(f"Warning: File missing or empty for {metric} 2018-21. Skipping this period.")
            else:
                existing_years_2018_21 = [y for y in years_2018_21 if y in df_total_2018_21.columns]
                if not existing_years_2018_21:
                    print(f"Warning: No valid columns found for 2018-2021 in {metric} files")
                    df_total_2018_21 = pd.DataFrame(index=df_total_2018_21.index)
                else:
                    df_total_2018_21 = convert_cols_to_numeric(df_total_2018_21, existing_years_2018_21)[existing_years_2018_21]
                print(f"Loaded {metric} 2018-21.")


            # 4. Concatenate all years using outer join
            dfs_to_concat = [df for df in [df_total_2006_14, df_total_2015_17, df_total_2018_21] if not df.empty]
            if not dfs_to_concat:
                print(f"Error: No data loaded for metric {metric}. Cannot continue.")
                continue # Skip to next metric

            df_all_years = pd.concat(dfs_to_concat, axis=1, join='outer')

            # 5. Rename columns to be specific and select final range
            final_col_map = {str(year): f'Total_{metric}_{year}' for year in range(2006, 2022)}
            df_all_years.rename(columns=final_col_map, inplace=True)
            # Select only the columns that actually exist after rename
            final_cols_exist = [f'Total_{metric}_{year}' for year in range(2006, 2022) if f'Total_{metric}_{year}' in df_all_years.columns]
            all_metrics_df[metric] = df_all_years[final_cols_exist]

            print(f"Successfully consolidated historical data for: {metric}")

        except Exception as e:
            print(f"Error consolidating historical data for {metric}: {e}")
            import traceback
            traceback.print_exc() # Print detailed error


    # Merge all metrics (Accidents, Killed, Injured) into one big DataFrame
    if 'Accidents' not in all_metrics_df or 'Killed' not in all_metrics_df or 'Injured' not in all_metrics_df:
        print("Error: Failed to load all three required historical metrics. Aborting.")
        return pd.DataFrame()

    final_historical_df = pd.concat(all_metrics_df.values(), axis=1, join='outer')

    # Clean final index AFTER merging
    final_historical_df.index = final_historical_df.index.str.replace('D & N Haveli', 'Dadra and Nagar Haveli', case=False, regex=False)
    final_historical_df.index = final_historical_df.index.str.replace('Daman & Diu', 'Daman and Diu', case=False, regex=False)
    final_historical_df.index = final_historical_df.index.str.replace('A & N Islands', 'Andaman and Nicobar Islands', case=False, regex=False)
    final_historical_df.index = final_historical_df.index.str.replace('Delhi \(Ut\)', 'Delhi', case=False, regex=True)
    final_historical_df.index = final_historical_df.index.str.replace('Delhi', 'Delhi', case=False, regex=False)
    final_historical_df.index = final_historical_df.index.str.replace('Jammu & Kashmir', 'Jammu and Kashmir', case=False, regex=False)
    final_historical_df.index = final_historical_df.index.str.replace('Puducherry', 'Pondicherry', case=False, regex=False)

    # Group by index to combine states with multiple names (e.g., D&N, Daman)
    final_historical_df = final_historical_df.groupby(final_historical_df.index).sum()

    final_historical_df.fillna(0, inplace=True)
    # Convert to integer, handling potential non-numeric placeholders if any survived
    for col in final_historical_df.columns:
        final_historical_df[col] = pd.to_numeric(final_historical_df[col], errors='coerce').fillna(0).astype(int)


    print("Historical data consolidation complete.")
    return final_historical_df


# --- Time-Series Forecasting Functions ---

def predict_2022_trend(row, base_col_name, years):
    """
    Trains a simple linear regression model on available years of data
    for a single state and predicts the 2022 value.
    """
    # Create the series with available years, ensuring index is integer year
    y_train_values = {year: row.get(f'{base_col_name}_{year}', np.nan) for year in years}
    y_train = pd.Series(y_train_values).dropna() # Drop years with missing data

    # Check if there's enough data (at least 2 points)
    if len(y_train) < 2 or y_train.sum() == 0:
        return 0 # Predict 0 if not enough data or all zeros

    # Prepare X (years) and y (values) for the model
    X_train_years = y_train.index.astype(int).values.reshape(-1, 1)
    y_train_vals = y_train.values

    model = LinearRegression()
    model.fit(X_train_years, y_train_vals)

    prediction = model.predict([[2022]])
    return int(max(0, prediction[0])) # Don't predict negative values


def run_time_series_forecast(analysis_df, actual_2022_df):
    """
    Runs the trend forecast (EXCLUDING 2020) and validates it against
    the actual 2022 data.
    """
    print("\n" + "="*50)
    print("Running 2022 Time-Series Forecast & Validation (Excluding 2020 Model)")
    print("="*50 + "\n")

    try:
        # --- 1. Define Years and Run Forecast ---
        # MODIFICATION: Train on 2006-2019 + 2021 (15 years)
        years_2006_to_2019 = list(range(2006, 2020))
        years_to_train = years_2006_to_2019 + [2021] 

        # Check which years actually have data in the loaded dataframe
        available_years = {}
        for metric in ['Accidents', 'Killed', 'Injured']:
            available_years[metric] = [y for y in years_to_train if f'Total_{metric}_{y}' in analysis_df.columns]
            if not available_years[metric]:
                print(f"Error: No historical data columns found for {metric}. Cannot forecast.")
                return
            else:
                print(f"Using {len(available_years[metric])} years of data (Excluding 2020) for {metric} forecast.")


        print(f"Forecasting 2022 totals based on available years (excluding 2020)...")

        analysis_df['Predicted_Accidents_2022'] = analysis_df.apply(
            predict_2022_trend, base_col_name='Total_Accidents', years=available_years['Accidents'], axis=1
        )
        analysis_df['Predicted_Killed_2022'] = analysis_df.apply(
            predict_2022_trend, base_col_name='Total_Killed', years=available_years['Killed'], axis=1
        )
        analysis_df['Predicted_Injured_2022'] = analysis_df.apply(
            predict_2022_trend, base_col_name='Total_Injured', years=available_years['Injured'], axis=1
        )

        # --- 2. Merge with Actual 2022 Data ---
        analysis_df.index = analysis_df.index.str.strip()
        actual_2022_df.index = actual_2022_df.index.str.strip()

        # Use outer join and reset index to handle potential index mismatches after grouping/renaming
        analysis_df.reset_index(inplace=True)
        actual_2022_df.reset_index(inplace=True)

        # Merge on 'State/UT', ensuring consistent naming
        validation_merged_df = pd.merge(analysis_df, actual_2022_df, on='State/UT', how='outer')
        validation_merged_df.set_index('State/UT', inplace=True) # Set index back after merge

        validation_merged_df.fillna(0, inplace=True) # Fill NaNs created by outer join

        # --- 3. Validate and Report ---
        print("\n--- 2022 FORECAST VALIDATION (Model: Linear Trend Excluding 2020) ---")

        # Filter for states where we have BOTH prediction AND actual 2022 data > 0
        validation_df_filtered = validation_merged_df[
            (validation_merged_df['Predicted_Accidents_2022'] > 0) | (validation_merged_df['Actual_Accidents_2022'] > 0) |
            (validation_merged_df['Predicted_Killed_2022'] > 0) | (validation_merged_df['Actual_Killed_2022'] > 0) |
            (validation_merged_df['Predicted_Injured_2022'] > 0) | (validation_merged_df['Actual_Injured_2022'] > 0)
        ].copy() # Keep row if either predicted or actual is non-zero

        # Further filter for rows where actual data is specifically non-zero for RMSE calc
        rmse_calc_df = validation_df_filtered[
            (validation_df_filtered['Actual_Accidents_2022'] > 0) &
            (validation_df_filtered['Actual_Killed_2022'] > 0) &
            (validation_df_filtered['Actual_Injured_2022'] > 0)
        ].copy()


        if rmse_calc_df.empty:
            print("Error: No overlapping states with valid actual 2022 data (>0) found for validation.")
            # Show the merged df for debugging
            print("Sample of merged data before validation filtering:")
            print(validation_merged_df[['Predicted_Accidents_2022', 'Actual_Accidents_2022', 'Predicted_Killed_2022', 'Actual_Killed_2022']].head())
            return

        print(f"Validating forecast on {len(rmse_calc_df)} states with Actual 2022 data > 0.")

        accidents_rmse = np.sqrt(mean_squared_error(
            rmse_calc_df['Actual_Accidents_2022'],
            rmse_calc_df['Predicted_Accidents_2022']
        ))
        killed_rmse = np.sqrt(mean_squared_error(
            rmse_calc_df['Actual_Killed_2022'],
            rmse_calc_df['Predicted_Killed_2022']
        ))
        injured_rmse = np.sqrt(mean_squared_error(
            rmse_calc_df['Actual_Injured_2022'],
            rmse_calc_df['Predicted_Injured_2022']
        ))

        print(f"\nTotal Accidents 2022 Forecast RMSE: {accidents_rmse:.2f}")
        print(f"Total Killed 2022 Forecast RMSE: {killed_rmse:.2f}")
        print(f"Total Injured 2022 Forecast RMSE: {injured_rmse:.2f}")

        # --- 4. Show Comparison Table ---
        # Use 2021 data as the comparison point
        results_df = validation_df_filtered[[
            'Total_Accidents_2021', 'Predicted_Accidents_2022', 'Actual_Accidents_2022',
            'Total_Killed_2021', 'Predicted_Killed_2022', 'Actual_Killed_2022',
            'Total_Injured_2021', 'Predicted_Injured_2022', 'Actual_Injured_2022'
        ]].fillna(0).astype(int) # Ensure integer type after fillna

        results_df['Killed_Error (Pred - Actual)'] = results_df['Predicted_Killed_2022'] - results_df['Actual_Killed_2022']

        print("\n--- Comparison: Predicted 2022 vs. Actual 2022 (Top 10 States by Actual Deaths) ---")
        print(results_df.sort_values(by='Actual_Killed_2022', ascending=False).head(10))

        # --- 5. Plot Validation ---
        
        # Plot for 'Killed'
        plt.figure(figsize=(10, 7))
        sns.regplot(data=results_df, x='Actual_Killed_2022', y='Predicted_Killed_2022', scatter_kws={'alpha':0.6})
        max_val = max(results_df['Actual_Killed_2022'].max(), results_df['Predicted_Killed_2022'].max())
        plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')
        plt.title('2022 Forecast Validation: Actual vs. Predicted Deaths (Trend Excluding 2020)', fontsize=14, pad=20)
        plt.xlabel('Actual Persons Killed (2022)', fontsize=12)
        plt.ylabel('Predicted Persons Killed (2022)', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot for 'Accidents'
        plt.figure(figsize=(10, 7))
        sns.regplot(data=results_df, x='Actual_Accidents_2022', y='Predicted_Accidents_2022', scatter_kws={'alpha':0.6})
        max_val = max(results_df['Actual_Accidents_2022'].max(), results_df['Predicted_Accidents_2022'].max())
        plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')
        plt.title('2022 Forecast Validation: Actual vs. Predicted Accidents (Trend Excluding 2020)', fontsize=14, pad=20)
        plt.xlabel('Actual Total Accidents (2022)', fontsize=12)
        plt.ylabel('Predicted Total Accidents (2022)', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred during Time-Series Forecast: {e}")
        import traceback
        traceback.print_exc()


# --- Main Execution ---

def main():
    """
    Main function to run the data preprocessing and 2022 time-series forecast.
    """

    # Define file paths with all 21 files provided
    files = {
        # 2022 actual data file
        'actual_2022': "StateUTsCity-wise Number of Cases Reported and Persons Injured & Died due to Traffic Accidents during 2022.csv",

        # 2006-2014 files
        'urban_accidents_2006_14': "Urban_Total_Accidents_2006-14.csv",
        'urban_killed_2006_14': "Urban_Persons_Killed_2006-14.csv",
        'urban_injured_2006_14': "Urban_Persons_Injured_2006-14.csv",
        'rural_accidents_2006_14': "Rural_Total_accidents_2006-14.csv",
        'rural_killed_2006_14': "Rural_Persons_Killed_2006-14.csv",
        'rural_injured_2006_14': "Rural_Persons_Injured_2006-14.csv",

        # 2014-2017 files
        'total_accidents_2014_17': "StateUT-wise Total Number of Road Accidents in India from 2014 to 2017.csv",
        'total_killed_2014_17': "StateUT-wise Total Number of Persons Killed in Road Accidents in India from 2014 to 2017.csv",
        'total_injured_2014_17': "StateUT-wise Total Number of Persons Injured in Road Accidents in India from 2014 to 2017.csv",

        # 2018-2021 files (BREAKDOWNS)
        # Accidents (Total)
        'total_accidents_2018_21': "StatesUTs-wise Total Number of Road Accidents in India from 2018 to 2021.csv",
        
        # Killed (Breakdowns)
        'nh_killed_2018_21': "StatesUTs-wise Total Number of Persons Killed in Road Accidents on National Highways from 2018 to 2021.csv",
        'sh_killed_2018_21': "StatesUTs-wise Total Number of Persons Killed in Road Accidents on State Highways from 2018 to 2021.csv",
        
        # Injured (Breakdowns)
        'nh_injured_2018_21': "StatesUTs-wise Total Number of Persons Injured in Road Accidents on National Highways from 2018 to 2021.csv",
        'sh_injured_2018_21': "StatesUTs-wise Total Number of Persons Injured in Road Accidents on State Highways from 2018 to 2021.csv",
        
        # Other Roads (Contains Accidents, Killed, AND Injured breakdowns)
        'other_roads_2018_21': "StatesUTs-wise Total number of Fatal Road Accidents, Total Road Accidents, Persons Killed and Injured on Other Roads from 2018 to 2021.csv",

        # Other 2021 files (Not used in this forecast model but available)
        'impact_vehicles_2021': "StatesUTs-wise Accidents classified according to type of impacting vehiclesobjects during 2021.csv",
        'traffic_violations_2021': "StatesUTs-wise Accidents Classified according to Type of Traffic Violations during 2021.csv",
        'weather_2021': "StatesUTs-wise Accidents Classified according to Type of Weather Condition during 2021.csv",
        'fatal_nh_2018_21': "StatesUTs-wise Total Number of Fatal Road Accidents on National Highways from 2018 to 2021.csv",
        'fatal_sh_2018_21': "StatesUTs-wise Total Number of Fatal Road Accidents on State Highways from 2018 to 2021.csv",
    }


    print("Starting data preprocessing for 2022 prediction model (Excluding 2020)...")

    # --- Load Historical Data (2006-2021) ---
    # We still load all data, as we need 2021 data for the final comparison table
    historical_data = load_historical_data_2006_2021(files)

    # --- Load 2022 Actual Data ---
    actual_2022_df = load_2022_actual_data(files['actual_2022'])

    # --- Merge historical and actual data (if needed later, but forecast runs on historical) ---
    if historical_data.empty:
        print("No historical data was loaded successfully. Exiting.")
        return
        
    if actual_2022_df.empty:
        print("No actual 2022 data was loaded successfully. Exiting.")
        return

    analysis_df = historical_data.copy()

    analysis_df.fillna(0, inplace=True)
    # Ensure all data is int after loading and merging
    for col in analysis_df.columns:
        analysis_df[col] = pd.to_numeric(analysis_df[col], errors='coerce').fillna(0).astype(int)
    print("\nData preprocessing complete.")
    
    # --- Run 2022 Time-Series Forecast & Validation ---
    run_time_series_forecast(analysis_df, actual_2022_df)

    print("\n" + "="*50)
    print("All tasks complete.")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()

