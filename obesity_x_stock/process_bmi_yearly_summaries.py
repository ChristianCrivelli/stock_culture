import os
import pandas as pd

def process_bmi_yearly_summaries(input_dir="./data/bmi_data", output_dir="./data/bmi_summaries"):
    """
    Reads all BMI files, ensures proper sorting by Sex and Year,
    calculates Year-over-Year (YoY) prevalence differences, 
    and saves clean outputs per country.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Locate all CSV datasets
    bmi_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    for file_name in bmi_files:
        file_path = os.path.join(input_dir, file_name)
        df = pd.read_csv(file_path)
        
        # Sort chronologically within each Sex group
        df = df.sort_values(by=['Sex', 'Year']).reset_index(drop=True)
        
        # Core target columns from the dataset
        obesity_col = 'Prevalence of BMI>=30 kg/m² (obesity)'
        morbid_obesity_col = 'Prevalence of BMI >=40 kg/m² (morbid obesity)'
        underweight_col = 'Prevalence of BMI<18.5 kg/m² (underweight)'
        
        # Calculate Year-over-Year shifts using .diff() per Sex group
        df['YoY_Obesity_Prevalence_Diff'] = df.groupby('Sex')[obesity_col].diff()
        df['YoY_Morbid_Obesity_Prevalence_Diff'] = df.groupby('Sex')[morbid_obesity_col].diff()
        df['YoY_Underweight_Prevalence_Diff'] = df.groupby('Sex')[underweight_col].diff()
        
        # Build clean dataframe structure
        cols_to_keep = [
            'Year', 'Sex', 'Country/Region/World', 'ISO',
            obesity_col, 'YoY_Obesity_Prevalence_Diff',
            morbid_obesity_col, 'YoY_Morbid_Obesity_Prevalence_Diff',
            underweight_col, 'YoY_Underweight_Prevalence_Diff'
        ]
        
        # Filter existing columns to handle any schema variances
        clean_df = df[[c for c in cols_to_keep if c in df.columns]]
        
        # Format the output file name cleanly
        clean_name = file_name.replace('NCD_RisC_Nature_2026_BMI_age_standardised_', '')
        output_path = os.path.join(output_dir, clean_name)
        
        # Save output to CSV
        clean_df.to_csv(output_path, index=False)
        print(f"Exported summary dataset to: {output_path}")

if __name__ == "__main__":
    # Get the directory where this script file is actually located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct absolute paths relative to the script location
    base_input_dir = os.path.join(script_dir, "data", "bmi_data")
    base_output_dir = os.path.join(script_dir, "data", "bmi_summaries")
    
    print(f"Looking for BMI data in: {base_input_dir}")
    
    process_bmi_yearly_summaries(input_dir=base_input_dir, output_dir=base_output_dir)