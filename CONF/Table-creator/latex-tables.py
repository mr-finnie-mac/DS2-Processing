import pandas as pd
import os

def generate_latex_table(csv_file, filename):
    # Load CSV file
    df = pd.read_csv(csv_file)
    
    # Assuming the CSV has 'Method', 'Test Size', and 'MAE' columns
    pivot_df = df.pivot_table(values='MAE', index='Method', columns='Test Size', aggfunc='mean')
    pivot_df = pivot_df.reset_index()
    
    # Start LaTeX table
    latex_table = f"""
    % Table generated for {filename}
    \begin{{table}}[]
        \centering
        \resizebox{{\columnwidth}}{{!}}{{%
        \begin\{{tabular}}{{|c|" + "|c|" * (len(pivot_df.columns) - 1) + "}} 
            \hline
    """
    
    # Add header
    headers = ["\\textbf{" + str(col) + "}" for col in pivot_df.columns]
    latex_table += " & ".join(headers) + " \\\\ \hline\n"
    
    # Add rows
    for _, row in pivot_df.iterrows():
        formatted_values = [f"{val:.6f}" if isinstance(val, (int, float)) else str(val) for val in row]
        latex_table += " & ".join(formatted_values) + " \\\\ \hline\n"
    
    # End LaTeX table
    latex_table += f"""
        \end{{tabular}}%
        }}
        \caption{{Mean Absolute Error (MAE) for {filename}}}
        \label{{tab:{filename.replace('.csv', '')}}}
    \end{{table}}
    """
    
    return latex_table

def process_folder(folder_path, output_txt):
    """Process all CSV files in a folder and append their LaTeX tables to a text file."""
    with open(output_txt, "w", encoding="utf-8") as f:
        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                file_path = os.path.join(folder_path, file)
                latex_code = generate_latex_table(file_path, file)
                f.write(latex_code + "\n\n")
    print(f"LaTeX tables saved to {output_txt}")


# Example usage
folder_path = "./CONF/Table-creator/sheets"  # Replace with your actual folder path
output_txt = "./CONF/Table-creator/sheets/output_tables.txt"
process_folder(folder_path, output_txt)
