import numpy as np
import pandas as pd
import sys
import os

def normalize_matrix(matrix):
    norm_matrix = matrix / np.sqrt((matrix**2).sum(axis=0))
    return norm_matrix

def calculate_ideal_solutions(weighted_matrix, impacts):
    ideal_positive = []
    ideal_negative = []
    
    for i in range(weighted_matrix.shape[1]):
        if impacts[i] == '+':
            ideal_positive.append(np.max(weighted_matrix[:, i]))
            ideal_negative.append(np.min(weighted_matrix[:, i]))
        else:
            ideal_positive.append(np.min(weighted_matrix[:, i]))
            ideal_negative.append(np.max(weighted_matrix[:, i]))
    
    return np.array(ideal_positive), np.array(ideal_negative)

def topsis(data, weights, impacts):
    matrix = data.iloc[:, 1:].values
    normalized_matrix = normalize_matrix(matrix)
    weighted_matrix = normalized_matrix * weights
    
    ideal_positive, ideal_negative = calculate_ideal_solutions(weighted_matrix, impacts)
    
    distance_positive = np.sqrt(((weighted_matrix - ideal_positive) ** 2).sum(axis=1))
    distance_negative = np.sqrt(((weighted_matrix - ideal_negative) ** 2).sum(axis=1))
    
    scores = distance_negative / (distance_positive + distance_negative)
    data['TOPSIS Score'] = scores
    data['Rank'] = pd.Series(scores).rank(ascending=False).astype(int)
    
    return data

def convert_to_csv(input_file):
    output_csv = "102217009-data.csv"
    file_ext = os.path.splitext(input_file)[-1].lower()
    
    if file_ext == '.xlsx':
        try:
            data = pd.read_excel(input_file)
            data.to_csv(output_csv, index=False)
            print(f"Converted {input_file} to {output_csv}")
            return output_csv
        except FileNotFoundError:
            print(f"Error: File '{input_file}' not found.")
            sys.exit(1)
    elif file_ext == '.csv':
        print(f"Using {input_file} as input CSV file")
        return input_file
    else:
        print("Error: Unsupported file format. Only .xlsx and .csv files are accepted.")
        sys.exit(1)

def validate_inputs(data, weights, impacts):
    if len(weights) != data.shape[1] - 1:
        print("Error: Number of weights must match the number of criteria.")
        sys.exit(1)
    
    if len(impacts) != data.shape[1] - 1:
        print("Error: Number of impacts must match the number of criteria.")
        sys.exit(1)
    
    if not all(impact in ['+', '-'] for impact in impacts):
        print("Error: Impacts must be either '+' or '-'.")
        sys.exit(1)
    
    for col in data.columns[1:]:
        if not pd.api.types.is_numeric_dtype(data[col]):
            print(f"Error: Column '{col}' contains non-numeric values. Please ensure all values are numeric.")
            sys.exit(1)

def main(input_file, weights, impacts, result_file):
    csv_file = convert_to_csv(input_file)
    data = pd.read_csv(csv_file)
    
    if data.shape[1] < 3:
        print("Error: Input file must contain three or more columns.")
        sys.exit(1)
    
    weights = np.array([float(w) for w in weights.split(',')])
    impacts = impacts.split(',')
    
    validate_inputs(data, weights, impacts)
    
    result = topsis(data, weights, impacts)
    result.to_csv(result_file, index=False)
    print(f"Results saved to {result_file}")

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python topsis.py <InputDataFile> <Weights> <Impacts> <ResultFileName>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    weights = sys.argv[2]
    impacts = sys.argv[3]
    result_file = sys.argv[4]
    
    main(input_file, weights, impacts, result_file)
