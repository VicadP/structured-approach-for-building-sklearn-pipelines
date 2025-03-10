import pandas as pd
import numpy as np

def get_averaged_predictions(predictions_paths, output_path, weights="uniform"):
    """
    Computes the weighted average of multiple prediction files and saves the result.

    Params:
        predictions_paths (List[str]): List of file paths containing prediction CSV files.
        output_path (str): Path to save the averaged predictions.
        weights (Union[str, List[float]]): Either "uniform" for equal weighting or a list of weights for each file.

    Raises:
        ValueError: If weights are not "uniform" or do not match the number of input files.
    """
    num_files = len(predictions_paths)

    if weights == "uniform":
        _weights = np.ones(num_files)
    elif isinstance(weights, (list, np.ndarray)) and len(weights) == num_files:
        _weights = np.array(weights) / np.sum(weights)
    else:
        raise ValueError("Weights should be 'uniform' or a list with one weight per file.")
    
    avereged_predictions = None

    for file_index, (file_path, _weight) in enumerate(zip(predictions_paths, _weights)):
        print(f"Parsing: {file_path} with weight: {_weight}")
        predictions = pd.read_csv(file_path)
        predictions.iloc[:, 1] *= _weight
        if avereged_predictions is None:
            avereged_predictions = predictions
        else:
            avereged_predictions.iloc[:, 1] += predictions.iloc[:, 1]

    avereged_predictions.iloc[:, 1] /= np.sum(_weights)
    avereged_predictions.to_csv(output_path, index=False)
    print(f"Wrote averaged predictions to {output_path}")

if __name__ == '__main__':
    get_averaged_predictions(
        predictions_paths=[
            './submissions/baseline.csv', 
            './submissions/submission_stack_1741599845.csv', 
            './submissions/submission_lasso_1741599693.csv'
        ],
        output_path='./submissions/averaged_submission.csv',
        weights=[0.9, 1.2, 1]
    )