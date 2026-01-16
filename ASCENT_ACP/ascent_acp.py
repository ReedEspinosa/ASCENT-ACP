"""ASCENT-ACP specific function for merging ICARTT files."""

import icartt_read_and_merge as ict


def run_ascent_acp_merge(
    data_directory="/Users/wrespino/Downloads/ACTIVATE_TEST",
    output_directory="/Users/wrespino/Downloads/ACTIVATE_TEST",
    output_filename='merged_LAS-SMPS-Optical_outfile',
    prefix_instr_name=True,
    mode_input='Merge_Beside',
    master_timeline=['2020-02-14 00:00:00', '2020-02-16 00:00:00', 5]
):
    """
    Run the ASCENT-ACP ICARTT merge operation.
    
    This function duplicates the uncommented Merge_Beside example from examples.py.
    
    Parameters
    ----------
    data_directory : str, optional
        Directory containing ICARTT files to merge. 
        Default: "/Users/wrespino/Downloads/ACTIVATE_TEST"
    output_directory : str, optional
        Directory where merged output will be saved.
        Default: "/Users/wrespino/Downloads/ACTIVATE_TEST"
    output_filename : str, optional
        Name of the output file (without extension).
        Default: 'merged_LAS-SMPS-Optical_outfile'
    prefix_instr_name : bool, optional
        Whether to prefix instrument names to column names.
        Default: True
    mode_input : str, optional
        Merge mode ('Merge_Beside' or 'Stack_On_Top').
        Default: 'Merge_Beside'
    master_timeline : list, optional
        Master timeline specification [start_date, end_date, time_step_seconds].
        Default: ['2020-02-14 00:00:00', '2020-02-16 00:00:00', 5]
    
    Returns
    -------
    df : pandas.DataFrame
        Merged DataFrame containing the ICARTT data
    meta : dict
        Metadata dictionary with information about the merge operation
    """
    df, meta = ict.icartt_merger(
        data_directory, 
        mode_input, 
        master_timeline,
        output_directory, 
        output_filename,
        prefix_instr_name
    )
    
    return df, meta


if __name__ == "__main__":
    # Run the merge operation with default parameters
    df, meta = run_ascent_acp_merge()
    print("Merge completed successfully!")
    print(f"DataFrame shape: {df.shape}")
    print(f"Metadata keys: {list(meta.keys())}")

