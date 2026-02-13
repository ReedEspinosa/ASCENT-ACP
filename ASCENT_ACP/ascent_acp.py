"""ASCENT-ACP specific function for merging ICARTT files."""

import icartt_read_and_merge as ict


def run_ascent_acp_merge(
    icartt_directory="/Users/wrespino/Downloads/ACTIVATE_TEST",
    pickle_directory=None,
    pickle_filename='merged_LAS-SMPS-Optical_outfile',
    prefix_instr_name=True,
    mode_input='Merge_Beside',
    master_timeline=['2020-02-14 00:00:00', '2020-02-16 00:00:00', 1]
):
    """
    Run the ASCENT-ACP ICARTT merge operation.

    This function duplicates the uncommented Merge_Beside example from examples.py.

    Parameters
    ----------
    icartt_directory : str, optional
        Directory containing ICARTT files to merge.
        Default: "/Users/wrespino/Downloads/ACTIVATE_TEST"
    pickle_directory : str, optional
        Directory where merged output pickle files will be saved.
        If None, no files will be saved. If empty string (''),
        output will be stored in the input icartt_directory.
        Default: None
    pickle_filename : str, optional
        Name of the output pickle file (without extension).
        Default: 'merged_LAS-SMPS-Optical_outfile'
    prefix_instr_name : bool, optional
        Whether to prefix instrument names to column names.
        Default: True
    mode_input : str, optional
        Merge mode ('Merge_Beside', 'Stack_On_Top', or 'Load_Pickle').
        Default: 'Merge_Beside'
    master_timeline : list, optional
        Master timeline specification [start_date, end_date, time_step_seconds].
        Required for 'Merge_Beside' mode, optional for others.
        Default: ['2020-02-14 00:00:00', '2020-02-16 00:00:00', 5]

    Returns
    -------
    df : pandas.DataFrame
        Merged DataFrame containing the ICARTT data
    meta : dict
        Metadata dictionary with information about the merge operation
    """
    df, meta = ict.icartt_merger(
        icartt_directory=icartt_directory,
        mode_input=mode_input,
        master_timeline=master_timeline,
        pickle_directory=pickle_directory,
        pickle_filename=pickle_filename,
        prefix_instr_name=prefix_instr_name
    )

    return df, meta


if __name__ == "__main__":
    # Run the merge operation with default parameters
    df, meta = run_ascent_acp_merge()
    print("Merge completed successfully!")
    print(f"DataFrame shape: {df.shape}")
    print(f"Metadata keys: {list(meta.keys())}")


