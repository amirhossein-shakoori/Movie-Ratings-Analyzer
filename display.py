def display_simple(df_obj, n_rows=10):
    """
    Simple display function using pandas built-in options
    """
    with pd.option_context('display.max_rows', n_rows,
                          'display.max_columns', None,
                          'display.width', 1000,
                          'display.max_colwidth', 40):
        if isinstance(df_obj, pd.DataFrame):
            print(df_obj.head(n_rows))
            print(f"\nShape: {df_obj.shape}")
        elif isinstance(df_obj, pd.Series):
            print(df_obj.head(n_rows).to_frame().T)
        else:
            print(df_obj)