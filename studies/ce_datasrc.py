class DataSrcExplorer(object):
    @classmethod
    def profiling_dataframe_to_html(cls, data, html_filename):
        generate_profiling = False
        if not generate_profiling:
            return None
        from pandas_profiling import ProfileReport
        profile = ProfileReport(data, title="Pandas Profiling Report")
        profile.to_file(html_filename)
        return profile

    @classmethod
    def explore_single_df(cls, df_sd):
        print(df_sd)
        desp = df_sd.describe()
        print(desp)
        print(desp.shape)
        unique_vals = []
        for i in range(len(df_sd)):
            if df_sd[i] not in unique_vals:
                unique_vals.append(df_sd[i])
        unique_vals = sorted(unique_vals)
        return desp, unique_vals
