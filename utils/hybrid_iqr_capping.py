def hybrid_iqr_capping(df, cols, factor=1.5):
    caps = {}
    df = df.copy()

    for col in cols:
        if col not in df.columns:
            continue
        if df[col].dtype.kind not in "bifc":
            continue

        series = df[col].dropna()

        # IQR
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        iqr_lower = Q1 - factor * IQR
        iqr_upper = Q3 + factor * IQR

        # Percentiles
        p1 = series.quantile(0.01)
        p99 = series.quantile(0.99)

        # Hybrid bounds
        lower = min(iqr_lower, p1)
        upper = max(iqr_upper, p99)

        df[col] = df[col].clip(lower, upper)
        caps[col] = {"lower": lower, "upper": upper}

    return df, caps
