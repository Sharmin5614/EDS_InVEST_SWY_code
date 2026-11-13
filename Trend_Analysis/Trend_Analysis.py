#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mann–Kendall and Sen's Slope analyses 


def run_station_trends():
    """Analyze the 12 individual streamgages and export statistics and plots."""
    """Annual Mann-Kendall and SenSlope trend analysis for 12 streamgages.
    Install: pip install pandas openpyxl numpy scipy matplotlib
    Run: python mann_kendall_12_streamgages.py
    """
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from scipy.stats import kendalltau, theilslopes, norm
    import matplotlib.pyplot as plt

    INPUT = Path(r'C:\Users\ASUS\OneDrive\Desktop\MS DESSERTATION\1st_Paper_Journal_of_EDS\Additional_Work_of Reviewer\Streamflow_Trend_Results_1985_2023.xlsx')
    if not INPUT.exists():
        INPUT = Path(__file__).with_name('Streamflow_Trend_Results_1985_2023.xlsx')
    OUT = INPUT.parent / 'Streamflow_MK_Sen_Results'
    OUT.mkdir(exist_ok=True)
    raw = pd.read_excel(INPUT, sheet_name='Annual Streamflow', header=None)
    records = []
    for j in range(0, 24, 2):
        station = str(raw.iat[0, j]).strip()
        years = pd.to_numeric(raw.iloc[2:, j], errors='coerce')
        flow = pd.to_numeric(raw.iloc[2:, j+1], errors='coerce')
        for year, value in zip(years, flow):
            if pd.notna(year) and pd.notna(value):
                records.append((station, int(year), float(value)))
    long = pd.DataFrame(records, columns=['Station', 'Year', 'Annual_mean_streamflow'])
    if long.duplicated(['Station','Year']).any():
        raise ValueError('Duplicate station/year records detected')
    wide = long.pivot(index='Year', columns='Station', values='Annual_mean_streamflow').sort_index()
    expected = set(range(1985, 2024))
    results = []
    for station in wide.columns:
        s = wide[station].dropna().sort_index()
        years = s.index.to_numpy(dtype=float)
        vals = s.to_numpy(dtype=float)
        missing = sorted(expected - set(s.index))
        if len(s) < 3:
            raise ValueError(f'Insufficient observations for {station}')
        tau, p = kendalltau(years, vals, method='auto')
        slope, intercept, lo, hi = theilslopes(vals, years, alpha=0.95)
        # S statistic and tie-adjusted variance, standard two-sided MK normal approximation
        differences = vals[np.newaxis, :] - vals[:, np.newaxis]
        S = int(np.sign(differences[np.triu_indices(len(vals), 1)]).sum())
        _, counts = np.unique(vals, return_counts=True)
        n = len(vals)
        varS = (n*(n-1)*(2*n+5) - sum(t*(t-1)*(2*t+5) for t in counts if t>1))/18
        z = (S-1)/np.sqrt(varS) if S>0 else ((S+1)/np.sqrt(varS) if S<0 else 0.)
        p_mk = 2*norm.sf(abs(z))
        trend = ('Increasing' if slope > 0 else 'Decreasing' if slope < 0 else 'No change') if p_mk < 0.05 else 'No significant trend'
        lag1 = np.corrcoef(vals[:-1], vals[1:])[0,1] if n > 3 else np.nan
        results.append(dict(Station=station, N=n, Start_year=int(years.min()), End_year=int(years.max()), Missing_years=', '.join(map(str, missing)), MK_S=S, MK_Z=z, Kendall_tau=tau, MK_p_value=p_mk, Sen_slope_per_year=slope, Sen_intercept=intercept, Sen_CI95_lower=lo, Sen_CI95_upper=hi, Lag1_autocorrelation=lag1, Trend_at_0_05=trend))
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(years, vals, marker='o', markersize=3, linewidth=1, label='Annual mean streamflow')
        ax.plot(years, intercept + slope*years, linestyle='--', linewidth=2, label="Sen's slope")
        ax.set(xlabel='Year', ylabel='Annual mean streamflow (verify units)', title=f'{station} | MK p={p_mk:.3g} | Sen slope={slope:.3g}/year')
        ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
        fig.savefig(OUT / f'{station}_trend.png', dpi=200)
        plt.close(fig)
    summary = pd.DataFrame(results).sort_values('Station')
    with pd.ExcelWriter(OUT / 'MK_Sen_12_Stations.xlsx', engine='openpyxl') as writer:
        summary.to_excel(writer, sheet_name='Trend Results', index=False)
        wide.reset_index().to_excel(writer, sheet_name='Aligned Annual Data', index=False)
        long.to_excel(writer, sheet_name='Long Format', index=False)
    print(summary[['Station','N','Kendall_tau','MK_p_value','Sen_slope_per_year','Trend_at_0_05']].to_string(index=False))
    print('\nResults saved to:', OUT)
    print('Caution: standard MK assumes independent annual observations; inspect lag-1 correlation.')



def run_precipitation_trend():
    """Analyze annual precipitation"""
    # ============================================================
    # ANNUAL PRECIPITATION TREND ANALYSIS (1985–2023)
    # ============================================================

    # ============================================================
    # 1. IMPORT LIBRARIES
    # ============================================================

    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from scipy.stats import norm, theilslopes, kendalltau
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter


    # ============================================================
    # 2. INPUT FILE AND OUTPUT DIRECTORY
    # ============================================================

    file_path = Path(
        r'C:\Users\ASUS\OneDrive\Desktop\MS DESSERTATION'
        r'\1st_Paper_Journal_of_EDS\Additional_Work_of Reviewer'
        r'\Annual_Precipitation_1985_2023_mm.xlsx'
    )

    output_dir = file_path.parent / 'Precipitation_MK_Sen_Trend'

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f'Excel file not found:\n{file_path}'
        )


    # ============================================================
    # 3. READ PRECIPITATION DATA
    # ============================================================

    df = pd.read_excel(
        file_path,
        sheet_name=0
    )

    df.columns = df.columns.astype(str).str.strip()

    required = {'Year', 'Precipitation_mm'}

    if not required.issubset(df.columns):
        raise ValueError(
            f'Expected columns {required}; '
            f'found {list(df.columns)}'
        )

    df = df[['Year', 'Precipitation_mm']].copy()

    df['Year'] = pd.to_numeric(
        df['Year'],
        errors='coerce'
    )

    df['Precipitation_mm'] = pd.to_numeric(
        df['Precipitation_mm'],
        errors='coerce'
    )

    if df.isna().any().any() or df['Year'].duplicated().any():
        raise ValueError(
            'Missing or duplicate year/precipitation observations'
        )

    df = df.sort_values('Year')

    if df['Year'].tolist() != list(range(1985, 2024)):
        raise ValueError(
            'Expected all calendar years 1985–2023'
        )

    x = df['Year'].to_numpy(dtype=float)
    y = df['Precipitation_mm'].to_numpy(dtype=float)

    n = len(y)


    # ============================================================
    # 4. ORIGINAL MANN–KENDALL TEST
    # METHODOLOGY UNCHANGED
    # ============================================================

    S = sum(
        int(np.sign(y[j] - y[i]))
        for i in range(n - 1)
        for j in range(i + 1, n)
    )

    _, counts = np.unique(
        y,
        return_counts=True
    )

    var_s = (
        n * (n - 1) * (2 * n + 5)
        - np.sum(
            counts * (counts - 1) * (2 * counts + 5)
        )
    ) / 18

    z = (
        (S - np.sign(S)) / np.sqrt(var_s)
        if var_s > 0
        else 0.0
    )

    p = (
        2 * norm.sf(abs(z))
        if var_s > 0
        else 1.0
    )

    tau, _ = kendalltau(x, y)


    # ============================================================
    # 5. ORIGINAL SEN'S SLOPE ESTIMATION
    # METHODOLOGY UNCHANGED
    # ============================================================

    slope, intercept, lo, hi = theilslopes(
        y,
        x,
        alpha=0.95
    )

    trend = (
        (
            'Significant decrease'
            if slope < 0
            else 'Significant increase'
        )
        if p < 0.05
        else 'Not statistically significant'
    )

    trend_line = intercept + slope * x


    # ============================================================
    # 6. FIGURE FORMATTING
    # EXACTLY MATCHING THE STREAMFLOW GRAPH
    # ============================================================

    plt.rcParams.update({

        'font.family': 'Times New Roman',

        'font.serif': ['Times New Roman'],

        'font.size': 16,

        'axes.labelsize': 19,

        'axes.labelweight': 'bold',

        'xtick.labelsize': 15,

        'ytick.labelsize': 15,

        'legend.fontsize': 16,

        'axes.linewidth': 1.4,

        'axes.spines.top': False,

        'axes.spines.right': False,

        'pdf.fonttype': 42,

        'ps.fonttype': 42,

        'svg.fonttype': 'none',

        'savefig.facecolor': 'white'
    })


    # ============================================================
    # 7. PROFESSIONAL COLOR PALETTE
    # SAME AS STREAMFLOW GRAPH
    # ============================================================

    BLUE = '#2166AC'
    RED = '#B2182B'
    DARK = '#202020'
    GRID = '#DCE1E6'


    # ============================================================
    # 8. CREATE FIGURE
    # ============================================================

    fig, ax = plt.subplots(
        figsize=(14, 7)
    )

    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')


    # ============================================================
    # 9. ANNUAL PRECIPITATION LINE
    # ============================================================

    ax.plot(
        x,
        y,

        color=BLUE,

        linewidth=2.4,

        linestyle='-',

        marker='o',

        markersize=5.5,

        markerfacecolor=BLUE,

        markeredgecolor='white',

        markeredgewidth=0.8,

        solid_capstyle='round',

        zorder=3,

        label='Annual precipitation'
    )


    # ============================================================
    # 10. SEN'S SLOPE TREND LINE
    # ============================================================

    ax.plot(
        x,
        trend_line,

        color=RED,

        linewidth=2.8,

        linestyle='--',

        dash_capstyle='round',

        zorder=4,

        label="Sen's slope trend"
    )


    # ============================================================
    # 11. BOLD AXIS LABELS
    # ============================================================

    ax.set_xlabel(
        'Year',

        fontsize=19,

        fontweight='bold',

        fontfamily='Times New Roman',

        color=DARK,

        labelpad=14
    )

    ax.set_ylabel(
        'Annual precipitation (mm)',

        fontsize=19,

        fontweight='bold',

        fontfamily='Times New Roman',

        color=DARK,

        labelpad=17
    )


    # ============================================================
    # 12. AXIS LIMITS AND TICKS
    # ============================================================
    # X-axis label: Times New Roman, bold
    ax.set_xlabel(
        'Year',
        fontsize=19,
        fontweight='bold',
        fontfamily='Times New Roman',
        color=DARK,
        labelpad=14
    )

    # X-axis tick labels: Times New Roman, bold
    ax.tick_params(
        axis='x',
        labelsize=15,
        width=1.3,
        length=7,
        colors=DARK,
        pad=8
    )

    for tick in ax.get_xticklabels():
        tick.set_fontfamily('Times New Roman')
        tick.set_fontweight('bold')

    # Y-axis ticks: 400, 500, 600, ... 1100
    ax.set_ylim(380, 1160)
    ax.set_yticks(np.arange(400, 1101, 100))

    # Times New Roman tick labels
    ax.tick_params(
        axis='y',
        labelsize=15,
        width=1.3,
        length=7,
        colors=DARK,
        pad=8
    )

    for tick in ax.get_yticklabels():
        tick.set_fontfamily('Times New Roman')
        tick.set_fontweight('semibold')

    # ============================================================
    # 13. GRIDLINES
    # ============================================================

    ax.grid(
        axis='y',

        color=GRID,

        linestyle='--',

        linewidth=0.85,

        alpha=0.75,

        zorder=0
    )

    ax.grid(
        axis='x',
        visible=False
    )

    ax.spines['left'].set_color(DARK)

    ax.spines['bottom'].set_color(DARK)


    # ============================================================
    # 14. LEGEND
    # SAME SIZE AND FORMATTING AS STREAMFLOW GRAPH
    # ============================================================

    legend = ax.legend(

        loc='upper right',

        bbox_to_anchor=(0.99, 0.99),

        fontsize=16,

        frameon=True,

        fancybox=False,

        facecolor='white',

        edgecolor='#BFC5CC',

        framealpha=0.98,

        borderpad=0.85,

        labelspacing=0.7,

        handlelength=3.0,

        handletextpad=0.9,

        markerscale=1.4
    )

    legend.get_frame().set_linewidth(0.9)

    for item in legend.get_texts():

        item.set_fontfamily('Times New Roman')

        item.set_color(DARK)


    # ============================================================
    # 15. STATISTICAL ANNOTATION
    # SAME FORMAT AND POSITION AS STREAMFLOW GRAPH
    # ============================================================

    stat_text = (
        f'Mann–Kendall p = {p:.4g}\n'
        f"Kendall's τ = {tau:.3f}\n"
        f"Sen's slope = {slope:.2f} mm/year\n"
        f'Trend: {trend}'
    )

    ax.text(

        0.025,
        0.97,

        stat_text,

        transform=ax.transAxes,

        fontsize=14.5,

        fontfamily='Times New Roman',

        color=DARK,

        va='top',

        ha='left',

        linespacing=1.5,

        bbox=dict(

            boxstyle='round,pad=0.65',

            facecolor='white',

            edgecolor='#C6CDD5',

            linewidth=0.9,

            alpha=0.97
        ),

        zorder=6
    )


    # ============================================================
    # 16. FINAL LAYOUT
    # ============================================================

    # No title, matching the streamflow graph

    fig.tight_layout(
        pad=2.0
    )


    # ============================================================
    # 17. SAVE PUBLICATION-QUALITY FIGURES
    # ============================================================

    png_path = (
        output_dir /
        'Annual_Precipitation_MK_Sen_1985_2023_Publication.png'
    )

    pdf_path = (
        output_dir /
        'Annual_Precipitation_MK_Sen_1985_2023_Publication.pdf'
    )

    svg_path = (
        output_dir /
        'Annual_Precipitation_MK_Sen_1985_2023_Publication.svg'
    )

    fig.savefig(
        png_path,

        dpi=600,

        bbox_inches='tight',

        pad_inches=0.15
    )

    fig.savefig(
        pdf_path,

        bbox_inches='tight',

        pad_inches=0.15
    )

    fig.savefig(
        svg_path,

        bbox_inches='tight',

        pad_inches=0.15
    )

    plt.show()


    # ============================================================
    # 18. PRINT STATISTICAL RESULTS
    # ============================================================

    print('\n========== PRECIPITATION TREND RESULTS ==========')

    print('Period: 1985–2023')

    print(f'Number of years: {n}')

    print('\n--- Mann–Kendall Test ---')

    print(f'MK S statistic: {S}')

    print(f'MK Z statistic: {z:.6f}')

    print(f'MK p-value: {p:.8f}')

    print(f"Kendall's tau: {tau:.6f}")

    print('\n--- Sen Slope ---')

    print(f"Sen's slope: {slope:.6f} mm/year")

    print(
        f'95% confidence interval: '
        f'[{lo:.6f}, {hi:.6f}] mm/year'
    )

    print(f'\nTrend classification: {trend}')

    print('\nSaved figures:')

    print(png_path)

    print(pdf_path)

    print(svg_path)

    print('\n===============================================')



def run_summed_streamflow_figure():
    """Plot summed discharge using saved statistics (no trend tests rerun)."""
    # Upper Red River Basin, Oklahoma
    #
    # Visualization only:
    # - No Mann–Kendall tests
    # - No Hamed–Rao tests
    # - No Sen's slope recalculation
    # - Uses previously saved statistical results
    # ============================================================

    # ============================================================
    # 1. IMPORT LIBRARIES
    # ============================================================

    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from matplotlib.ticker import (
        MaxNLocator,
        StrMethodFormatter
    )

    # ============================================================
    # 2. FILE PATHS
    # ============================================================

    file_path = Path(
        r'C:\Users\ASUS\OneDrive\Desktop\MS DESSERTATION'
        r'\1st_Paper_Journal_of_EDS\Additional_Work_of Reviewer'
        r'\Streamflow_Trend_Results_1985_2023.xlsx'
    )

    output_dir = file_path.parent / 'Summed_Streamflow_Trend'

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    statistics_file = (
        output_dir / 'Summed_Streamflow_Trend_Statistics.csv'
    )

    # Check required files
    if not file_path.exists():
        raise FileNotFoundError(
            f'Excel file not found:\n{file_path}'
        )

    if not statistics_file.exists():
        raise FileNotFoundError(
            'Previously saved statistics file not found:\n'
            f'{statistics_file}\n'
            'Run the original analysis once or provide '
            'the previously calculated results.'
        )

    # ============================================================
    # 3. READ ANNUAL STREAMFLOW DATA
    # ============================================================

    raw = pd.read_excel(
        file_path,
        sheet_name='Annual Streamflow',
        header=None
    )

    station_series = []

    for col in range(0, 24, 2):

        station = str(raw.iat[0, col]).strip()

        df = pd.DataFrame({
            'Year': pd.to_numeric(
                raw.iloc[2:, col],
                errors='coerce'
            ),
            station: pd.to_numeric(
                raw.iloc[2:, col + 1],
                errors='coerce'
            )
        }).dropna(subset=['Year'])

        df['Year'] = df['Year'].astype(int)

        if df['Year'].duplicated().any():
            raise ValueError(
                f'Duplicate years detected for {station}'
            )

        station_series.append(
            df.set_index('Year')
        )

    combined = pd.concat(
        station_series,
        axis=1
    ).sort_index()

    # Validate input data
    if combined.shape[1] != 12:
        raise ValueError(
            'Expected 12 streamgauges.'
        )

    if combined.columns.duplicated().any():
        raise ValueError(
            'Duplicate station names detected.'
        )

    if list(combined.index) != list(range(1985, 2024)):
        raise ValueError(
            'Expected years 1985 through 2023.'
        )

    if combined.isna().any().any():
        raise ValueError(
            'Missing streamflow observations detected.'
        )

    # ============================================================
    # 4. PREPARE OBSERVED STREAMFLOW
    # ============================================================

    # Sum annual mean discharge across the 12 gauges.
    # This is a sum of gauge measurements, not necessarily
    # the total watershed discharge.

    annual = combined.sum(axis=1)

    x = annual.index.to_numpy(dtype=float)
    y = annual.to_numpy(dtype=float)

    # ============================================================
    # 5. LOAD PREVIOUSLY CALCULATED STATISTICS
    # ============================================================

    stats = pd.read_csv(statistics_file)

    if len(stats) != 1:
        raise ValueError(
            'Expected exactly one row of trend statistics.'
        )

    required_columns = [
        'MK_p_Original',
        'MK_p_Hamed_Rao',
        'Kendall_tau',
        'Sen_slope_cfs_per_year',
        'Adjusted_Trend'
    ]

    missing_columns = [
        col for col in required_columns
        if col not in stats.columns
    ]

    if missing_columns:
        raise ValueError(
            f'Missing statistics columns: {missing_columns}'
        )

    p_original = float(
        stats.loc[0, 'MK_p_Original']
    )

    p_adjusted = float(
        stats.loc[0, 'MK_p_Hamed_Rao']
    )

    kendall_tau = float(
        stats.loc[0, 'Kendall_tau']
    )

    slope = float(
        stats.loc[0, 'Sen_slope_cfs_per_year']
    )

    trend_adjusted = str(
        stats.loc[0, 'Adjusted_Trend']
    )

    # Reconstruct the line intercept using the saved slope.
    # This does not recompute Sen's slope or run a test.
    intercept = float(
        np.median(y - slope * x)
    )

    trend_line = intercept + slope * x

    # ============================================================
    # 6. PUBLICATION FIGURE SETTINGS
    # ============================================================

    plt.rcParams.update({

        'font.family': 'Times New Roman',

        'font.size': 16,

        'axes.labelsize': 19,
        'axes.labelweight': 'bold',

        'xtick.labelsize': 15,
        'ytick.labelsize': 15,

        'legend.fontsize': 16,

        'axes.linewidth': 1.4,

        'axes.spines.top': False,
        'axes.spines.right': False,

        'pdf.fonttype': 42,
        'ps.fonttype': 42,

        'savefig.facecolor': 'white'
    })

    # Scientific, high-contrast color palette
    BLUE = '#2166AC'
    RED = '#B2182B'
    DARK = '#202020'
    GRID = '#DCE1E6'

    # ============================================================
    # 7. CREATE FIGURE
    # ============================================================

    fig, ax = plt.subplots(
        figsize=(14, 7)
    )

    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # ============================================================
    # 8. OBSERVED STREAMFLOW
    # ============================================================

    ax.plot(
        x,
        y,

        color=BLUE,

        linewidth=2.4,

        linestyle='-',

        marker='o',

        markersize=5.5,

        markerfacecolor=BLUE,

        markeredgecolor='white',

        markeredgewidth=0.8,

        solid_capstyle='round',

        zorder=3,

        label='Observed streamflow'
    )

    # ============================================================
    # 9. SEN'S SLOPE TREND LINE
    # ============================================================

    ax.plot(
        x,
        trend_line,

        color=RED,

        linewidth=2.8,

        linestyle='--',

        dash_capstyle='round',

        zorder=4,

        label="Sen's slope trend"
    )

    # ============================================================
    # 10. AXIS LABELS
    # ============================================================

    ax.set_xlabel(
        'Year',

        fontsize=19,

        fontweight='bold',

        fontfamily='Times New Roman',

        color=DARK,

        labelpad=14
    )

    ax.set_ylabel(
        'Summed annual mean streamflow (cfs)',

        fontsize=19,

        fontweight='bold',

        fontfamily='Times New Roman',

        color=DARK,

        labelpad=17
    )

    # ============================================================
    # 11. AXIS LIMITS AND TICKS
    # ============================================================

    ax.set_xlim(
        1984,
        2024
    )

    ax.set_xticks([
        1985,
        1990,
        1995,
        2000,
        2005,
        2010,
        2015,
        2020,
        2023
    ])

    ax.set_ylim(
        bottom=0,
        top=max(y.max(), trend_line.max()) * 1.17
    )

    ax.yaxis.set_major_locator(
        MaxNLocator(nbins=6)
    )

    ax.yaxis.set_major_formatter(
        StrMethodFormatter('{x:,.0f}')
    )

    ax.tick_params(
        axis='both',

        which='major',

        direction='out',

        length=7,

        width=1.3,

        labelsize=15,

        colors=DARK,

        pad=8
    )

    for tick in ax.get_xticklabels():
        tick.set_fontfamily('Times New Roman')
        tick.set_fontweight('semibold')

    for tick in ax.get_yticklabels():
        tick.set_fontfamily('Times New Roman')
        tick.set_fontweight('semibold')

    # ============================================================
    # 12. GRIDLINES
    # ============================================================

    ax.grid(
        axis='y',

        color=GRID,

        linestyle='--',

        linewidth=0.85,

        alpha=0.75,

        zorder=0
    )

    ax.grid(
        axis='x',
        visible=False
    )

    ax.spines['left'].set_color(DARK)
    ax.spines['bottom'].set_color(DARK)

    # ============================================================
    # 13. LEGEND
    # ============================================================

    legend = ax.legend(

        loc='upper right',

        bbox_to_anchor=(0.99, 0.99),

        fontsize=16,

        frameon=True,

        fancybox=False,

        facecolor='white',

        edgecolor='#BFC5CC',

        framealpha=0.98,

        borderpad=0.85,

        labelspacing=0.7,

        handlelength=3.0,

        handletextpad=0.9,

        markerscale=1.4
    )

    legend.get_frame().set_linewidth(0.9)

    for item in legend.get_texts():

        item.set_fontfamily('Times New Roman')
        item.set_color(DARK)

    # ============================================================
    # 14. STATISTICAL ANNOTATION
    # ============================================================

    stat_text = (
        f'Mann–Kendall p = {p_original:.4g}\n'
        f'Hamed–Rao adjusted p = {p_adjusted:.4g}\n'
        f"Kendall's τ = {kendall_tau:.3f}\n"
        f"Sen's slope = {slope:.2f} cfs/year\n"
        f'Trend: {trend_adjusted}'
    )

    ax.text(

        0.025,
        0.97,

        stat_text,

        transform=ax.transAxes,

        fontsize=14.5,

        fontfamily='Times New Roman',

        color=DARK,

        va='top',

        ha='left',

        linespacing=1.5,

        bbox=dict(

            boxstyle='round,pad=0.65',

            facecolor='white',

            edgecolor='#C6CDD5',

            linewidth=0.9,

            alpha=0.97
        ),

        zorder=6
    )

    # ============================================================
    # 15. FINAL LAYOUT
    # ============================================================

    # No title, as requested.

    fig.tight_layout(
        pad=2.0
    )

    # ============================================================
    # 16. SAVE FIGURE
    # ============================================================

    png_path = (
        output_dir /
        'Summed_Streamflow_1985_2023_Publication.png'
    )

    pdf_path = (
        output_dir /
        'Summed_Streamflow_1985_2023_Publication.pdf'
    )

    svg_path = (
        output_dir /
        'Summed_Streamflow_1985_2023_Publication.svg'
    )

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches='tight',
        pad_inches=0.15
    )

    fig.savefig(
        pdf_path,
        bbox_inches='tight',
        pad_inches=0.15
    )

    fig.savefig(
        svg_path,
        bbox_inches='tight',
        pad_inches=0.15
    )

    plt.show()

    # ============================================================
    # 17. CONFIRM OUTPUT
    # ============================================================

    print('\n========== FIGURE GENERATED ==========')

    print('Period: 1985–2023')
    print('Number of gauges:', combined.shape[1])
    print('Number of years:', len(x))

    print('\nPreviously saved statistics:')
    print(f'Mann–Kendall p: {p_original:.8f}')
    print(f'Hamed–Rao adjusted p: {p_adjusted:.8f}')
    print(f"Kendall's tau: {kendall_tau:.6f}")
    print(f"Sen's slope: {slope:.6f} cfs/year")
    print(f'Trend: {trend_adjusted}')

    print('\nFiles saved:')
    print(png_path)
    print(pdf_path)
    print(svg_path)

    print('\nNo statistical tests were rerun.')
    print('======================================')



def main():
    """Run a selected research workflow, or all three in sequence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis", choices=("all", "stations", "precipitation", "summed"),
        default="all", help="Analysis section to execute (default: all)."
    )
    args = parser.parse_args()

    if args.analysis in ("all", "stations"):
        print("\n========== 12 STREAMGAGE TREND ANALYSIS ==========\n")
        run_station_trends()
    if args.analysis in ("all", "precipitation"):
        print("\n========== ANNUAL PRECIPITATION TREND ==========\n")
        run_precipitation_trend()
    if args.analysis in ("all", "summed"):
        print("\n========== SUMMED STREAMFLOW FIGURE ==========\n")
        run_summed_streamflow_figure()


if __name__ == "__main__":
    main()
