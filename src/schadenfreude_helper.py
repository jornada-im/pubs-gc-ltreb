import pandas as pd
import numpy as np
from scipy import stats


def ppt_trt_rel_means(df, ci=0.95):
    """
    Take a long-term dataframe, take the mean + confidence intervals of all plots per
    treatment, and calculate relative biomass for each treatment.
    """
    # Group by treatment and calculate means & stddev
    gby = df.groupby(['year','ppt_trt'])
    means = gby.mean(numeric_only=True).reset_index()
    stds = gby.std(ddof=1, numeric_only=True).reset_index()
    # Observation counts and standard errors
    n = gby.count().reset_index().iloc[:,4::]
    se = stds.iloc[:,2::] / np.sqrt(n)
    # t-statistic and margin of error
    alpha = 1 - ci
    t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
    margin_error = t_critical * se
    # Calculate confidence intervals
    ci_lower = means.iloc[:,0:2].join(means.iloc[:,2::] - margin_error)
    ci_upper = means.iloc[:,0:2].join(means.iloc[:,2::] + margin_error)
    
    # Calculate relative biomass for the 3 treatments
    # Ambient = 1.0, 0.2 & 1.8 are the difference from ambient
    for stat in [means,]:
        stat['rbiomass_grass'] = np.nan
        stat['rbiomass_shrub'] = np.nan
        stat['rbiomass_total'] = np.nan
        test = stat.ppt_trt=='0.2'
        stat.loc[test, 'rbiomass_grass'] = stat['biomass_grass'].diff(-1)[test]
        stat.loc[test, 'rbiomass_shrub'] = stat['biomass_shrub'].diff(-1)[test]
        stat.loc[test, 'rbiomass_total'] = stat['biomass_total'].diff(-1)[test]
        test = stat.ppt_trt=='1.8'
        stat.loc[test, 'rbiomass_grass'] = stat['biomass_grass'].diff()[test]
        stat.loc[test, 'rbiomass_shrub'] = stat['biomass_shrub'].diff()[test]
        stat.loc[test, 'rbiomass_total'] = stat['biomass_total'].diff()[test]
        test = stat.ppt_trt=='1.0'
        stat.loc[test, 'rbiomass_grass'] = 0
        stat.loc[test, 'rbiomass_shrub'] = 0
        stat.loc[test, 'rbiomass_total'] = 0
    
    return(means, ci_upper, ci_lower)