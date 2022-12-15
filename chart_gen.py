import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text
from matplotlib.patches import Rectangle

ds = pd.read_csv('bjcp_stats.csv')

srm_colors = {
    2: (.98, .86, .47),
    3: (.98, .8, .35),
    5: (.97, .7, .15),
    7: (.93, .62, .03),
    10: (.85, .49, .03),
    14: (.75, .36, .02),
    17: (.67, .28, .02),
    19: (.61, .22, .02),
    22: (.55, .18, .02),
    30: (.37, .1, .02),
    35: (.27, .04, .02),
    40: (.15, .05, .02),
}


def interpolate_beer_color(srm):
    srm_refs = list(srm_colors.keys())
    #
    if srm <= 0:
        return np.array([1., 1., 1.])
    #
    first_srm = srm_refs[0]
    if srm <= first_srm:
        tt = srm / first_srm
        first_color = np.array(srm_colors[first_srm])
        return tt * first_color + (1-tt) * np.array([1., 1., 1.])
    last_srm = srm_refs[-1]
    #
    if srm > last_srm:
        exp_dist = (srm - last_srm) / last_srm
        tt = np.exp(-exp_dist)
        return tt * np.array(srm_colors[last_srm])
    #
    n_srm_samples = len(srm_refs)
    for foo in range(n_srm_samples-1):
        next = srm_refs[foo+1]
        if srm <= next:
            prev = srm_refs[foo]
            tt = (next - srm) / (next - prev)
            color = np.array(srm_colors[prev]) * tt + (1 - tt) * np.array(srm_colors[next])
            return color
    return np.array([0., 0., 0.])

def generate_ibu_v_srm_chart():
    xcol = 'ibu_mid'
    ycol = 'srm_mid'
    xx = ds[xcol]
    yy = ds[ycol]
    styles = ds['Style']

    fig, ax = plt.subplots(figsize=(16, 16))
    srm_colors = [interpolate_beer_color(srm) for srm in yy]
    ax.scatter(xx, yy, color=srm_colors)
    plt.xlim((0,90))
    plt.ylim((1, 40))
    plt.xlabel('Bitterness (IBU)')
    plt.ylabel('Color (SRM)')
    plt.title('Color vs Bitterness of BJCP Styles')
    #
    texts = [plt.text(xx[i], yy[i], styles[i], ha='center', va='center') for i in range(len(xx))]
    adjust_text(texts)
    for srm in range(1, 40):
        ax.add_patch(Rectangle((0, srm), 1, 1, color=interpolate_beer_color(srm)))
    plt.savefig('charts/color_v_bitterness.svg')
    plt.show()

def generate_grav_v_abv_chart():
    abvs = ds['abv_mid']
    abv_errs = ds['abv_delta']
    gravs = ds['final_gravity_mid']
    grav_errs = ds['final_gravity_delta']
    srms = ds['srm_mid']
    styles = ds['Style']
    srm_colors = [interpolate_beer_color(srm) for srm in srms]

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.errorbar(gravs, abvs, fmt='o', xerr=grav_errs, yerr=abv_errs, ecolor='lightgray')
    ax.scatter(gravs, abvs, color = srm_colors)
    #
    texts = [plt.text(gravs[i], abvs[i], styles[i], ha='center', va='center') for i in range(len(gravs))]
    adjust_text(texts)

    plt.xlabel('Final Gravity')
    plt.ylabel('ABV')
    plt.title('ABV vs Gravity of BJCP Styles')
    plt.savefig('charts/abv_v_grav.svg')
    plt.show()

def generate_grav_v_srm_chart():
    xcol = 'final_gravity_mid'
    ycol = 'srm_mid'
    xx = ds[xcol]
    yy = ds[ycol]
    styles = ds['Style']

    fig, ax = plt.subplots(figsize=(16, 16))
    srm_colors = [interpolate_beer_color(srm) for srm in yy]
    ax.scatter(xx, yy, color=srm_colors)
    plt.ylim((1, 40))
    plt.xlim((1,1.03))
    plt.xlabel('Final Specific Gravity')
    plt.ylabel('Color (SRM)')
    plt.title('Color vs Gravity of BJCP Styles')
    #
    texts = [plt.text(xx[i], yy[i], styles[i], ha='center', va='center') for i in range(len(xx))]
    adjust_text(texts)
    for srm in range(1, 40):
        ax.add_patch(Rectangle((1, srm), .001, 1, color=interpolate_beer_color(srm)))
    plt.savefig('charts/color_v_grav.svg')
    plt.show()


def generate_ibu_v_grav_chart():
    xcol = 'ibu_mid'
    ycol = 'final_gravity_mid'
    xx = ds[xcol]
    yy = ds[ycol]
    styles = ds['Style']

    fig, ax = plt.subplots(figsize=(16, 16))
    srm_colors = [interpolate_beer_color(srm) for srm in yy]
    ax.scatter(xx, yy, color=srm_colors)
    plt.xlabel('IBU')
    plt.ylabel('Gravity')
    plt.title('Color vs Bitterness of BJCP Styles')
    #
    texts = [plt.text(xx[i], yy[i], styles[i], ha='center', va='center') for i in range(len(xx))]
    adjust_text(texts)
    plt.savefig('charts/ibu_v_grav.svg')
    plt.show()


if __name__ == "__main__":
    # generate_grav_v_srm_chart()
    # generate_grav_v_abv_chart()
    generate_ibu_v_grav_chart()


