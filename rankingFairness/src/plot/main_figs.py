import numpy as np
import random
import seaborn as sns
import os.path as osp
import matplotlib.pyplot as plt
from matplotlib import lines
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as mticker

from rankingFairness.src.experimentMultipleGroups import MARKERS, COLORMAP


def plot_synthetic(EO_constraint, cost_groups, total_cost, rankingAlgos, rankingAlgos_dict, SynthMarker_dict,
    save_path, filename, fig_kw, dcg_option=False, legendFileName=None, dcg=None):
    markevery=fig_kw.get('markevery')
    linewidth=fig_kw.get('linewidth')
    markersize=fig_kw.get('markersize')
    markeredgecolor=fig_kw.get('markeredgecolor')
    markeredgewidth=fig_kw.get('markeredgewidth')
    ticklabelsize=fig_kw.get('ticklabelsize')
    yLabelSize=fig_kw.get('yLabelSize')
    alpha_stochastic=fig_kw.get('alpha_stochastic')
    offset=fig_kw.get('offset')
    figsize=fig_kw.get('figsize')
    nrows=fig_kw.get('nrows')
    ncols=fig_kw.get('ncols')
    interval_y=fig_kw.get('interval_y')
    hspace=fig_kw.get('hspace')
    wspace=fig_kw.get('wspace')
    rasterized=fig_kw.get('rasterized')

    plt.style.use('default')
    fig= plt.figure(figsize=figsize)
    grid = plt.GridSpec(nrows=nrows, ncols=ncols, hspace=hspace, wspace=wspace)
    plt.rc('font', family='serif')
    
    
    

    if dcg_option:
        subfigs = fig.subfigures(1, 3, width_ratios=[2, 3, 1])
        axs2 = subfigs[2].subplots(2,1)
    else:
        subfigs = fig.subfigures(1, 3, width_ratios=[2, 3, 2])
        axs2 = subfigs[2].subplots(1,1)
    axs0 = subfigs[0].subplots(1,1)
    axs1 = subfigs[1].subplots(2,3,sharex=True, sharey=True)
    
    
    plt.rc('font', family='serif')

    num_docs=total_cost.shape[-1]
    num_groups=cost_groups.shape[1]
    for i,r in rankingAlgos_dict.items():
        axs0.plot(np.arange(num_docs)+1, EO_constraint[r,:], label=str(rankingAlgos[r].name()),
                c=COLORMAP[rankingAlgos[r].name()], rasterized=rasterized)


    for axis in axs1.flatten():
        axis.tick_params(axis='y', which='major', labelsize=ticklabelsize)
        axis.tick_params(axis='x', which='major', labelsize=ticklabelsize)
        axis.set_ylim(-offset, 1 + offset)
        axis.set_xlim(1-offset, num_docs*(1+offset))
        axis.xaxis.set_major_locator(mticker.MultipleLocator(int(num_docs*(1))))
        axis.yaxis.set_major_locator(mticker.MultipleLocator(1))
        axis.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        axis.plot(np.arange(num_docs)+1, [1-(i/num_docs) for i in range(1, num_docs+1)], linewidth=linewidth,
        color=COLORMAP['Uniform'], rasterized=rasterized, alpha=alpha_stochastic, label=r'$\bf{\mathbb{E}[\sigma^{unif}]}$')
        

    for g in range(num_groups):
        axs1[0,0].plot(np.arange(num_docs)+1, cost_groups[rankingAlgos_dict.get('EOR'),g,...], linestyle='dashed',
         marker=MARKERS[g], markeredgecolor=markeredgecolor, markeredgewidth=markeredgewidth,
        c=COLORMAP['EOR'], linewidth=linewidth, markevery=markevery, markerfacecolor='w', 
        markersize=markersize, rasterized=rasterized, label='EOR')


        axs1[0,1].plot(np.arange(num_docs)+1, cost_groups[rankingAlgos_dict.get('DP'),g,...], linestyle='dashed',
         marker=MARKERS[g], markeredgecolor=markeredgecolor, markeredgewidth=markeredgewidth,
        c=COLORMAP['DP'], linewidth=linewidth, markevery=markevery, markerfacecolor='w',
        markersize=markersize, rasterized=rasterized, label='DP')


        axs1[1,0].plot(np.arange(num_docs)+1, cost_groups[rankingAlgos_dict.get('PRP'),g,...], linestyle='dashed',
         marker=MARKERS[g], markeredgecolor=markeredgecolor, markeredgewidth=markeredgewidth,
        c=COLORMAP['PRP'], linewidth=linewidth, markevery=markevery, markerfacecolor='w',
        markersize=markersize, rasterized=rasterized, label='PRP')


        axs1[1,1].plot(np.arange(num_docs)+1, cost_groups[rankingAlgos_dict.get('TS'),g,...], linestyle='dashed',
         marker=MARKERS[g], markeredgecolor=markeredgecolor, markeredgewidth=markeredgewidth,
        c=COLORMAP['TS'], linewidth=linewidth, markevery=markevery, markerfacecolor='w',
        markersize=markersize, rasterized=rasterized, label='TS')
    
        axs1[0,2].plot(np.arange(num_docs)+1, cost_groups[rankingAlgos_dict.get('EXP'),g,...], linestyle='dashed', 
        marker=MARKERS[g], markeredgecolor=markeredgecolor, markeredgewidth=markeredgewidth,
        c=COLORMAP['EXP'], linewidth=linewidth, markevery=markevery, markerfacecolor='w',
        markersize=markersize, rasterized=rasterized, label='EXP')
        
        axs1[1,2].plot(np.arange(num_docs)+1, cost_groups[rankingAlgos_dict.get('RA'),g,...], linestyle='dashed',
         marker=MARKERS[g], markeredgecolor=markeredgecolor, markeredgewidth=markeredgewidth,
        c=COLORMAP['RA'], linewidth=linewidth, markevery=markevery, markerfacecolor='w',
        markersize=markersize, rasterized=rasterized, label='RA')
        
    for axis in [axs0]:
        axis.tick_params(axis='y', which='major', labelsize=ticklabelsize)
        axis.tick_params(axis='x', which='major', labelsize=ticklabelsize)
        axis.xaxis.set_major_locator(mticker.MultipleLocator(int(num_docs*(1))))
        axis.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axis.set_xlim(-offset, num_docs*(1+offset))
    axs0.yaxis.set_major_locator(mticker.MultipleLocator(interval_y))
    axs0.set_ylabel(r'$\delta(\sigma_k)$',fontsize=yLabelSize)

    if dcg_option:
        for i,r in enumerate(rankingAlgos): 
            axs2[0].plot(np.arange(num_docs)+1, total_cost[len(rankingAlgos)-1-i,...], linestyle='dashed', 
            marker=MARKERS[-1], markeredgewidth=markeredgewidth,
            c=COLORMAP[rankingAlgos[len(rankingAlgos)-1-i].name()], linewidth=linewidth, markevery=markevery, markerfacecolor='w',
            markersize=markersize, markeredgecolor=COLORMAP[rankingAlgos[len(rankingAlgos)-1-i].name()], rasterized=rasterized)

            axs2[1].plot(np.arange(num_docs)+1, dcg[len(rankingAlgos)-1-i,...], linestyle='solid', 
            c=COLORMAP[rankingAlgos[len(rankingAlgos)-1-i].name()], linewidth=linewidth,rasterized=rasterized)

        axs2[1].set_ylabel(r'nDCG',fontsize=yLabelSize)
        axs2[0].set_ylabel("Total Costs ", fontsize=yLabelSize)
        axs2[1].yaxis.set_major_locator(mticker.MultipleLocator(np.max(np.array(dcg))*0.5))
        for axis in axs2.flatten():
            axis.tick_params(axis='y', which='major', labelsize=ticklabelsize)
            axis.tick_params(axis='x', which='major', labelsize=ticklabelsize)
            axis.xaxis.set_major_locator(mticker.MultipleLocator(int(num_docs*(1))))
            axis.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            axis.set_xlim(1-offset, num_docs+1)
    else:
        for i,r in enumerate(rankingAlgos): 
            axs2.plot(np.arange(num_docs)+1, total_cost[len(rankingAlgos)-1-i,...], linestyle='dashed', 
            marker=MARKERS[-1], markeredgewidth=markeredgewidth,
            c=COLORMAP[rankingAlgos[len(rankingAlgos)-1-i].name()], linewidth=linewidth, markevery=markevery, markerfacecolor='w',
            markersize=markersize, markeredgecolor=COLORMAP[rankingAlgos[len(rankingAlgos)-1-i].name()], rasterized=rasterized)
        axs2.set_ylabel("Total Costs ", fontsize=yLabelSize)

        axs2.tick_params(axis='y', which='major', labelsize=ticklabelsize)
        axs2.tick_params(axis='x', which='major', labelsize=ticklabelsize)
        axs2.xaxis.set_major_locator(mticker.MultipleLocator(int(num_docs*(1))))
        axs2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axs2.set_xlim(1-offset, num_docs+1)


    if dcg_option:
        subfigs[0].supxlabel("(a)", fontsize=yLabelSize, y=-0.07)
    else:
        subfigs[0].supxlabel("(a)", fontsize=yLabelSize, y=-0.08)
    axs0.set_xlabel("Length of Ranking (k)", fontsize=ticklabelsize, y=-0.5)
    
    axs0.axhline(y=0, xmin=0, xmax=num_docs, color='black', linestyle='dashed', rasterized=rasterized)
    axs0.text(num_docs-8, -0.07, r'$\delta_k=0$', c='black', fontsize=10)  
    subfigs[1].supxlabel("(b)", fontsize=yLabelSize, y=-0.06)
    subfigs[2].supxlabel("(c)", fontsize=yLabelSize, y=-0.06)

    subfigs[1].supylabel("Group Costs", fontsize=yLabelSize)
    
    marker_ls=[]

    for k,v in rankingAlgos_dict.items():
        marker_ls.append(Line2D([], [], color=COLORMAP[rankingAlgos[v].name()], linestyle='solid',
                    linewidth=linewidth, label=rankingAlgos[v].name()))
   
    axs1[0,0].legend(handles=[Line2D([], [], color=COLORMAP['EOR'], linestyle='solid',
                linewidth=linewidth, label='EOR')], fontsize=ticklabelsize-3, frameon=False, loc='upper right')
    axs1[0,1].legend(handles=[Line2D([], [], color=COLORMAP['DP'], linestyle='solid', 
                linewidth=linewidth, label='DP')], fontsize=ticklabelsize-3, frameon=False, loc='upper right')
    axs1[1,0].legend(handles=[Line2D([], [], color=COLORMAP['PRP'], linestyle='solid', 
                linewidth=linewidth, label='PRP')], fontsize=ticklabelsize-3, frameon=False, loc='upper right')
    axs1[1,1].legend(handles=[Line2D([], [], color=COLORMAP['TS'], linestyle='solid', 
                linewidth=linewidth, label='TS')], fontsize=ticklabelsize-3, frameon=False, loc='upper right')
    axs1[0,2].legend(handles=[Line2D([], [], color=COLORMAP['EXP'], linestyle='solid', 
                linewidth=linewidth, label='EXP')], fontsize=ticklabelsize-3, frameon=False, loc='upper right')
    axs1[1,2].legend(handles=[Line2D([], [], color=COLORMAP['RA'], linestyle='solid', 
                linewidth=linewidth, label='RA')], fontsize=ticklabelsize-3, frameon=False, loc='upper right')
    

    for k,v in SynthMarker_dict.items():
        marker_ls.append(Line2D([], [], marker=v , markerfacecolor='w', markersize=markersize, linestyle='None', 
                    label=k, markeredgecolor=markeredgecolor))

    if len(rankingAlgos)<7:
        subfigs[1].delaxes(axs1[1,2])
    if legendFileName is not None:
        legend = plt.legend(handles=marker_ls, fontsize=ticklabelsize, loc='upper center',
                 bbox_to_anchor=(-0.8, 1.5),ncol=8, numpoints=1)

        bbox=legend.get_window_extent()
        expand=[-5,-5,5,5]
        bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())

        plt.savefig(f"{osp.join(save_path,legendFileName)}", bbox_inches=bbox)

    plt.savefig(osp.join(save_path, filename), bbox_inches='tight')

    plt.show() 
    plt.close()


def plot_census(EO_constraint, cost_groups, total_cost, rankingAlgos, rankingAlgos_dict, SynthMarker_dict,
    save_path, filename, fig_kw, dcg_option=False, legendFileName=None, dcg=None):
    markevery=fig_kw.get('markevery')
    linewidth=fig_kw.get('linewidth')
    markersize=fig_kw.get('markersize')
    markeredgecolor=fig_kw.get('markeredgecolor')
    markeredgewidth=fig_kw.get('markeredgewidth')
    ticklabelsize=fig_kw.get('ticklabelsize')
    yLabelSize=fig_kw.get('yLabelSize')
    alpha_stochastic=fig_kw.get('alpha_stochastic')
    offset=fig_kw.get('offset')
    figsize=fig_kw.get('figsize')
    nrows=fig_kw.get('nrows')
    ncols=fig_kw.get('ncols')
    interval_y=fig_kw.get('interval_y')
    hspace=fig_kw.get('hspace')
    wspace=fig_kw.get('wspace')
    rasterized=fig_kw.get('rasterized')

    plt.style.use('default')
    fig= plt.figure(figsize=figsize)
    grid = plt.GridSpec(nrows=nrows, ncols=ncols, hspace=hspace, wspace=wspace)
    plt.rc('font', family='serif')
    
    if dcg_option:
        subfigs = fig.subfigures(1, 3, width_ratios=[2, 3, 1])
        axs2 = subfigs[2].subplots(2,1)
        axs1 = subfigs[1].subplots(2,3,sharex=True, sharey=True)
    else:
        subfigs = fig.subfigures(1, 3, width_ratios=[1, 1, 1])
        axs2 = subfigs[2].subplots(1,1)
        axs1 = subfigs[1].subplots(2,2,sharex=True, sharey=True)
    axs0 = subfigs[0].subplots(1,1)
    
    
    
    plt.rc('font', family='serif')

    num_docs=total_cost.shape[-1]
    num_groups=cost_groups.shape[1]
    for i,r in rankingAlgos_dict.items():
        axs0.plot(np.arange(num_docs)+1, EO_constraint[r,:], label=str(rankingAlgos[r].name()),
                c=COLORMAP[rankingAlgos[r].name()], rasterized=rasterized)


    for axis in axs1.flatten():
        axis.tick_params(axis='y', which='major', labelsize=ticklabelsize)
        axis.tick_params(axis='x', which='major', labelsize=ticklabelsize)
        axis.set_ylim(-offset*4, 1 + offset*2)
        axis.set_xlim(1-offset, num_docs*(1+offset*2))
        axis.xaxis.set_major_locator(mticker.MultipleLocator(int(num_docs*(1))))
        axis.yaxis.set_major_locator(mticker.MultipleLocator(1))
        axis.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        axis.plot(np.arange(num_docs)+1, [1-(i/num_docs) for i in range(1, num_docs+1)], linewidth=linewidth,
        color=COLORMAP['Uniform'], rasterized=rasterized, alpha=alpha_stochastic, label=r'$\bf{\mathbb{E}[\sigma^{unif}]}$')
        

    for g in range(num_groups):
        axs1[0,0].plot(np.arange(num_docs)+1, cost_groups[rankingAlgos_dict.get('EOR'),g,...], linestyle='dashed', 
        marker=MARKERS[g], markeredgecolor=markeredgecolor, markeredgewidth=markeredgewidth,
        c=COLORMAP['EOR'], linewidth=linewidth, markevery=markevery, markerfacecolor='w', 
        markersize=markersize, rasterized=rasterized, label='EOR')


        axs1[0,1].plot(np.arange(num_docs)+1, cost_groups[rankingAlgos_dict.get('DP'),g,...], linestyle='dashed', 
        marker=MARKERS[g], markeredgecolor=markeredgecolor, markeredgewidth=markeredgewidth,
        c=COLORMAP['DP'], linewidth=linewidth, markevery=markevery, markerfacecolor='w',
        markersize=markersize, rasterized=rasterized, label='DP')


        axs1[1,0].plot(np.arange(num_docs)+1, cost_groups[rankingAlgos_dict.get('PRP'),g,...], linestyle='dashed',
         marker=MARKERS[g], markeredgecolor=markeredgecolor, markeredgewidth=markeredgewidth,
        c=COLORMAP['PRP'], linewidth=linewidth, markevery=markevery, markerfacecolor='w',
        markersize=markersize, rasterized=rasterized, label='PRP')


        axs1[1,1].plot(np.arange(num_docs)+1, cost_groups[rankingAlgos_dict.get('TS'),g,...], linestyle='dashed',
         marker=MARKERS[g], markeredgecolor=markeredgecolor, markeredgewidth=markeredgewidth,
        c=COLORMAP['TS'], linewidth=linewidth, markevery=markevery, markerfacecolor='w',
        markersize=markersize, rasterized=rasterized, label='TS')

        if dcg_option:
            axs1[0,2].plot(np.arange(num_docs)+1, cost_groups[rankingAlgos_dict.get('RA'),g,...], linestyle='dashed',
             marker=MARKERS[g], markeredgecolor=markeredgecolor, markeredgewidth=markeredgewidth,
            c=COLORMAP['RA'], linewidth=linewidth, markevery=markevery, markerfacecolor='w',
            markersize=markersize, rasterized=rasterized, label='RA')
        
    for axis in [axs0]:
        axis.tick_params(axis='y', which='major', labelsize=ticklabelsize)
        axis.tick_params(axis='x', which='major', labelsize=ticklabelsize)
        axis.xaxis.set_major_locator(mticker.MultipleLocator(int(num_docs*(1))))
        axis.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axis.set_xlim(-offset, num_docs*(1+offset))
    axs0.yaxis.set_major_locator(mticker.MultipleLocator(interval_y))
    axs0.set_ylabel(r'$\delta(\sigma_k)$',fontsize=yLabelSize)

    if dcg_option:
        subfigs[1].delaxes(axs1[1,2])
        for i,r in enumerate(rankingAlgos): 
            axs2[0].plot(np.arange(num_docs)+1, total_cost[len(rankingAlgos)-1-i,...], linestyle='dashed', 
            marker=MARKERS[-1], markeredgewidth=markeredgewidth,
            c=COLORMAP[rankingAlgos[len(rankingAlgos)-1-i].name()], linewidth=linewidth, markevery=markevery, markerfacecolor='w',
            markersize=markersize, markeredgecolor=COLORMAP[rankingAlgos[len(rankingAlgos)-1-i].name()], rasterized=rasterized)

            axs2[1].plot(np.arange(num_docs)+1, dcg[len(rankingAlgos)-1-i,...], linestyle='solid', 
            c=COLORMAP[rankingAlgos[len(rankingAlgos)-1-i].name()], linewidth=linewidth,rasterized=rasterized)

        axs2[1].set_ylabel(r'nDCG',fontsize=yLabelSize)
        axs2[0].set_ylabel("Total Costs ", fontsize=yLabelSize)
        axs2[0].yaxis.set_major_locator(mticker.MultipleLocator(0.5))
        axs2[1].yaxis.set_major_locator(mticker.MultipleLocator(np.max(np.array(dcg))*0.5))
        
        for axis in axs2.flatten():
            axis.tick_params(axis='y', which='major', labelsize=ticklabelsize)
            axis.tick_params(axis='x', which='major', labelsize=ticklabelsize)
            axis.xaxis.set_major_locator(mticker.MultipleLocator(int(num_docs*(1))))
            axis.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            axis.set_xlim(1-offset, num_docs*(1+offset))
        subfigs[1].supylabel("Group Costs", fontsize=yLabelSize)
    else:
        for i,r in enumerate(rankingAlgos): 
            axs2.plot(np.arange(num_docs)+1, total_cost[len(rankingAlgos)-1-i,...], linestyle='dashed', 
            marker=MARKERS[-1], markeredgewidth=markeredgewidth,
            c=COLORMAP[rankingAlgos[len(rankingAlgos)-1-i].name()], linewidth=linewidth, markevery=markevery, markerfacecolor='w',
            markersize=markersize, markeredgecolor=COLORMAP[rankingAlgos[len(rankingAlgos)-1-i].name()], rasterized=rasterized)
        axs2.set_ylabel("Total Costs ", fontsize=yLabelSize)

        axs2.tick_params(axis='y', which='major', labelsize=ticklabelsize)
        axs2.tick_params(axis='x', which='major', labelsize=ticklabelsize)
        axs2.xaxis.set_major_locator(mticker.MultipleLocator(int(num_docs*(1))))
        axs2.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
        axs2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axs2.set_xlim(1-offset, num_docs*(1+offset))
        subfigs[1].supylabel("Group Costs", fontsize=yLabelSize, x=-0.04)


    if dcg_option:
        subfigs[0].supxlabel("(a)", fontsize=yLabelSize, y=-0.07)
    else:
        subfigs[0].supxlabel("(a)", fontsize=yLabelSize, y=-0.08)
    axs0.set_xlabel("Length of Ranking (k)", fontsize=ticklabelsize)
    
    axs0.axhline(y=0, xmin=0, xmax=num_docs, color='black', linestyle='dashed', rasterized=rasterized)
    axs0.text(num_docs*(0.7), -offset, r'$\delta_k=0$', c='black', fontsize=10)  
    subfigs[1].supxlabel("(b)", fontsize=yLabelSize, y=-0.07)
    subfigs[2].supxlabel("(c)", fontsize=yLabelSize, y=-0.07)

    
    
    marker_ls=[]

    for k,v in rankingAlgos_dict.items():
        marker_ls.append(Line2D([], [], color=COLORMAP[rankingAlgos[v].name()], linestyle='solid',
                    linewidth=linewidth, label=rankingAlgos[v].name()))
   
    axs1[0,0].legend(handles=[Line2D([], [], color=COLORMAP['EOR'], linestyle='solid',
                linewidth=linewidth, label='EOR')], fontsize=ticklabelsize-3, frameon=False, loc='upper right')
    axs1[0,1].legend(handles=[Line2D([], [], color=COLORMAP['DP'], linestyle='solid', 
                linewidth=linewidth, label='DP')], fontsize=ticklabelsize-3, frameon=False, loc='upper right')
    axs1[1,0].legend(handles=[Line2D([], [], color=COLORMAP['PRP'], linestyle='solid', 
                linewidth=linewidth, label='PRP')], fontsize=ticklabelsize-3, frameon=False, loc='upper right')
    axs1[1,1].legend(handles=[Line2D([], [], color=COLORMAP['TS'], linestyle='solid', 
                linewidth=linewidth, label='TS')], fontsize=ticklabelsize-3, frameon=False, loc='upper right')
    if dcg_option:
        axs1[0,2].legend(handles=[Line2D([], [], color=COLORMAP['RA'], linestyle='solid', 
                linewidth=linewidth, label='RA')], fontsize=ticklabelsize-3, frameon=False, loc='upper right')
    

    for k,v in SynthMarker_dict.items():
        marker_ls.append(Line2D([], [], marker=v , markerfacecolor='w', markersize=markersize, linestyle='None', 
                    label=k, markeredgecolor=markeredgecolor))

    if legendFileName is not None:
        legend = plt.legend(handles=marker_ls, fontsize=ticklabelsize, loc='upper center',
                 bbox_to_anchor=(-0.8, 1.5),ncol=8, numpoints=1)

        bbox=legend.get_window_extent()
        expand=[-5,-5,5,5]
        bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())

        plt.savefig(f"{osp.join(save_path,legendFileName)}", bbox_inches=bbox)

    plt.savefig(osp.join(save_path, filename), bbox_inches='tight')

    plt.show() 
    plt.close()