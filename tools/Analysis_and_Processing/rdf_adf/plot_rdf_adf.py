from pylab import *
import numpy as np

def Plot_rdf_adf():

    aw = 3
    fs = 20
    font = {'size'   : fs}
    matplotlib.rc('font', **font)
    matplotlib.rc('axes' , linewidth=aw)

    def set_fig_properties(ax_list):
        tl = 8
        tw = 2
        tlm = 4

        for ax in ax_list:
            ax.tick_params(which='major', length=tl, width=tw)
            ax.tick_params(which='minor', length=tlm, width=tw)
            ax.tick_params(which='both', axis='both', direction='in', right=True, top=True)

    figure(figsize=(9,11))

    dftrdf = np.loadtxt('rdf.dat', skiprows=2)
    com_list = open('rdf.dat').readlines()[1].split()
    OOnum = com_list.index('O-O') - 3
    OHnum = com_list.index('O-H') - 3

    subplot(3,1,1)
    set_fig_properties([gca()])
    plot(dftrdf[:,0]*0.1, dftrdf[:,OOnum], color='r',linewidth=3)
    xlim([0.15, 0.8])
    gca().set_xticks([0.2,0.4,0.6,0.8])
    ylim([0, 4])
    gca().set_yticks([0,1,2,3,4])
    ylabel(r'$g_{\rm OO}$($r$)')
    xlabel(r'$r$ (nm)')
    legend(['water'],frameon=False,loc='upper right')
    text(0.07,4,'(a)')

    subplot(3,1,2)
    set_fig_properties([gca()])
    plot(dftrdf[:,0]*0.1, dftrdf[:,OHnum], color='b',linewidth=3)
    xlim([0.05, 0.4])
    gca().set_xticks([0.1,0.2,0.3,0.4])
    ylim([0, 4])
    gca().set_yticks([0,1,2,3,4])
    ylabel(r'$g_{\rm OH}$($r$)')
    xlabel(r'$r$ (nm)')
    legend(['water'],frameon=False,loc='upper right')
    text(0.01,4,'(b)')

    adfooo = np.loadtxt('adf-ooo.dat')
    adfhoh = np.loadtxt('adf-hoh.dat')
    subplot(3,1,3)
    set_fig_properties([gca()])
    plot(adfooo[:,0], adfooo[:,1]/np.sum(adfooo[:,1]), color='g',linewidth=3)
    plot(adfhoh[:,0], adfhoh[:,1]/np.sum(adfhoh[:,1]), color='y',linestyle='--',linewidth=3)
    xlim([0, 180])
    gca().set_xticks([0,30,60,90,120,150,180])
    ylim([0, 0.07])
    gca().set_yticks([0,0.02,0.04,0.06])
    ylabel(r'$P$($\theta$)')
    xlabel(r'$\theta$ ($^{\rm o}$)')
    legend(['OOO','HOH'],frameon=False,loc='upper right')
    text(-22,0.07,'(c)')

    tight_layout()
    savefig('rdf_adf.pdf', dpi=300)

Plot_rdf_adf()
