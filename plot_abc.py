import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import corner


thetas = np.loadtxt("nonly_thetas.dat")

niter = 20
nparts = 100

for i in range(niter):

    lims = []

    fig = corner.corner(thetas[nparts * i:nparts * (i + 1), :],
                        labels=['alpha','logM0','logM1', 'logMmin','sig'],
                        show_titles=True, title_args={"fontsize": 12},
                        truths=[1.15, 11.92, 13.94, 12.79, 0.39],
                        plot_datapoints=True,
                        levels=[0.68, 0.95],
                        bins=8, smooth=1.0)

    picname = "nonly_thetas_{0}.pdf".format(str(i))
    plotaddress = picname

    fig.savefig(plotaddress)

    os.system("scp {0} broiler:~/public_html/".format(plotaddress))
    os.system("ssh broiler chmod 644 /home/kilian/public_html/{0}"
                  .format(picname))
