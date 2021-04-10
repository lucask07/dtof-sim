[![DOI](https://zenodo.org/badge/356401320.svg)](https://zenodo.org/badge/latestdoi/356401320)

# dtof-sim

Analysis of direct TOF (dToF) timing precision that uses analytical calculations and Monte-Carlo simulations. 

Examples of questions pursued:

* How does TDC bin size effect timing precision?
* How does the noise depend on the number of signal and background photons? 
* Do analytical calculations match the Monte-Carlo results? 

For more details see:

> Koerner, L.J. "Models of Direct Time-of-Flight Sensor Precision that Enable Optimal Design and Dynamic Configuration" IEEE 
Transactions on Instrumentation and Measurement, to appear. 

The four main files are:

1. *analytical_calcs.py*: functions to calculate the timing precision using the expression dervied from the work of Thompson, et al. 2002 and the Cramer-Rao bound (CRB).
2. *analytical_plots.py*: functions to plot the analytical results  
3. *histogram_mc.py*: run Monte-Carlo simulations. That are many adjustable parameters controlled by the configuration dictionary `cfg`.
4. *plot_mc.py*: plot the results of the Monte-Carlo simulations.

A top-level script *run_all.py* runs examples of each. This code generates most of the figures in the paper mentioned above.

If this work benefits you please cite the IEEE Transactions on Instrumentation and Measurement mentioned above.
