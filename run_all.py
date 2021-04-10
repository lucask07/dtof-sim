# high-level script to run three scripts to create figures
# Lucas Koerner, 2021/4/9

# Plot results from the Monte-Carlo sims
#  (at times plot_mc uses analytical calculations
#   to overlay onto the Monte-Carlo results)
# 	creates paper figures 2,3,5
exec(open('plot_mc.py').read())

# Plot analytical results
# 	creates paper figures 4,7
exec(open('analytical_plots.py').read())

# run Monte Carlo simulations
#	currently setup for a short simulation 
#	creates paper figure 1 
exec(open('histogram_mc.py').read())
