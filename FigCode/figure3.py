# pylint: disable=C,W
import numpy as np 
import plotUtils as pu

figName = "quasiParticlePredictions"

fo = pu.FigObj(2)

masses = np.load("masses_shapiro.npy")
shapiros = np.load("shapiros.npy")
prediction = np.load("prediction_shapiro.npy")


fo.AddPlot(masses, shapiros, ls = ' ', mk = 'o', color = 'k', label = r'simulated data')
fo.AddLine(masses, prediction, color = 'r', label = r'prediction')
fo.SetYLim(np.min(shapiros)/1.1 , np.max(shapiros)*1.1)
fo.SetLogLog(masses, shapiros)
fo.SetXLim(np.min(masses)/1.2 , np.max(masses)*1.2)
fo.SetXLabel(r'$m_{22}$')
fo.SetYLabel(r'$|\delta t^\mathrm{sh}| \, [\mathrm{ns}]$')
fo.SetTitle(r'Shapiro delay')
# fo.legend()


masses = np.load("masses_z.npy")
potential = np.load("potentials.npy")
prediction = np.load("prediction_z.npy")

fo.AddPlot(masses, potential, ls = ' ', mk = 'o', color = 'k', label = r'simulated data')
fo.AddLine(masses, prediction, color = 'r', label = r'prediction')
fo.legend()

fo.SetLogLog(masses, potential)
fo.SetYLabel(r'$\delta \Phi_\mathrm{rms} / c^2$')
fo.SetXLim(np.min(masses)/1.1 , np.max(masses)*1.1)
fo.SetYLim(np.min(potential)/1.1 , np.max(potential)*1.1)
fo.SetTitle(r'Potential rms')
fo.SetXLabel(r'$m_{22}$')

fo.SetWhiteSpace(0.3,0)
fo.save("quasiParticlePredictions")

fo.show()