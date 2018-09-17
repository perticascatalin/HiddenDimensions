import matplotlib.pyplot as plt

Z = [2, 6, 10, 14, 20, 32, 64, 128]
KM_sil = [0.37, 0.17, 0.11, 0.09, 0.07, 0.04, 0.05, 0.05]
SOM_sil = [0.11, 0.17, 0.15, 0.11, 0.10, 0.10, 0.17, 0.13]
KM_hom = [0.52, 0.54, 0.53, 0.52, 0.49, 0.47, 0.51, 0.47]
SOM_hom = [0.61, 0.76, 0.78, 0.79, 0.80, 0.79, 0.79, 0.81]

plt.xlabel('Z dim')
plt.plot(Z, KM_sil, Z, SOM_sil, Z, KM_hom, Z, SOM_hom)
plt.legend(['KM sil', 'SOM sil', 'KM hom', 'SOM hom'])
#plt.show()
plt.savefig('cluster_plots.png')