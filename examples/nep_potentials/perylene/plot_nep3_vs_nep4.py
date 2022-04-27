
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import site
site.addsitedir('/home/elindgren/repos/calorine')
from calorine.io import read_loss  # nopep8

#title = 'MD@300K monomers and dimers'
#nep3_folder = 'md-perylene-nep3'
#nep4_folder = 'md-perylene-nep4'

title = 'Rattled Monomers'
nep3_folder = 'rattled_perylene_nep3'
nep4_folder = 'rattled_perylene_nep4'


# Load NEP3 and NEP4 rattled monomers
rattled_nep3_loss = read_loss(f'{nep3_folder}/loss.out')
rattled_nep4_loss = read_loss(f'{nep4_folder}/loss.out')

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(rattled_nep3_loss['RMSE_E_train'], label='NEP3 train')
ax[0].plot(rattled_nep3_loss['RMSE_E_test'], label='NEP3 test')
ax[0].plot(rattled_nep4_loss['RMSE_E_train'], label='NEP4 train')
ax[0].plot(rattled_nep4_loss['RMSE_E_test'], label='NEP4 test')
ax[0].legend(loc='best')
ax[0].set_xlabel('Generation')
ax[0].set_ylabel('Energy (eV)')


ax[1].plot(rattled_nep3_loss['RMSE_F_train'], label='NEP3 train')
ax[1].plot(rattled_nep3_loss['RMSE_F_test'], label='NEP3 test')
ax[1].plot(rattled_nep4_loss['RMSE_F_train'], label='NEP4 train')
ax[1].plot(rattled_nep4_loss['RMSE_F_test'], label='NEP4 test')
ax[1].legend(loc='best')
ax[1].set_xlabel('Generation')
ax[1].set_ylabel('Force (eV/Å)')
plt.suptitle(f'NEP3 vs NEP4 - {title} - Training history')
plt.tight_layout()
plt.savefig(f'{title}_training_history.png')

rattled_nep3_energy_train = np.loadtxt(f'{nep3_folder}/energy_train.out')
rattled_nep3_energy_test = np.loadtxt(f'{nep3_folder}/energy_test.out')
rattled_nep3_forces_train = np.loadtxt(f'{nep3_folder}/force_train.out')
rattled_nep3_forces_test = np.loadtxt(f'{nep3_folder}/force_test.out')

rattled_nep4_energy_train = np.loadtxt(f'{nep4_folder}/energy_train.out')
rattled_nep4_energy_test = np.loadtxt(f'{nep4_folder}/energy_test.out')
rattled_nep4_forces_train = np.loadtxt(f'{nep4_folder}/force_train.out')
rattled_nep4_forces_test = np.loadtxt(f'{nep4_folder}/force_test.out')


fig, ax = plt.subplots(2, 2, figsize=(12, 12))
rattled_nep3_energy_train_r2 = r2_score(
    rattled_nep3_energy_train[:, 1], rattled_nep3_energy_train[:, 0])
rattled_nep4_energy_train_r2 = r2_score(
    rattled_nep4_energy_train[:, 1], rattled_nep4_energy_train[:, 0])
emax, emin = np.max(rattled_nep3_energy_train[:, 1]), np.min(rattled_nep3_energy_train[:, 1])
ax[0, 0].scatter(rattled_nep3_energy_train[:, 1], rattled_nep3_energy_train[:, 0], label='NEP3')
ax[0, 0].scatter(rattled_nep4_energy_train[:, 1],
                 rattled_nep4_energy_train[:, 0], alpha=0.5, label='NEP4')
ax[0, 0].plot([emin, emax], [emin, emax], c='r')
ax[0, 0].text(x=(emin+emax)/2, y=(emin+emin)/2,
              s=f'NEP3 R2: {rattled_nep3_energy_train_r2:.3f}\nNEP4 R2: {rattled_nep4_energy_train_r2:.3f}')
ax[0, 0].legend(loc='best')
ax[0, 0].set_ylabel('Predicted Energy (eV)')
ax[0, 0].set_xlabel('DFT Energy (eV/atom)')
ax[0, 0].set_title('Energy - Train')

rattled_nep3_energy_test_r2 = r2_score(
    rattled_nep3_energy_test[:, 1], rattled_nep3_energy_test[:, 0])
rattled_nep4_energy_test_r2 = r2_score(
    rattled_nep4_energy_test[:, 1], rattled_nep4_energy_test[:, 0])
emax, emin = np.max(rattled_nep3_energy_test[:, 1]), np.min(rattled_nep3_energy_test[:, 1])
ax[0, 1].scatter(rattled_nep3_energy_test[:, 1], rattled_nep3_energy_test[:, 0], label='NEP3')
ax[0, 1].scatter(rattled_nep4_energy_test[:, 1],
                 rattled_nep4_energy_test[:, 0], alpha=0.5, label='NEP4')
ax[0, 1].plot([emin, emax], [emin, emax], c='r')
ax[0, 1].text(x=(emin+emax)/2, y=(emin+emin)/2,
              s=f'NEP3 R2: {rattled_nep3_energy_test_r2:.3f}\nNEP4 R2: {rattled_nep4_energy_test_r2:.3f}')
ax[0, 1].legend(loc='best')
ax[0, 1].set_ylabel('Predicted Energy (eV)')
ax[0, 1].set_xlabel('DFT Energy (eV/atom)')
ax[0, 1].set_title('Energy - Test')


rattled_nep3_force_train_r2 = r2_score(
    rattled_nep3_forces_train[:, 3:].flatten(),
    rattled_nep3_forces_train[:, 0:3].flatten())
rattled_nep4_force_train_r2 = r2_score(
    rattled_nep4_forces_train[:, 3:].flatten(),
    rattled_nep4_forces_train[:, 0:3].flatten())
emax, emin = np.max(rattled_nep3_forces_train[:, 3:].flatten()), np.min(
    rattled_nep3_forces_train[:, 3:].flatten())
ax[1, 0].scatter(rattled_nep3_forces_train[:, 3:].flatten(),
                 rattled_nep3_forces_train[:, 0:3].flatten(), label='NEP3')
ax[1, 0].scatter(rattled_nep4_forces_train[:, 3:].flatten(),
                 rattled_nep4_forces_train[:, 0:3].flatten(), alpha=0.5, label='NEP4')
ax[1, 0].plot([emin, emax], [emin, emax], c='r')
ax[1, 0].text(x=(emin+emax)/2, y=(emin+emin)/2,
              s=f'NEP3 R2: {rattled_nep3_force_train_r2:.3f}\nNEP4 R2: {rattled_nep4_force_train_r2:.3f}')
ax[1, 0].legend(loc='best')
ax[1, 0].set_ylabel('Predicted Force (eV)')
ax[1, 0].set_xlabel('DFT Force (eV/Å)')
ax[1, 0].set_title('Force - train')


rattled_nep3_force_test_r2 = r2_score(
    rattled_nep3_forces_test[:, 3:].flatten(),
    rattled_nep3_forces_test[:, 0:3].flatten())
rattled_nep4_force_test_r2 = r2_score(
    rattled_nep4_forces_test[:, 3:].flatten(),
    rattled_nep4_forces_test[:, 0:3].flatten())
emax, emin = np.max(rattled_nep3_forces_test[:, 3:].flatten()), np.min(
    rattled_nep3_forces_test[:, 3:].flatten())
ax[1, 1].scatter(rattled_nep3_forces_test[:, 3:].flatten(),
                 rattled_nep3_forces_test[:, 0:3].flatten(), label='NEP3')
ax[1, 1].scatter(rattled_nep4_forces_test[:, 3:].flatten(),
                 rattled_nep4_forces_test[:, 0:3].flatten(), alpha=0.5, label='NEP4')
ax[1, 1].plot([emin, emax], [emin, emax], c='r')
ax[1, 1].text(x=(emin+emax)/2, y=(emin+emin)/2,
              s=f'NEP3 R2: {rattled_nep3_force_test_r2:.3f}\nNEP4 R2: {rattled_nep4_force_test_r2   :.3f}')
ax[1, 1].legend(loc='best')
ax[1, 1].set_ylabel('Predicted Force (eV/Å)')
ax[1, 1].set_xlabel('DFT Force (eV/Å)')
ax[1, 1].set_title('Force - Test')
plt.suptitle(f'NEP3 vs NEP4 - {title} - Parity plots')
plt.tight_layout()
plt.savefig(f'{title}_parity.png')
