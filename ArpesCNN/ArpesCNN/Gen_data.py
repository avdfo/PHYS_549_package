"""Main module.
This module generates tight-binding (TB) and ange-resolved photoemission spectroscopy (ARPES) data sets based on the
Chinook package. The simulated data from this module is customized for the training purpose of convolutional neural
network (CNN). Code or functions here that bear similar roles in Chinook like diagonalizing TB hamiltonian, computing
ARPES matrix element effets, constructing 3D datacubes and so on are adapted from Chinook package source code.
With the "simulation" class defined here, users don't have to import Chinook in Python code but only need to provide
Chinook-like basis, Hamiltonian, ARPES, and momentum (optional) dictionaries as the input to the "simulation".

One key difference of the band structure generated here is that we also construct a 3D datacube for the TB model, just
like the one for ARPES. Meanwhile, all the pixels coinciding with certain calculated energy eigenvalues in the TB 2D or
3D arrays are assigned with a weight of 1, while pixels with no bands are assigned with weight 0. In that sense,it would
be conducive for the CNN to compare the normalized ARPES data (with weight on each pixel between 0 and 1) with the
ground truth, TB data. To accelerate the data simulation process, we also add a function generating ARPES data without
considering matrix element effect.
This module is written by Yichen Zhang, with the input from Ziqin Yue on adding noise to simulated ARPES data.
"""

import numpy as np
import chinook.build_lib as build_lib
import chinook.intensity_map as imap
from chinook.ARPES_lib import experiment
from chinook.ARPES_lib import pol_2_sph
import scipy.ndimage as nd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import os

class simulation:

    def __init__(self, basis, hamiltonian, arpes, kdict=None):
        self.basis = basis
        self.hamiltonian = hamiltonian
        self.basis_object = build_lib.gen_basis(self.basis)
        self.ARPES = arpes
        if kdict == None:
            self.TB = build_lib.gen_TB(self.basis_object,self.hamiltonian)
        else:
            self.kdict = kdict
            self.k_object = build_lib.gen_K(self.kdict)
            self.TB = build_lib.gen_TB(self.basis_object, self.hamiltonian, self.k_object)
        self.AE = experiment(self.TB, self.ARPES)

    # Function providing the 3D TB datacube and/or the selected 2D slice plot.
    def TB_cube(self, ARPES_dict=None, slice_select=None, add_map=False, plot_bands=False, ax=None, colourmap=None):
        if not hasattr(self.AE, 'pks'):
            self.TB_diagonalize()

        if ARPES_dict is not None:
            self.AE.update_pars(ARPES_dict)

        if colourmap is None:
            colourmap = cm.Blues

        fermi = self.AE.T_distribution()
        w = np.linspace(*self.AE.cube[2])

        I = np.zeros((self.AE.cube[1][2], self.AE.cube[0][2], self.AE.cube[2][2]))

        accu = 2. * (self.AE.cube[2][1] - self.AE.cube[2][0]) / self.AE.cube[2][2]

        # Notice that the TB band strucutre here also bears a Fermi-Dirac distribution
        for p in range(len(self.AE.pks)):
            I[int(np.real(self.AE.pks[p, 1])), int(np.real(self.AE.pks[p, 2])), :] += np.where(np.abs(w - self.AE.pks[p,3])<accu, 1., 0.) * fermi

        I[I>1.] = 1.
        I = np.array(I, dtype='float32')

        if slice_select != None:
            ax_img = self.AE.plot_intensity_map(I, slice_select, plot_bands, ax, colourmap=colourmap)

        if add_map:
            self.AE.maps.append(
                imap.intensity_map(len(self.AE.maps), I, self.AE.cube, self.AE.kz, self.AE.T, self.AE.hv, self.AE.pol, self.AE.dE, self.AE.dk,
                                    self.AE.SE_args, self.AE.sarpes, self.AE.ang))
        else:
            self.AE.maps = [
                imap.intensity_map(len(self.AE.maps), I, self.AE.cube, self.AE.kz, self.AE.T, self.AE.hv, self.AE.pol, self.AE.dE, self.AE.dk,
                                    self.AE.SE_args, self.AE.sarpes, self.AE.ang)]
        if slice_select:
            return I, ax_img
        else:
            return I

    # Similar to the datacube calculating in Chinook. However, to save time for TB model, we delete the part for
    # calculating ARPES matrix elements
    def TB_diagonalize(self, ARPES_dict=None, diagonalize=False):
        if ARPES_dict is not None:
            self.AE.update_pars(ARPES_dict, True)

        self.AE.basis = self.AE.rot_basis()

        print('Initiate diagonalization: ')
        self.AE.diagonalize(diagonalize)
        print('Diagonalization Complete.')
        nstates = len(self.AE.basis)
        if self.AE.truncate:
            self.AE.basis, self.AE.Ev = self.AE.truncate_model()

        dE = (self.AE.cube[2][1] - self.AE.cube[2][0]) / self.AE.cube[2][2]
        dig_range = (self.AE.cube[2][0] - 5 * dE, self.AE.cube[2][1] + 5 * dE)

        self.AE.pks = np.array([[i, np.floor(np.floor(i / nstates) / np.shape(self.AE.X)[1]),
                              np.floor(i / nstates) % np.shape(self.AE.X)[1], self.AE.Eb[i]] for i in range(len(self.AE.Eb)) if
                             dig_range[0] <= self.AE.Eb[i] <= dig_range[-1]])
        if len(self.AE.pks) == 0:
            raise ValueError(
                'ARPES Calculation Error: no states found in energy window. Consider refining the region of interest')

    # Function providing the 3D ARPES datacube and/or the selected 2D slice plot.
    def ARPES_cube(self, ARPES_dict=None, slice_select=None, add_map=False, plot_bands=False, ax=None, colourmap=None, noise=False, NSR=0.8):
        if not hasattr(self.AE, 'Mk'):
            self.AE.datacube()

        if ARPES_dict is not None:
            self.AE.update_pars(ARPES_dict)

        if colourmap is None:
            colourmap = cm.Blues

        if self.AE.sarpes is not None:
            spin_Mk = self.AE.sarpes_projector()
            if self.AE.coord_type == 'momentum':
                pol = pol_2_sph(self.AE.pol)

                M_factor = np.power(abs(np.einsum('ij,j->i', spin_Mk[:, int((self.AE.sarpes[0] + 1) / 2), :], pol)), 2)
            elif self.AE.coord_type == 'angle':
                all_pol = self.AE.gen_all_pol()
                M_factor = np.power(abs(np.einsum('ij,ij->i', spin_Mk[:, int((self.AE.sarpes[0] + 1) / 2), :], all_pol)),
                                    2)
        else:
            if self.AE.coord_type == 'momentum':
                pol = pol_2_sph(self.AE.pol)

                M_factor = np.sum(np.power(abs(np.einsum('ijk,k->ij', self.AE.Mk, pol)), 2), axis=1)
            elif self.AE.coord_type == 'angle':
                all_pol = self.AE.gen_all_pol()
                M_factor = np.sum(np.power(abs(np.einsum('ijk,ik->ij', self.AE.Mk, all_pol)), 2), axis=1)

        SE = self.AE.SE_gen()
        fermi = self.AE.T_distribution()
        w = np.linspace(*self.AE.cube[2])

        I = np.zeros((self.AE.cube[1][2], self.AE.cube[0][2], self.AE.cube[2][2]))

        if np.shape(SE) == np.shape(I):
            SE_k = True
        else:
            SE_k = False

        for p in range(len(self.AE.pks)):
            if not SE_k:
                I[int(np.real(self.AE.pks[p, 1])), int(np.real(self.AE.pks[p, 2])), :] += M_factor[p] * np.imag(
                    -1. / (np.pi * (w - self.AE.pks[p, 3] - (SE - 0.00005j)))) * fermi
            else:
                I[int(np.real(self.AE.pks[p, 1])), int(np.real(self.AE.pks[p, 2])), :] += M_factor[p] * np.imag(-1. / (
                        np.pi * (w - self.AE.pks[p, 3] - (
                            SE[int(np.real(self.AE.pks[p, 1])), int(np.real(self.AE.pks[p, 2])), :] - 0.00005j)))) * fermi

        #kxg = (self.AE.cube[0][2] * self.AE.dk / (self.AE.cube[0][1] - self.AE.cube[0][0]) if abs(
            #self.AE.cube[0][1] - self.AE.cube[0][0]) > 0 else 0)
        #kyg = (self.AE.cube[1][2] * self.AE.dk / (self.AE.cube[1][1] - self.AE.cube[1][0]) if abs(
            #self.AE.cube[1][1] - self.AE.cube[1][0]) > 0 else 0)
        #wg = (self.AE.cube[2][2] * self.AE.dE / (self.AE.cube[2][1] - self.AE.cube[2][0]) if abs(
            #self.AE.cube[2][1] - self.AE.cube[2][0]) > 0 else 0)

        # For the option of simulating noise on ARPES data
        #if noise:
            # Added by Ziqin Yue 09/27/2022
            '''
            Convolute poisson noise before Gaussian brodening
            '''
            #Imax = np.max(I)
            # noisemap = np.ones((self.cube[1][2],self.cube[0][2],self.cube[2][2]))
            # Ip = I+np.random.poisson(noisemap)*Imax
            #Ip = np.random.poisson(I / Imax) / Imax
            # Ip = Ip/np.max(Ip)
            #Ig = nd.gaussian_filter(Ip, (kyg, kxg, wg)) # Gaussian convolution to mimic ARPES instrumental resolution limit

            '''
            Add Gaussian noise to represent circuit noise
            '''
            #Ig = Ig + np.random.normal(0, 0.1, size=(self.AE.cube[1][2], self.AE.cube[0][2], self.AE.cube[2][2])) / Imax * NSR
            #Igmax = np.max(Ig)
            #Ig = Ig / Igmax  # Normalize the data
            #Ig = Ig * (np.sign(Ig) + 1.) / 2  # Remove the negative part of the spectra (unrealistic), i.e. keeping weight of data between 0 and 1.
        #else:
            #Ig = nd.gaussian_filter(I, (kyg, kxg, wg)) # Only an instrumetnal resolution limit filter
            #Igmax = np.max(Ig)
            #Ig = Ig / Igmax
            #Ig = Ig * (np.sign(Ig) + 1.) / 2  # No Poison and circuit Gausian noise if noise=False

        I = np.array(I, dtype="float32")

        if slice_select != None:
            ax_img = self.plot_band_structure(I, slice_select, plot_bands, ax, colourmap=colourmap, noise=noise, NSR=NSR)

        if add_map:
            self.AE.maps.append(
                imap.intensity_map(len(self.AE.maps), I, self.AE.cube, self.AE.kz, self.AE.T, self.AE.hv, self.AE.pol, self.AE.dE, self.AE.dk,
                                    self.AE.SE_args, self.AE.sarpes, self.AE.ang))
        else:
            self.AE.maps = [
                imap.intensity_map(len(self.AE.maps), I, self.AE.cube, self.AE.kz, self.AE.T, self.AE.hv, self.AE.pol, self.AE.dE, self.AE.dk,
                                    self.AE.SE_args, self.AE.sarpes, self.AE.ang)]
        if slice_select:
            return I, ax_img
        else:
            return I

    # Function providing the 3D ARPES datacube and/or the selected 2D slice plot without computing matrix elements.
    def ARPESraw_cube(self, ARPES_dict=None, slice_select=None, add_map=False, plot_bands=False, ax=None, colourmap=None, noise=False, NSR=0.8):
        if not hasattr(self.AE, 'pks'):
            self.TB_diagonalize()

        if ARPES_dict is not None:
            self.AE.update_pars(ARPES_dict)

        if colourmap is None:
            colourmap = cm.Blues

        SE = self.AE.SE_gen()
        fermi = self.AE.T_distribution()
        w = np.linspace(*self.AE.cube[2])

        I = np.zeros((self.AE.cube[1][2], self.AE.cube[0][2], self.AE.cube[2][2]))

        if np.shape(SE) == np.shape(I):
            SE_k = True
        else:
            SE_k = False

        for p in range(len(self.AE.pks)):
            if not SE_k:
                I[int(np.real(self.AE.pks[p, 1])), int(np.real(self.AE.pks[p, 2])), :] += np.imag(
                    -1. / (np.pi * (w - self.AE.pks[p, 3] - (SE - 0.00005j)))) * fermi
            else:
                I[int(np.real(self.AE.pks[p, 1])), int(np.real(self.AE.pks[p, 2])), :] += np.imag(-1. / (
                        np.pi * (w - self.AE.pks[p, 3] - (
                            SE[int(np.real(self.AE.pks[p, 1])), int(np.real(self.AE.pks[p, 2])), :] - 0.00005j)))) * fermi

        #kxg = (self.AE.cube[0][2] * self.AE.dk / (self.AE.cube[0][1] - self.AE.cube[0][0]) if abs(
            #self.AE.cube[0][1] - self.AE.cube[0][0]) > 0 else 0)
        #kyg = (self.AE.cube[1][2] * self.AE.dk / (self.AE.cube[1][1] - self.AE.cube[1][0]) if abs(
            #self.AE.cube[1][1] - self.AE.cube[1][0]) > 0 else 0)
        #wg = (self.AE.cube[2][2] * self.AE.dE / (self.AE.cube[2][1] - self.AE.cube[2][0]) if abs(
            #self.AE.cube[2][1] - self.AE.cube[2][0]) > 0 else 0)

        # For the option of simulating noise on ARPES data
        #if noise:
            # Added by Ziqin Yue 09/27/2022
            '''
            Convolute poisson noise before Gaussian brodening
            '''
            #Imax = np.max(I)
            # noisemap = np.ones((self.cube[1][2],self.cube[0][2],self.cube[2][2]))
            # Ip = I+np.random.poisson(noisemap)*Imax
            #Ip = np.random.poisson(I / Imax) / Imax
            # Ip = Ip/np.max(Ip)
            #Ig = nd.gaussian_filter(Ip, (kyg, kxg, wg)) # Gaussian convolution to mimic ARPES instrumental resolution limit

            '''
            Add Gaussian noise to represent circuit noise
            '''
            #Ig = Ig + np.random.normal(0, 0.1, size=(self.AE.cube[1][2], self.AE.cube[0][2], self.AE.cube[2][2])) / Imax * NSR
            #Igmax = np.max(Ig)
            #Ig = Ig / Igmax  # Normalize the data
            #Ig = Ig * (np.sign(Ig) + 1.) / 2  # Remove the negative part of the spectra (unrealistic), i.e. keeping weight of data between 0 and 1.
        #else:
            #Ig = nd.gaussian_filter(I, (kyg, kxg, wg)) # Only an instrumetnal resolution limit filter
            #Igmax = np.max(Ig)
            #Ig = Ig / Igmax
            #Ig = Ig * (np.sign(Ig) + 1.) / 2  # No Poison and circuit Gausian noise if noise=False

        I = np.array(I, dtype="float32")

        if slice_select != None:
            ax_img = self.plot_band_structure(I, slice_select, plot_bands, ax, colourmap=colourmap, noise=noise, NSR=NSR)

        if add_map:
            self.AE.maps.append(
                imap.intensity_map(len(self.AE.maps), I, self.AE.cube, self.AE.kz, self.AE.T, self.AE.hv, self.AE.pol, self.AE.dE, self.AE.dk,
                                    self.AE.SE_args, self.AE.sarpes, self.AE.ang))
        else:
            self.AE.maps = [
                imap.intensity_map(len(self.AE.maps), I, self.AE.cube, self.AE.kz, self.AE.T, self.AE.hv, self.AE.pol, self.AE.dE, self.AE.dk,
                                    self.AE.SE_args, self.AE.sarpes, self.AE.ang)]
        if slice_select:
            return I, ax_img
        else:
            return I

    # Extracting a 2D array from a 3D datacube array
    def extract2D(self, datacube, slice_select, cal_type, NSR, noise=False):
        if type(slice_select[0]) is str:
            str_opts = [['x', 'kx'], ['y', 'ky'], ['energy', 'w', 'e']]
            dim = 0
            for i in range(3):
                if slice_select[0].lower() in str_opts[i]:
                    dim = i
            x = np.linspace(*self.AE.cube[dim])
            index = np.where(abs(x - slice_select[1]) == abs(x - slice_select[1]).min())[0][0]
            slice_select = [dim, int(index)]

        limits = np.zeros((3, 2), dtype=int)
        limits[:, 1] = np.shape(datacube)[1], np.shape(datacube)[0], np.shape(datacube)[2]
        limits[slice_select[0]] = [slice_select[1], slice_select[1] + 1]

        plottable = np.squeeze(datacube[limits[1, 0]:limits[1, 1], limits[0, 0]:limits[0, 1], limits[2, 0]:limits[2, 1]])
        I = plottable

        wg = (self.AE.cube[2][2] * self.AE.dE / (self.AE.cube[2][1] - self.AE.cube[2][0]) if abs(
            self.AE.cube[2][1] - self.AE.cube[2][0]) > 0 else 0)
        kg = (self.AE.cube[1][2] * self.AE.dk / (self.AE.cube[1][1] - self.AE.cube[1][0]) if abs(
            self.AE.cube[1][1] - self.AE.cube[1][0]) > 0 else 0)

        # For the option of simulating noise on ARPES data
        if noise:
            # Added by Ziqin Yue 09/27/2022
            '''
            Convolute poisson noise before Gaussian brodening
            '''
            Imax = np.max(I)
            # noisemap = np.ones((self.cube[1][2],self.cube[0][2],self.cube[2][2]))
            # Ip = I+np.random.poisson(noisemap)*Imax
            Ip = np.random.poisson(I / Imax) / Imax
            # Ip = Ip/np.max(Ip)
            Ig = nd.gaussian_filter(Ip,
                                    (kg, wg))  # Gaussian convolution to mimic ARPES instrumental resolution limit

            '''
            Add Gaussian noise to represent circuit noise
            '''
            Ig = Ig + np.random.normal(0, 0.1,
                                       size=(self.AE.cube[0][2], self.AE.cube[2][2])) / Imax * NSR
            Igmax = np.max(Ig)
            Ig = Ig / Igmax  # Normalize the data
            Ig = Ig * (np.sign(
                Ig) + 1.) / 2  # Remove the negative part of the spectra (unrealistic), i.e. keeping weight of data between 0 and 1.
        else:
            Ig = nd.gaussian_filter(I, (kg, wg))  # Only an instrumetnal resolution limit filter
            Igmax = np.max(Ig)
            Ig = Ig / Igmax
            Ig = Ig * (np.sign(Ig) + 1.) / 2  # No Poison and circuit Gausian noise if noise=False

        if cal_type == 'TB':
            return plottable
        elif cal_type == 'ARPES':
            return Ig
        else:
            raise Exception('Please use TB or ARPES as input for cal_type')

    # Modified from Chinook self.AE.plot_intensity_map to avoid ripple effect from convolving Gaussian filter on the 3D array
    def plot_band_structure(self, plot_map, slice_select, NSR, plot_bands=False, ax_img=None, colourmap=None, noise=False):
        if colourmap is None:
            colourmap = cm.magma

        if ax_img is None:
            fig, ax_img = plt.subplots()
            fig.set_tight_layout(False)

        if type(slice_select[0]) is str:
            str_opts = [['x', 'kx'], ['y', 'ky'], ['energy', 'w', 'e']]
            dim = 0
            for i in range(3):
                if slice_select[0].lower() in str_opts[i]:
                    dim = i
            x = np.linspace(*self.AE.cube[dim])
            index = np.where(abs(x - slice_select[1]) == abs(x - slice_select[1]).min())[0][0]
            slice_select = [dim, int(index)]

            # new option
        index_dict = {2: (0, 1), 1: (2, 0), 0: (2, 1)}

        X, Y = np.meshgrid(np.linspace(*self.AE.cube[index_dict[slice_select[0]][0]]),
                           np.linspace(*self.AE.cube[index_dict[slice_select[0]][1]]))
        limits = np.zeros((3, 2), dtype=int)
        limits[:, 1] = np.shape(plot_map)[1], np.shape(plot_map)[0], np.shape(plot_map)[2]
        limits[slice_select[0]] = [slice_select[1], slice_select[1] + 1]

        ax_xlimit = (self.AE.cube[index_dict[slice_select[0]][0]][0], self.AE.cube[index_dict[slice_select[0]][0]][1])
        ax_ylimit = (self.AE.cube[index_dict[slice_select[0]][1]][0], self.AE.cube[index_dict[slice_select[0]][1]][1])
        plottable = np.squeeze(
            plot_map[limits[1, 0]:limits[1, 1], limits[0, 0]:limits[0, 1], limits[2, 0]:limits[2, 1]])
        I = plottable

        wg = (self.AE.cube[2][2] * self.AE.dE / (self.AE.cube[2][1] - self.AE.cube[2][0]) if abs(
            self.AE.cube[2][1] - self.AE.cube[2][0]) > 0 else 0)
        kg = (self.AE.cube[1][2] * self.AE.dk / (self.AE.cube[1][1] - self.AE.cube[1][0]) if abs(
            self.AE.cube[1][1] - self.AE.cube[1][0]) > 0 else 0)

        # For the option of simulating noise on ARPES data
        if noise:
            # Added by Ziqin Yue 09/27/2022
            '''
            Convolute poisson noise before Gaussian brodening
            '''
            Imax = np.max(I)
            # noisemap = np.ones((self.cube[1][2],self.cube[0][2],self.cube[2][2]))
            # Ip = I+np.random.poisson(noisemap)*Imax
            Ip = np.random.poisson(I / Imax) / Imax
            # Ip = Ip/np.max(Ip)
            Ig = nd.gaussian_filter(Ip,
                                    (kg, wg))  # Gaussian convolution to mimic ARPES instrumental resolution limit

            '''
            Add Gaussian noise to represent circuit noise
            '''
            Ig = Ig + np.random.normal(0, 0.1,
                                       size=(self.AE.cube[0][2], self.AE.cube[2][2])) / Imax * NSR
            Igmax = np.max(Ig)
            Ig = Ig / Igmax  # Normalize the data
            Ig = Ig * (np.sign(
                Ig) + 1.) / 2  # Remove the negative part of the spectra (unrealistic), i.e. keeping weight of data between 0 and 1.
        else:
            Ig = nd.gaussian_filter(I, (kg, wg))  # Only an instrumetnal resolution limit filter
            Igmax = np.max(Ig)
            Ig = Ig / Igmax
            Ig = Ig * (np.sign(Ig) + 1.) / 2  # No Poison and circuit Gausian noise if noise=False

        p = ax_img.pcolormesh(X, Y, Ig, cmap=colourmap)
        if plot_bands and slice_select[0] != 2:
            k = np.linspace(*self.cube[index_dict[slice_select[0]][1]])
            if slice_select[0] == 1:
                indices = np.array([len(k) * slice_select[1] + ii for ii in range(len(k))])
            elif slice_select[0] == 0:
                indices = np.array([slice_select[1] + ii * self.cube[0][2] for ii in range(len(k))])
            for ii in range(len(self.TB.basis)):
                ax_img.plot(self.TB.Eband[indices, ii], k, alpha=0.4, c='w')
        elif plot_bands and slice_select[0] == 2:
            for ii in range(len(self.TB.basis)):
                if self.TB.Eband[:, ii].min() <= x[slice_select[1]] and self.TB.Eband[:, ii].max() >= x[
                    slice_select[1]]:
                    reshape = self.TB.Eband[:, ii].reshape(np.shape(X))
                    ax_img.contour(X, Y, reshape, levels=[x[slice_select[1]]], colors='w', alpha=0.2)

        #
        ax_img.set_xlim(*ax_xlimit)
        ax_img.set_ylim(*ax_ylimit)

        plt.colorbar(p, ax=ax_img)
        plt.tight_layout()

        return ax_img

    # Generating compressed data files. 2D slices are selected from either the 3D TB or the 3D ARPES datacube.
    def npz_from_cube(self, datacube, N, y_start, y_end, cal_type):
        if y_start<self.AE.cube[1][0] or y_end>self.AE.cube[1][1]:
            raise Exception('Slice out of the cube range')

        Data_set = []
        if cal_type == 'TB':
            for i in range(N):
                TB_arr = self.extract2D(datacube, slice_select=('ky', y_start + i * np.abs(y_end-y_start)/(N-1)))
                Data_set.append(TB_arr)
            np.savez_compressed('TB_sim_0_' + str(N-1), *Data_set[:N])
        elif cal_type == 'ARPES':
            for i in range(N):
                ARPES_arr = self.extract2D(datacube, slice_select=('ky', y_start + i * np.abs(y_end-y_start)/(N-1)))
                Data_set.append(ARPES_arr)
            np.savez_compressed('ARPES_sim_0_' + str(N-1), *Data_set[:N])
        else:
            raise Exception('Please use TB or ARPES as input for cal_type')

    # Generating compressed npz file of the TB or ARPES datacube.
    def npz_cube(self, datacube, cal_type, i):
        if cal_type == 'TB':
            TB_cube = self.TB_cube()
            np.savez_compressed('TB_cube_' + str(i), TB_cube)
        elif cal_type == 'ARPES':
            ARPES_cube = self.ARPES_cube()
            np.savez_compressed('ARPES_cube_' + str(i), ARPES_cube)
        else:
            raise Exception('Please use TB or ARPES as input for cal_type')

    # Generating 2D npy files. 2D slices are selected from either the 3D TB or the 3D ARPES datacube.
    def npy_from_cube(self, datacube, cube_idx, N, y_start, y_end, cal_type, path, noise=False, NSR=0.8):
        if y_start<self.AE.cube[1][0] or y_end>self.AE.cube[1][1]:
            raise Exception('Slice out of the cube range')

        if cal_type == 'TB':
            makedir(path + '/TB_simulation')
            for i in range(N):
                TB_arr = self.extract2D(datacube, slice_select=('ky', y_start + i * np.abs(y_end-y_start)/(N-1)), cal_type=cal_type, NSR=NSR, noise=False)
                np.save("%s/TB_simulation/TB_sim_%d_%d" %(path, cube_idx, i), TB_arr)

        elif cal_type == 'ARPES':
            makedir(path + '/ARPES_simulation')
            for i in range(N):
                ARPES_arr = self.extract2D(datacube, slice_select=('ky', y_start + i * np.abs(y_end-y_start)/(N-1)), cal_type=cal_type, NSR=NSR, noise=noise)
                np.save("%s/ARPES_simulation/ARPES_sim_%d_%d" %(path, cube_idx, i), ARPES_arr)

        else:
            raise Exception('Please use TB or ARPES as input for cal_type')

def makedir(path):
    try:
        os.mkdir(path)
    except:
        pass


