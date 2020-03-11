import numpy as np
import pandas as pd
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splrep, splprep, splev, BSpline
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN
import pydicom
from math import *
from tkinter import *
from tkinter import ttk
from tkinter import filedialog


# For BSPLINES
import json, os
import scipy.linalg
from time import time

# FROM https://github.com/rstebbing/bspline-regression
# from uniform_bspline import UniformBSpline
# from util import raise_if_not_shape


"""
def findSpline(xy):
    SOLVER_TYPES = frozenset(['dn', 'lm'])

    degree = 3
    num_control_points = 30
    dim = 2
    is_closed = True

    c = UniformBSpline(degree, num_control_points, dim, is_closed)
"""    

class MainMenu(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.parent.protocol("WM_DELETE_WINDOW", self.myQuit)
        self.parent.title("RT structure generator")

        BUTTON_WIDTH = 30
        TEXT_WIDTH = 10

        self.dicomVolume = dict() # { z : np.array } REDUCED to conform to xyz boundaries
        self.transformation = None
        self.ds = None
        
        self.topContainer = Frame(self, borderwidth=10)
        self.midContainer = Frame(self, borderwidth=10)
        self.bottomContainer = Frame(self, borderwidth=10)

        self.bottomContainer.pack(anchor=S, side=BOTTOM,fill=X, expand=1)
        self.topContainer.pack(fill=X, expand=1)
        self.midContainer.pack(fill=X, expand=1)

        self.useRelativePosContainer = Frame(self.midContainer)
        self.useClosedCurveLimiterContainer = Frame(self.midContainer)
        self.posEntryContainer = Frame(self.midContainer)
        self.deltaEntryContainer = Frame(self.midContainer)
        self.angleEntryContainer = Frame(self.midContainer)
        self.DBSCANepsContainer = Frame(self.midContainer)
        self.DBSCANMinPtsContainer = Frame(self.midContainer)

        self.useRelativePosContainer.pack(fill=X)
        self.useClosedCurveLimiterContainer.pack(fill=X)
        self.posEntryContainer.pack(fill=X)
        self.deltaEntryContainer.pack(fill=X)
        self.angleEntryContainer.pack(fill=X)
        
        self.useRelativePosVar = IntVar(value=0)
        self.useClosedCurveLimiterVar = IntVar(value=1)
        self.zPosVar = DoubleVar(value=0)
        self.zDeltaVar = DoubleVar(value=0.5)
        self.angleVar = DoubleVar(value=0)
        
        self.useRelativePosCheck = Checkbutton(self.useRelativePosContainer,
                                               text="Use relative positions? ", variable=self.useRelativePosVar)
        self.useRelativePosCheck.pack(anchor=W)

        Label(self.posEntryContainer, text="Slice position (z): ").pack(anchor=W)
        Entry(self.posEntryContainer, textvariable=self.zPosVar, width=TEXT_WIDTH).pack(side=LEFT)

        Label(self.deltaEntryContainer, text="Slice thickness (z): ").pack(anchor=W)
        Entry(self.deltaEntryContainer, textvariable=self.zDeltaVar, width=TEXT_WIDTH).pack(side=LEFT)

        Label(self.angleEntryContainer, text="Tilt angle (x-axis, ±180 deg): ").pack(anchor=W)
        Entry(self.angleEntryContainer, textvariable=self.angleVar, width=TEXT_WIDTH).pack(side=LEFT)

        self.buttonLoadCSV = Button(self.bottomContainer, text="Load CSV mesh (C)",
                                    command=self.commandLoadCSVMesh, width=BUTTON_WIDTH)
        self.buttonLoadCSV.pack(anchor=S, side=LEFT)
        
        self.buttonMakePlots = Button(self.bottomContainer, text="Make plots (Enter)",
                                      command=self.commandMakePlots, width=BUTTON_WIDTH)
        self.buttonMakePlots.pack(side=LEFT)
        
        self.buttonLoadDICOM = Button(self.bottomContainer, text="Load DICOM series (D)",
                                      command=self.commandLoadDICOM, width=BUTTON_WIDTH)
        self.buttonLoadDICOM.pack(side=LEFT)

        self.buttonQuit = Button(self.bottomContainer, text="Exit (Esc)",
                                 command=self.myQuit, width=BUTTON_WIDTH)
        self.buttonQuit.pack(side=LEFT)

        self.parent.bind("c", lambda event=None: self.buttonLoadCSV.invoke())
        self.parent.bind("d", lambda event=None: self.buttonLoadDICOM.invoke())
        self.parent.bind("<Return>", lambda event=None: self.buttonMakePlots.invoke())
        self.parent.bind("<Escape>", lambda event=None: self.buttonQuit.invoke())

        df = pd.read_csv("meshCSV/zz010440HUH44_1.2.826.0.1.3680043.2.968.3.8323329.27824.1530802493.162.csv")

        #x = df['FractionMeshPointsX'].values
        #y = df['FractionMeshPointsY'].values
        #z = df['FractionMeshPointsZ'].values

        x = df['PlanningMeshPointsX'].values
        y = df['PlanningMeshPointsY'].values
        z = df['PlanningMeshPointsZ'].values

        self.x0 = np.mean(x)
        self.y0 = np.mean(y)
        self.z0 = np.mean(z)

        x = x - self.x0
        y = y - self.y0
        z = z - self.z0

        self.xyz = np.array(list(zip(x,y,z)))

        self.pack()
        self.commandMakePlotsPlaceHolder()

    def myQuit(self):
        self.parent.destroy()
        plt.close("all")
        self.quit()

    def commandLoadCSVMesh(self):
        newfile = filedialog.askopenfilename(initialdir="meshCSV/")
        df = pd.read_csv(newfile)

        x = df['FractionMeshPointsX'].values
        y = df['FractionMeshPointsY'].values
        z = df['FractionMeshPointsZ'].values

        self.x0 = np.mean(x)
        self.y0 = np.mean(y)
        self.z0 = np.mean(z)

        x = x - self.x0
        y = y - self.y0
        z = z - self.z0

        self.xyz = np.array(list(zip(x,y,z)))

    def commandLoadDICOM(self):
        newfile = filedialog.askdirectory(initialdir="DICOM/")
        #boundaries = [[50*floor(np.min(self.xyz[:,k])/50), 50*ceil(np.max(self.xyz[:,k])/50)] for k in range(2)]
        boundariesMM = [[int(np.min(self.xyz[:,k]))-50, int(np.max(self.xyz[:,k]))+50] for k in range(2)]
        
        self.ds = None
        for root, dirs, files in os.walk(newfile):
            for file in files:
                if not "CT" in file: continue

                ds = pydicom.dcmread(f"{root}/{file}")
                if not self.ds:
                    self.ds = ds
                img = ds.pixel_array + ds.RescaleIntercept
                xy0 = [float(k) for k in ds.ImagePositionPatient] # Upper left corner
                dxy = float(ds.PixelSpacing[0])

                print("xy0", xy0)
                
                boundaries = [0,0]
                for dim in range(2):
                    boundaries[dim] = [int((boundariesMM[dim][0] - xy0[dim])/dxy), int((boundariesMM[dim][1] - xy0[dim])/dxy)]

                print("boundaries", boundaries)
                    
                self.dicomVolume[int(ds.SliceLocation)] = img[boundaries[1][0]:boundaries[1][1], boundaries[0][0]:boundaries[0][1]]

        """
        zList = np.array(zList)
        zList -= self.zPosVar.get()
        zListAbs = np.absolute(zList)
        zListAbsMin = zListAbs.argmin()

        for root, dirs, files in os.walk(newfile):
            file = files[zListAbsMin]
            if file:
                print(f"Choosing file {file} with z = {zList[zListAbsMin]} (want {self.zPosVar.get()})")
                self.ds = pydicom.dcmread(f"{root}/{file}")
                break

        self.img = self.ds.pixel_array + self.ds.RescaleIntercept
        """
        
        self.commandMakePlots()

    def commandMakePlotsPlaceHolder(self):
        self.fig, self.axs = plt.subplots(1,4, figsize=(20,5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.topContainer)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        self.canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=1)

        def onclick(event):
            self.zPosVar.set(event.ydata)
            self.commandMakePlots()
            
        def onscroll(event):
            self.angleVar.set(self.angleVar.get() + 5 * event.step)

            # New default slice position to account of rotation
            r = Rotation.from_euler("x", self.angleVar.get(), degrees=True)
            xyz = r.apply(self.xyz)
            self.zPosVar.set(np.mean(xyz[:,2]))
            
            self.commandMakePlots()
            
        cid = self.canvas.mpl_connect('button_press_event', onclick)
        cid2 = self.canvas.mpl_connect('scroll_event', onscroll)
    
    def commandMakePlots(self):
        for ax in self.axs:
            ax.clear()
        self.makePlot(self.axs)
        self.canvas.draw()

    def makePlot(self, axs):
        xyz = self.xyz.copy()
        
        if self.angleVar.get():
            r = Rotation.from_euler("x", self.angleVar.get(), degrees=True)
            xyz = r.apply(xyz)
            
        x = xyz[:,0]
        y = xyz[:,1]
        z = xyz[:,2]

        zPos = self.zPosVar.get()
        zDelta = self.zDeltaVar.get()

        zFilter1 = zPos - zDelta/2 < z
        zFilter2 = z < zPos + zDelta/2

        xFilter = x[zFilter1 & zFilter2]
        yFilter = y[zFilter1 & zFilter2]

        # SORT by angle
        xDC = xFilter - np.mean(xFilter)
        yDC = yFilter - np.mean(yFilter)
        angles = np.zeros(np.shape(xFilter), dtype=np.float64)

        for idx in range(len(xFilter)):
            angles[idx] = atan2(yDC[idx], xDC[idx])

        angleSortIdx = np.argsort(angles)
        anglesSorted = angles[angleSortIdx]
        xDCSort = xDC[angleSortIdx]
        yDCSort = yDC[angleSortIdx]

        xSort = xDCSort + np.mean(xFilter)
        ySort = yDCSort + np.mean(yFilter)

        step = pi/45
        anglesAveraged = np.arange(-pi, pi+step, step)
        
        xAveraged = np.zeros(np.shape(anglesAveraged))
        yAveraged = np.zeros(np.shape(anglesAveraged))
        idx = 0
        
        for angle in anglesAveraged:
            lowerFilter = anglesSorted > angle - step/2
            upperFilter = anglesSorted < angle + step/2
            
            if np.sum(xDCSort[lowerFilter & upperFilter]) == 0:
                continue

            xDCSortFilter = xDCSort[lowerFilter & upperFilter]
            yDCSortFilter = yDCSort[lowerFilter & upperFilter]
            rAngleMax = np.max(np.sqrt(xDCSortFilter**2 + yDCSortFilter**2))
            
            if isnan(rAngleMax):
                continue
            
            xAveraged[idx] = rAngleMax * cos(angle)
            yAveraged[idx] = rAngleMax * sin(angle)
            idx += 1

        xAveraged = xAveraged[:idx] + np.mean(xFilter)
        yAveraged = yAveraged[:idx] + np.mean(yFilter)
        
        xlabel = "x - x0 [mm]"
        ylabel = "y - y0 [mm]"
        zlabel = "z - z0 [mm]"

        if not self.useRelativePosVar.get():
            x += self.x0
            y += self.y0
            z += self.z0
            xSort += self.x0
            ySort += self.y0
            xAveraged += self.x0
            yAveraged += self.y0
            zPos += self.z0
            
            xlabel = "x [mm]"
            ylabel = "y [mm]"
            zlabel = "z [mm]"


        xAveragedExtra = np.append(xAveraged, xAveraged[0])
        yAveragedExtra = np.append(yAveraged, yAveraged[0])

        axs[0].scatter(x,y,alpha=0.1,marker=".")
        axs[0].plot(xAveragedExtra, yAveragedExtra, 'r-')
        axs[0].set_xlabel(xlabel)
        axs[0].set_ylabel(ylabel)
        axs[0].set_title(f"All points XY, rotated {self.angleVar.get()}° about X")

        axs[1].scatter(x,z,alpha=0.1,marker=".")
        axs[1].plot(axs[1].get_xlim(), [zPos, zPos], 'r-')
        axs[1].set_xlabel(xlabel)
        axs[1].set_ylabel(zlabel)
        axs[1].set_title(f"All points XZ, rotated {self.angleVar.get()}° about X \n(click to set Z)")
    
        axs[2].scatter(xSort, ySort, marker=",", label="All points")
        axs[2].plot(xAveragedExtra, yAveragedExtra, "r-", label="Angular average")
        
        axs[2].set_title(f"z = {zPos:.1f} ± {zDelta:.1f} mm with angular average")
        axs[2].set_xlabel(xlabel)
        axs[2].set_ylabel(ylabel)

        if np.sum(self.dicomVolume):
            zList = list(self.dicomVolume.keys())
            zList2 = [k-self.zPosVar.get() for k in zList]
            zListAbs = np.absolute(zList2)
            zToUse = zList[zListAbs.argmin()]

            img = self.dicomVolume[zToUse]

            extent = np.shape(img)[0] * self.ds.PixelSpacing[0]
            xiso = self.ds.ImagePositionPatient[0]
            yiso = self.ds.ImagePositionPatient[1]
            axs[3].imshow(img, cmap="gray")#, extent=(xiso, xiso+extent, extent+yiso, yiso))
            axs[3].plot(xAveragedExtra, yAveragedExtra, "r-", label="Angular average")

root = Tk()
mainmenu = MainMenu(root)
root.mainloop()


# ROTTEN CODE BUT I DON'T WANT TO DELETE

"""
xyDC = np.array(list(zip(xDCSort, yDCSort)))
db = DBSCAN(eps=self.DBSCANepsVar.get(), min_samples=self.DBSCANMinPtsVar.get()).fit(xyDC)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = xyDC[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=5)

    xy = xyDC[class_member_mask & ~core_samples_mask]
    axs[3].plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=2)

axs[3].set_title('Estimated number of clusters: %d' % n_clusters_)


xyDC = np.array(list(zip(xDCSort, yDCSort)))
# spline = findSpline(xyDC)

# B Spline fitting with a low number of knots
weights=None # Use these to select one of the detected areas
knots = np.linspace( .02, .98, 25 )
taskLeastSquaresSpline = -1
# SPRPREP crashes due to errors in fitpack
# u: values of the parameters [0,1]
# tck: knots (t); Bspline coefficients (c), degree (k)
# tck, u = splprep([xDCSort, yDCSort], w=weights, t=knots, task=taskLeastSquaresSpline)
# print(tck, u)

points = np.linspace(0, 1, len(xDCSort))
parameter = dict()
for d in [0,1]:
    tck = splrep(points, xyDC[:,d], w=weights, t=knots, task=taskLeastSquaresSpline, k=3)
    parameter[d] = tck[1]

parameter = list(parameter.values())
tck = np.array([knots, parameter, 3])
print("tck", tck)
spline = splev(points, tck)

axs[3].scatter(xSort, ySort, marker=",", label="All points")
axs[3].plot(spline[0], spline[1], "r", marker=",", label="Spline")
axs[3].set_title("z = {zPos} ± {zDelta} mm with B-spline regression")
axs[3].set_xlim([-30, 30])
axs[3].set_ylim([-30,30])

#if self.useClosedCurveLimiterVar.get():
    #axs[3].plot(anglesAveraged, sdList)
    #axs[3].set_xlabel("Angle")
    #axs[3].set_ylabel("radial SD")
"""
