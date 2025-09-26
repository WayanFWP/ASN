import numpy as np
import pandas as pd
from Utils import *
from Plot import *

class Coeficient:
    def __init__(self, fs=None):
        self.fs = fs
        self.original_fs = fs
        self.qj = np.zeros((6,10000))
        self.a = None
        self.b = None
        print(f"Initialized Coeficient with sampling rate: {self.fs} Hz")
        print(f"Original sampling rate set to: {self.original_fs} Hz")
    
    def getAnBvalues(self, j):
        self.a = -(round(2**j) + round(2**(j-1)) - 2)
        self.b =-(1-round(2**(j-1))) +1
        print(f"Computed a: {self.a}, b: {self.b} for j: {j}")
        return self.a, self.b
        
    def initialize_qj_filter(self):
        # Placeholder for filter initialization logic
        print("QJ filter initialized.")
        j = 1
        a , b = self.getAnBvalues(j)
        k_index1 = []
        for k in range(a,b):
            k_index1.append(k)
            self.qj[1][k+abs(a)] = -2 * (dirac(k) - dirac(k+1))
        print(f"Filter coefficients for j={j}: {self.qj[1][k_index1]}")
        
        j = 2
        a , b = self.getAnBvalues(j)
        k_index2 = []
        for k in range(a,b):
            k_index2.append(k)
            self.qj[2][k+abs(a)] = -1/4 * (dirac(k-1) + 3*dirac(k) + 2*dirac(k+1) - 2*dirac(k+2) - 3*dirac(k+3) - dirac(k+4))
        print(f"Filter coefficients for j={j}: {self.qj[2][k_index2]}")
        
        j = 3
        a , b = self.getAnBvalues(j)
        k_index3 = []
        for k in range(a,b):
            k_index3.append(k)
            self.qj[3][k+abs(a)] = -1/32 * (dirac(k-3) + 3*dirac(k-2) + 6*dirac(k-1) + 10*dirac(k) + 11*dirac(k+1) + 9*dirac(k+2) + 4*dirac(k+3) - 4*dirac(k+4) - 9*dirac(k+5) - 11*dirac(k+6) - 10*dirac(k+7) - 6*dirac(k+8) - 3*dirac(k+9) - dirac(k+10))
        print(f"Filter coefficients for j={j}: {self.qj[3][k_index3]}")        
        
        j = 4
        a , b = self.getAnBvalues(j)
        k_index4 = []
        for k in range(a,b):
            k_index4.append(k)
            self.qj[4][k+abs(a)] = -1/256 * (dirac(k-7) + 3*dirac(k-6) + 6*dirac(k-5) + 10*dirac(k-4) + 15*dirac(k-3) + 21*dirac(k-2) + 28*dirac(k-1) + 36*dirac(k) + 37*dirac(k+1) + 34*dirac(k+2) + 27*dirac(k+3) + 18*dirac(k+4) + 10*dirac(k+5) + 4*dirac(k+6) - 4*dirac(k+7) - 10*dirac(k+8) - 18*dirac(k+9) - 27*dirac(k+10) - 34*dirac(k+11) - 37*dirac(k+12) - 36*dirac(k+13) - 28*dirac(k+14) - 21*dirac(k+15) - 15*dirac(k+16) - 10*dirac(k+17) - 6*dirac(k+18) - 3*dirac(k+19) - dirac(k+20))
        print(f"Filter coefficients for j={j}: {self.qj[4][k_index4]}")
        
        plotRow(
            x=[k_index1, k_index2, k_index3, k_index4],
            y=[self.qj[1][0:len(k_index1)], self.qj[2][0:len(k_index2)], self.qj[3][0:len(k_index3)], self.qj[4][0:len(k_index4)]],
            plot_type="bar",
            title="QJ Filter Coefficients",
            xlabel="k",
            ylabel="Coefficient Value"
        )   
        
