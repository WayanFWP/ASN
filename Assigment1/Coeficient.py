import numpy as np
import matplotlib.pyplot as plt
from Utils import *

class Coeficient:
    def __init__(self, fs=None):
        self.fs = fs
        self.original_fs = fs
        self.qj = np.zeros((9,1000))
        self.a = None
        self.b = None
        self.frequency_responses = {}
        
        self.g_coeffs = np.array([0, 0, -2, 2])
        self.h_coeffs = np.array([0, 1/8, 3/8, 3/8])
        self.n_range = np.arange(-2, 2)

        
    def getAnBvalues(self, j):
        self.a = -(round(2**j) + round(2**(j-1)) - 2)
        self.b =-(1-round(2**(j-1))) +1
        print(f"Computed a: {self.a}, b: {self.b} for j: {j}")
        return self.a, self.b
        
    def initialize_qj_filter(self):
        print("Initializing QJ filters using frequency domain approach...")
        
        j = 1
        a , b = self.getAnBvalues(j)
        k_index1 = []
        for k in range(a,b):
            k_index1.append(k)
            self.qj[1][k+abs(a)] = -2 * (dirac(k) - dirac(k+1))
        
        j = 2
        a , b = self.getAnBvalues(j)
        k_index2 = []
        for k in range(a,b):
            k_index2.append(k)
            self.qj[2][k+abs(a)] = -1/4 * (dirac(k-1) + 3*dirac(k) + 2*dirac(k+1) - 2*dirac(k+2) - 3*dirac(k+3) - dirac(k+4))
        
        j = 3
        a , b = self.getAnBvalues(j)
        k_index3 = []
        for k in range(a,b):
            k_index3.append(k)
            self.qj[3][k+abs(a)] = -1/32 * (dirac(k-3) + 3*dirac(k-2) + 6*dirac(k-1) + 10*dirac(k) + 11*dirac(k+1) + 9*dirac(k+2) + 4*dirac(k+3) - 4*dirac(k+4) - 9*dirac(k+5) - 11*dirac(k+6) - 10*dirac(k+7) - 6*dirac(k+8) - 3*dirac(k+9) - dirac(k+10))
        
        j = 4
        a , b = self.getAnBvalues(j)
        k_index4 = []
        for k in range(a,b):
            k_index4.append(k)
            self.qj[4][k+abs(a)] = -1/256 * (dirac(k-7) + 3*dirac(k-6) + 6*dirac(k-5) + 10*dirac(k-4) + 15*dirac(k-3) + 21*dirac(k-2) + 28*dirac(k-1) + 36*dirac(k) + 37*dirac(k+1) + 34*dirac(k+2) + 27*dirac(k+3) + 18*dirac(k+4) + 10*dirac(k+5) + 4*dirac(k+6) - 4*dirac(k+7) - 10*dirac(k+8) - 18*dirac(k+9) - 27*dirac(k+10) - 34*dirac(k+11) - 37*dirac(k+12) - 36*dirac(k+13) - 28*dirac(k+14) - 21*dirac(k+15) - 15*dirac(k+16) - 10*dirac(k+17) - 6*dirac(k+18) - 3*dirac(k+19) - dirac(k+20))
        
        j = 5
        a , b = self.getAnBvalues(j)
        k_index5 = []
        for k in range(a,b):
            k_index5.append(k)
            self.qj[5][k+abs(a)] = -1/2048 * (dirac(k-15) + 3*dirac(k-14) + 6*dirac(k-13) + 10*dirac(k-12) + 15*dirac(k-11) + 21*dirac(k-10) + 28*dirac(k-9) + 36*dirac(k-8) + 45*dirac(k-7) + 55*dirac(k-6) + 66*dirac(k-5) + 78*dirac(k-4) + 91*dirac(k-3) + 105*dirac(k-2) + 120*dirac(k-1) + 136*dirac(k) + 149*dirac(k+1) + 159*dirac(k+2) + 166*dirac(k+3) + 170*dirac(k+4) + 171*dirac(k+5) + 169*dirac(k+6) + 164*dirac(k+7) + 156*dirac(k+8) + 145*dirac(k+9) + 131*dirac(k+10) + 114*dirac(k+11) + 94*dirac(k+12) + 71*dirac(k+13) + 45*dirac(k+14) + 16*dirac(k+15) - 16*dirac(k+16) - 45*dirac(k+17) - 71*dirac(k+18) - 94*dirac(k+19) - 114*dirac(k+20) - 131*dirac(k+21) - 145*dirac(k+22) - 156*dirac(k+23) - 164*dirac(k+24) - 169*dirac(k+25) - 171*dirac(k+26) - 170*dirac(k+27) - 166*dirac(k+28) - 159*dirac(k+29) - 149*dirac(k+30) - 136*dirac(k+31) - 120*dirac(k+32) - 105*dirac(k+33) - 91*dirac(k+34) - 78*dirac(k+35) - 66*dirac(k+36) - 55*dirac(k+37) - 45*dirac(k+38) - 36*dirac(k+39) - 28*dirac(k+40) - 21*dirac(k+41) - 15*dirac(k+42) - 10*dirac(k+43) - 6*dirac(k+44) - 3*dirac(k+45) - dirac(k+46))
        
        j = 6
        a , b = self.getAnBvalues(j)
        k_index6 = []
        for k in range(a,b):
            k_index6.append(k)
            self.qj[6][k+abs(a)] = -1/16384 * (dirac(k-31) + 3*dirac(k-30) + 6*dirac(k-29) + 10*dirac(k-28) + 15*dirac(k-27) + 21*dirac(k-26) + 28*dirac(k-25) + 36*dirac(k-24) + 45*dirac(k-23) + 55*dirac(k-22) + 66*dirac(k-21) + 78*dirac(k-20) + 91*dirac(k-19) + 105*dirac(k-18) + 120*dirac(k-17) + 136*dirac(k-16) + 153*dirac(k-15) + 171*dirac(k-14) + 190*dirac(k-13) + 210*dirac(k-12) + 231*dirac(k-11) + 253*dirac(k-10) + 276*dirac(k-9) + 300*dirac(k-8) + 325*dirac(k-7) + 351*dirac(k-6) + 378*dirac(k-5) + 406*dirac(k-4) + 435*dirac(k-3) + 465*dirac(k-2) + 496*dirac(k-1) + 528*dirac(k) + 557*dirac(k+1) + 583*dirac(k+2) + 606*dirac(k+3) + 626*dirac(k+4) + 643*dirac(k+5) + 657*dirac(k+6) + 668*dirac(k+7) + 676*dirac(k+8) + 681*dirac(k+9) + 683*dirac(k+10) + 682*dirac(k+11) + 678*dirac(k+12) + 671*dirac(k+13) + 661*dirac(k+14) + 648*dirac(k+15) + 632*dirac(k+16) + 613*dirac(k+17) + 591*dirac(k+18) + 566*dirac(k+19) + 538*dirac(k+20) + 507*dirac(k+21) + 473*dirac(k+22) + 436*dirac(k+23) + 396*dirac(k+24) + 353*dirac(k+25) + 307*dirac(k+26) + 258*dirac(k+27) + 206*dirac(k+28) + 151*dirac(k+29) + 93*dirac(k+30) + 32*dirac(k+31) - 32*dirac(k+32) - 93*dirac(k+33) - 151*dirac(k+34) - 206*dirac(k+35) - 258*dirac(k+36) - 307*dirac(k+37) - 353*dirac(k+38) - 396*dirac(k+39) - 436*dirac(k+40) - 473*dirac(k+41) - 507*dirac(k+42) - 538*dirac(k+43) - 566*dirac(k+44) - 591*dirac(k+45) - 613*dirac(k+46) - 632*dirac(k+47) - 648*dirac(k+48) - 661*dirac(k+49) - 671*dirac(k+50) - 678*dirac(k+51) - 682*dirac(k+52) - 683*dirac(k+53) - 681*dirac(k+54) - 676*dirac(k+55) - 668*dirac(k+56) - 657*dirac(k+57) - 643*dirac(k+58) - 626*dirac(k+59) - 606*dirac(k+60) - 583*dirac(k+61) - 557*dirac(k+62) - 528*dirac(k+63) - 496*dirac(k+64) - 465*dirac(k+65) - 435*dirac(k+66) - 406*dirac(k+67) - 378*dirac(k+68) - 351*dirac(k+69) - 325*dirac(k+70) - 300*dirac(k+71) - 276*dirac(k+72) - 253*dirac(k+73) - 231*dirac(k+74) - 210*dirac(k+75) - 190*dirac(k+76) - 171*dirac(k+77) - 153*dirac(k+78) - 136*dirac(k+79) - 120*dirac(k+80) - 105*dirac(k+81) - 91*dirac(k+82) - 78*dirac(k+83) - 66*dirac(k+84) - 55*dirac(k+85) - 45*dirac(k+86) - 36*dirac(k+87) - 28*dirac(k+88) - 21*dirac(k+89) - 15*dirac(k+90) - 10*dirac(k+91) - 6*dirac(k+92) - 3*dirac(k+93) - dirac(k+94))

        j = 7
        a , b = self.getAnBvalues(j)
        k_index7 = []
        for k in range(a,b):
            k_index7.append(k)
            self.qj[7][k+abs(a)] = -1/131072 * (dirac(k-63) + 3*dirac(k-62) + 6*dirac(k-61) + 10*dirac(k-60) + 15*dirac(k-59) + 21*dirac(k-58) + 28*dirac(k-57) + 36*dirac(k-56) + 45*dirac(k-55) + 55*dirac(k-54) + 66*dirac(k-53) + 78*dirac(k-52) + 91*dirac(k-51) + 105*dirac(k-50) + 120*dirac(k-49) + 136*dirac(k-48) + 153*dirac(k-47) + 171*dirac(k-46) + 190*dirac(k-45) + 210*dirac(k-44) + 231*dirac(k-43) + 253*dirac(k-42) + 276*dirac(k-41) + 300*dirac(k-40) + 325*dirac(k-39) + 351*dirac(k-38) + 378*dirac(k-37) + 406*dirac(k-36) + 435*dirac(k-35) + 465*dirac(k-34) + 496*dirac(k-33) + 528*dirac(k-32) + 561*dirac(k-31) + 595*dirac(k-30) + 630*dirac(k-29) + 666*dirac(k-28) + 703*dirac(k-27) + 741*dirac(k-26) + 780*dirac(k-25) + 820*dirac(k-24) + 861*dirac(k-23) + 903*dirac(k-22) + 946*dirac(k-21) + 990*dirac(k-20) + 1035*dirac(k-19) + 1081*dirac(k-18) + 1128*dirac(k-17) + 1176*dirac(k-16) + 1225*dirac(k-15) + 1275*dirac(k-14) + 1326*dirac(k-13) + 1378*dirac(k-12) + 1431*dirac(k-11) + 1485*dirac(k-10) + 1540*dirac(k-9) + 1596*dirac(k-8) + 1653*dirac(k-7) + 1711*dirac(k-6) + 1770*dirac(k-5) + 1830*dirac(k-4) + 1891*dirac(k-3) + 1953*dirac(k-2) + 2016*dirac(k-1) + 2080*dirac(k) + 2141*dirac(k+1) + 2199*dirac(k+2) + 2254*dirac(k+3) + 2306*dirac(k+4) + 2355*dirac(k+5) + 2401*dirac(k+6) + 2444*dirac(k+7) + 2484*dirac(k+8) + 2521*dirac(k+9) + 2555*dirac(k+10) + 2586*dirac(k+11) + 2614*dirac(k+12) + 2639*dirac(k+13) + 2661*dirac(k+14) + 2680*dirac(k+15) + 2696*dirac(k+16) + 2709*dirac(k+17) + 2719*dirac(k+18) + 2726*dirac(k+19) + 2730*dirac(k+20) + 2731*dirac(k+21) + 2729*dirac(k+22) + 2724*dirac(k+23) + 2716*dirac(k+24) + 2705*dirac(k+25) + 2691*dirac(k+26) + 2674*dirac(k+27) + 2654*dirac(k+28) + 2631*dirac(k+29) + 2605*dirac(k+30) + 2576*dirac(k+31) + 2544*dirac(k+32) + 2509*dirac(k+33) + 2471*dirac(k+34) + 2430*dirac(k+35) + 2386*dirac(k+36) + 2339*dirac(k+37) + 2289*dirac(k+38) + 2236*dirac(k+39) + 2180*dirac(k+40) + 2121*dirac(k+41) + 2059*dirac(k+42) + 1994*dirac(k+43) + 1926*dirac(k+44) + 1855*dirac(k+45) + 1781*dirac(k+46) + 1704*dirac(k+47) + 1624*dirac(k+48) + 1541*dirac(k+49) + 1455*dirac(k+50) + 1366*dirac(k+51) + 1274*dirac(k+52) + 1179*dirac(k+53) + 1081*dirac(k+54) + 980*dirac(k+55) + 876*dirac(k+56) + 769*dirac(k+57) + 659*dirac(k+58) + 546*dirac(k+59) + 430*dirac(k+60) + 311*dirac(k+61) + 189*dirac(k+62) + 64*dirac(k+63) - 64*dirac(k+64) - 189*dirac(k+65) - 311*dirac(k+66) - 430*dirac(k+67) - 546*dirac(k+68) - 659*dirac(k+69) - 769*dirac(k+70) - 876*dirac(k+71) - 980*dirac(k+72) - 1081*dirac(k+73) - 1179*dirac(k+74) - 1274*dirac(k+75) - 1366*dirac(k+76) - 1455*dirac(k+77) - 1541*dirac(k+78) - 1624*dirac(k+79) - 1704*dirac(k+80) - 1781*dirac(k+81) - 1855*dirac(k+82) - 1926*dirac(k+83) - 1994*dirac(k+84) - 2059*dirac(k+85) - 2121*dirac(k+86) - 2180*dirac(k+87) - 2236*dirac(k+88) - 2289*dirac(k+89) - 2339*dirac(k+90) - 2386*dirac(k+91) - 2430*dirac(k+92) - 2471*dirac(k+93) - 2509*dirac(k+94) - 2544*dirac(k+95) - 2576*dirac(k+96) - 2605*dirac(k+97) - 2631*dirac(k+98) - 2654*dirac(k+99) - 2674*dirac(k+100) - 2691*dirac(k+101) - 2705*dirac(k+102) - 2716*dirac(k+103) - 2724*dirac(k+104) - 2729*dirac(k+105) - 2731*dirac(k+106) - 2730*dirac(k+107) - 2726*dirac(k+108) - 2719*dirac(k+109) - 2709*dirac(k+110) - 2696*dirac(k+111) - 2680*dirac(k+112) - 2661*dirac(k+113) - 2639*dirac(k+114) - 2614*dirac(k+115) - 2586*dirac(k+116) - 2555*dirac(k+117) - 2521*dirac(k+118) - 2484*dirac(k+119) - 2444*dirac(k+120) - 2401*dirac(k+121) - 2355*dirac(k+122) - 2306*dirac(k+123) - 2254*dirac(k+124) - 2199*dirac(k+125) - 2141*dirac(k+126) - 2080*dirac(k+127) - 2016*dirac(k+128) - 1953*dirac(k+129) - 1891*dirac(k+130) - 1830*dirac(k+131) - 1770*dirac(k+132) - 1711*dirac(k+133) - 1653*dirac(k+134) - 1596*dirac(k+135) - 1540*dirac(k+136) - 1485*dirac(k+137) - 1431*dirac(k+138) - 1378*dirac(k+139) - 1326*dirac(k+140) - 1275*dirac(k+141) - 1225*dirac(k+142) - 1176*dirac(k+143) - 1128*dirac(k+144) - 1081*dirac(k+145) - 1035*dirac(k+146) - 990*dirac(k+147) - 946*dirac(k+148) - 903*dirac(k+149) - 861*dirac(k+150) - 820*dirac(k+151) - 780*dirac(k+152) - 741*dirac(k+153) - 703*dirac(k+154) - 666*dirac(k+155) - 630*dirac(k+156) - 595*dirac(k+157) - 561*dirac(k+158) - 528*dirac(k+159) - 496*dirac(k+160) - 465*dirac(k+161) - 435*dirac(k+162) - 406*dirac(k+163) - 378*dirac(k+164) - 351*dirac(k+165) - 325*dirac(k+166) - 300*dirac(k+167) - 276*dirac(k+168) - 253*dirac(k+169) - 231*dirac(k+170) - 210*dirac(k+171) - 190*dirac(k+172) - 171*dirac(k+173) - 153*dirac(k+174) - 136*dirac(k+175) - 120*dirac(k+176) - 105*dirac(k+177) - 91*dirac(k+178) - 78*dirac(k+179) - 66*dirac(k+180) - 55*dirac(k+181) - 45*dirac(k+182) - 36*dirac(k+183) - 28*dirac(k+184) - 21*dirac(k+185) - 15*dirac(k+186) - 10*dirac(k+187) - 6*dirac(k+188) - 3*dirac(k+189) - dirac(k+190))

        j = 8
        a, b = self.getAnBvalues(j)
        k_index8 = []
        for k in range(a, b):
            k_index8.append(k)
            coeff = 0
            coeff += dirac(k-127) + 3*dirac(k-126) + 6*dirac(k-125) + 10*dirac(k-124) + 15*dirac(k-123) + 21*dirac(k-122)
            coeff += 28*dirac(k-121) + 36*dirac(k-120) + 45*dirac(k-119) + 55*dirac(k-118) + 66*dirac(k-117) + 78*dirac(k-116)
            coeff += 91*dirac(k-115) + 105*dirac(k-114) + 120*dirac(k-113) + 136*dirac(k-112) + 153*dirac(k-111) + 171*dirac(k-110)
            coeff += 190*dirac(k-109) + 210*dirac(k-108) + 231*dirac(k-107) + 253*dirac(k-106) + 276*dirac(k-105) + 300*dirac(k-104)
            coeff += 325*dirac(k-103) + 351*dirac(k-102) + 378*dirac(k-101) + 406*dirac(k-100)
            
            coeff += 435*dirac(k-99) + 465*dirac(k-98) + 496*dirac(k-97) + 528*dirac(k-96) + 561*dirac(k-95) + 595*dirac(k-94)
            coeff += 630*dirac(k-93) + 666*dirac(k-92) + 703*dirac(k-91) + 741*dirac(k-90) + 780*dirac(k-89) + 820*dirac(k-88)
            coeff += 861*dirac(k-87) + 903*dirac(k-86) + 946*dirac(k-85) + 990*dirac(k-84) + 1035*dirac(k-83) + 1081*dirac(k-82)
            coeff += 1128*dirac(k-81) + 1176*dirac(k-80) + 1225*dirac(k-79) + 1275*dirac(k-78) + 1326*dirac(k-77) + 1378*dirac(k-76)
            coeff += 1431*dirac(k-75) + 1485*dirac(k-74) + 1540*dirac(k-73) + 1596*dirac(k-72) + 1653*dirac(k-71) + 1711*dirac(k-70)
            coeff += 1770*dirac(k-69) + 1830*dirac(k-68) + 1891*dirac(k-67) + 1953*dirac(k-66) + 2016*dirac(k-65) + 2080*dirac(k-64)
            coeff += 2145*dirac(k-63) + 2211*dirac(k-62) + 2278*dirac(k-61) + 2346*dirac(k-60) + 2415*dirac(k-59) + 2485*dirac(k-58)
            coeff += 2556*dirac(k-57) + 2628*dirac(k-56) + 2701*dirac(k-55) + 2775*dirac(k-54) + 2850*dirac(k-53) + 2926*dirac(k-52)
            coeff += 3003*dirac(k-51) + 3081*dirac(k-50)
            
            coeff += 3160*dirac(k-49) + 3240*dirac(k-48) + 3321*dirac(k-47) + 3403*dirac(k-46) + 3486*dirac(k-45) + 3570*dirac(k-44)
            coeff += 3655*dirac(k-43) + 3741*dirac(k-42) + 3828*dirac(k-41) + 3916*dirac(k-40) + 4005*dirac(k-39) + 4095*dirac(k-38)
            coeff += 4186*dirac(k-37) + 4278*dirac(k-36) + 4371*dirac(k-35) + 4465*dirac(k-34) + 4560*dirac(k-33) + 4656*dirac(k-32)
            coeff += 4753*dirac(k-31) + 4851*dirac(k-30) + 4950*dirac(k-29) + 5050*dirac(k-28) + 5151*dirac(k-27) + 5253*dirac(k-26)
            coeff += 5356*dirac(k-25) + 5460*dirac(k-24) + 5565*dirac(k-23) + 5671*dirac(k-22) + 5778*dirac(k-21) + 5886*dirac(k-20)
            coeff += 5995*dirac(k-19) + 6105*dirac(k-18) + 6216*dirac(k-17) + 6328*dirac(k-16) + 6441*dirac(k-15) + 6555*dirac(k-14)
            coeff += 6670*dirac(k-13) + 6786*dirac(k-12) + 6903*dirac(k-11) + 7021*dirac(k-10) + 7140*dirac(k-9) + 7260*dirac(k-8)
            coeff += 7381*dirac(k-7) + 7503*dirac(k-6) + 7626*dirac(k-5) + 7750*dirac(k-4) + 7875*dirac(k-3) + 8001*dirac(k-2)
            coeff += 8128*dirac(k-1) + 8256*dirac(k)
            
            coeff += 8381*dirac(k+1) + 8503*dirac(k+2) + 8622*dirac(k+3) + 8738*dirac(k+4) + 8851*dirac(k+5) + 8961*dirac(k+6)
            coeff += 9068*dirac(k+7) + 9172*dirac(k+8) + 9273*dirac(k+9) + 9371*dirac(k+10) + 9466*dirac(k+11) + 9558*dirac(k+12)
            coeff += 9647*dirac(k+13) + 9733*dirac(k+14) + 9816*dirac(k+15) + 9896*dirac(k+16) + 9973*dirac(k+17) + 10047*dirac(k+18)
            coeff += 10118*dirac(k+19) + 10186*dirac(k+20) + 10251*dirac(k+21) + 10313*dirac(k+22) + 10372*dirac(k+23) + 10428*dirac(k+24)
            coeff += 10481*dirac(k+25) + 10531*dirac(k+26) + 10578*dirac(k+27) + 10622*dirac(k+28) + 10663*dirac(k+29) + 10701*dirac(k+30)
            coeff += 10736*dirac(k+31) + 10768*dirac(k+32) + 10797*dirac(k+33) + 10823*dirac(k+34) + 10846*dirac(k+35) + 10866*dirac(k+36)
            coeff += 10883*dirac(k+37) + 10897*dirac(k+38) + 10908*dirac(k+39) + 10916*dirac(k+40) + 10921*dirac(k+41) + 10923*dirac(k+42)
            coeff += 10922*dirac(k+43) + 10918*dirac(k+44) + 10911*dirac(k+45) + 10901*dirac(k+46) + 10888*dirac(k+47) + 10872*dirac(k+48)
            coeff += 10853*dirac(k+49) + 10831*dirac(k+50)
            coeff += 10806*dirac(k+51) + 10778*dirac(k+52) + 10747*dirac(k+53) + 10713*dirac(k+54) + 10676*dirac(k+55) + 10636*dirac(k+56)
            coeff += 10593*dirac(k+57) + 10547*dirac(k+58) + 10498*dirac(k+59) + 10446*dirac(k+60) + 10391*dirac(k+61) + 10333*dirac(k+62)
            coeff += 10272*dirac(k+63) + 10208*dirac(k+64) + 10141*dirac(k+65) + 10071*dirac(k+66) + 9998*dirac(k+67) + 9922*dirac(k+68)
            coeff += 9843*dirac(k+69) + 9761*dirac(k+70) + 9676*dirac(k+71) + 9588*dirac(k+72) + 9497*dirac(k+73) + 9403*dirac(k+74)
            coeff += 9306*dirac(k+75) + 9206*dirac(k+76) + 9103*dirac(k+77) + 8997*dirac(k+78) + 8888*dirac(k+79) + 8776*dirac(k+80)
            coeff += 8661*dirac(k+81) + 8543*dirac(k+82) + 8422*dirac(k+83) + 8298*dirac(k+84) + 8171*dirac(k+85) + 8041*dirac(k+86)
            coeff += 7908*dirac(k+87) + 7772*dirac(k+88) + 7633*dirac(k+89) + 7491*dirac(k+90) + 7346*dirac(k+91) + 7198*dirac(k+92)
            coeff += 7047*dirac(k+93) + 6893*dirac(k+94) + 6736*dirac(k+95) + 6576*dirac(k+96) + 6413*dirac(k+97) + 6247*dirac(k+98)
            coeff += 6078*dirac(k+99) + 5906*dirac(k+100)
            coeff += 5731*dirac(k+101) + 5553*dirac(k+102) + 5372*dirac(k+103) + 5188*dirac(k+104) + 5001*dirac(k+105) + 4811*dirac(k+106)
            coeff += 4618*dirac(k+107) + 4422*dirac(k+108) + 4223*dirac(k+109) + 4021*dirac(k+110) + 3816*dirac(k+111) + 3608*dirac(k+112)
            coeff += 3397*dirac(k+113) + 3183*dirac(k+114) + 2966*dirac(k+115) + 2746*dirac(k+116) + 2523*dirac(k+117) + 2297*dirac(k+118)
            coeff += 2068*dirac(k+119) + 1836*dirac(k+120) + 1601*dirac(k+121) + 1363*dirac(k+122) + 1122*dirac(k+123) + 878*dirac(k+124)
            coeff += 631*dirac(k+125) + 381*dirac(k+126) + 128*dirac(k+127)
            coeff -= 128*dirac(k+128) + 381*dirac(k+129) + 631*dirac(k+130) + 878*dirac(k+131) + 1122*dirac(k+132) + 1363*dirac(k+133)
            coeff -= 1601*dirac(k+134) + 1836*dirac(k+135) + 2068*dirac(k+136) + 2297*dirac(k+137) + 2523*dirac(k+138) + 2746*dirac(k+139)
            coeff -= 2966*dirac(k+140) + 3183*dirac(k+141) + 3397*dirac(k+142) + 3608*dirac(k+143) + 3816*dirac(k+144) + 4021*dirac(k+145)
            coeff -= 4223*dirac(k+146) + 4422*dirac(k+147) + 4618*dirac(k+148) + 4811*dirac(k+149) + 5001*dirac(k+150) + 5188*dirac(k+151)
            coeff -= 5372*dirac(k+152) + 5553*dirac(k+153) + 5731*dirac(k+154) + 5906*dirac(k+155) + 6078*dirac(k+156) + 6247*dirac(k+157)
            coeff -= 6413*dirac(k+158) + 6576*dirac(k+159) + 6736*dirac(k+160) + 6893*dirac(k+161) + 7047*dirac(k+162) + 7198*dirac(k+163)
            coeff -= 7346*dirac(k+164) + 7491*dirac(k+165) + 7633*dirac(k+166) + 7772*dirac(k+167) + 7908*dirac(k+168) + 8041*dirac(k+169)
            coeff -= 8171*dirac(k+170) + 8298*dirac(k+171) + 8422*dirac(k+172) + 8543*dirac(k+173) + 8661*dirac(k+174) + 8776*dirac(k+175)
            coeff -= 8888*dirac(k+176) + 8997*dirac(k+177) + 9103*dirac(k+178) + 9206*dirac(k+179) + 9306*dirac(k+180) + 9403*dirac(k+181)
            coeff -= 9497*dirac(k+182) + 9588*dirac(k+183) + 9676*dirac(k+184) + 9761*dirac(k+185) + 9843*dirac(k+186) + 9922*dirac(k+187)
            coeff -= 9998*dirac(k+188) + 10071*dirac(k+189) + 10141*dirac(k+190) + 10208*dirac(k+191) + 10272*dirac(k+192) + 10333*dirac(k+193)
            coeff -= 10391*dirac(k+194) + 10446*dirac(k+195) + 10498*dirac(k+196) + 10547*dirac(k+197) + 10593*dirac(k+198) + 10636*dirac(k+199)
            coeff -= 10676*dirac(k+200)            
            coeff -= 10713*dirac(k+201) + 10747*dirac(k+202) + 10778*dirac(k+203) + 10806*dirac(k+204) + 10831*dirac(k+205) + 10853*dirac(k+206)
            coeff -= 10872*dirac(k+207) + 10888*dirac(k+208) + 10901*dirac(k+209) + 10911*dirac(k+210) + 10918*dirac(k+211) + 10922*dirac(k+212)
            coeff -= 10923*dirac(k+213) + 10921*dirac(k+214) + 10916*dirac(k+215) + 10908*dirac(k+216) + 10897*dirac(k+217) + 10883*dirac(k+218)
            coeff -= 10866*dirac(k+219) + 10846*dirac(k+220) + 10823*dirac(k+221) + 10797*dirac(k+222) + 10768*dirac(k+223) + 10736*dirac(k+224)
            coeff -= 10701*dirac(k+225) + 10663*dirac(k+226) + 10622*dirac(k+227) + 10578*dirac(k+228) + 10531*dirac(k+229) + 10481*dirac(k+230)
            coeff -= 10428*dirac(k+231) + 10372*dirac(k+232) + 10313*dirac(k+233) + 10251*dirac(k+234) + 10186*dirac(k+235) + 10118*dirac(k+236)
            coeff -= 10047*dirac(k+237) + 9973*dirac(k+238) + 9896*dirac(k+239) + 9816*dirac(k+240) + 9733*dirac(k+241) + 9647*dirac(k+242)
            coeff -= 9558*dirac(k+243) + 9466*dirac(k+244) + 9371*dirac(k+245) + 9273*dirac(k+246) + 9172*dirac(k+247) + 9068*dirac(k+248)
            coeff -= 8961*dirac(k+249) + 8851*dirac(k+250) + 8738*dirac(k+251) + 8622*dirac(k+252) + 8503*dirac(k+253) + 8381*dirac(k+254)
            coeff -= 8256*dirac(k+255) + 8128*dirac(k+256) + 8001*dirac(k+257) + 7875*dirac(k+258) + 7750*dirac(k+259) + 7626*dirac(k+260)
            coeff -= 7503*dirac(k+261) + 7381*dirac(k+262) + 7260*dirac(k+263) + 7140*dirac(k+264) + 7021*dirac(k+265) + 6903*dirac(k+266)
            coeff -= 6786*dirac(k+267) + 6670*dirac(k+268) + 6555*dirac(k+269) + 6441*dirac(k+270) + 6328*dirac(k+271) + 6216*dirac(k+272)
            coeff -= 6105*dirac(k+273) + 5995*dirac(k+274) + 5886*dirac(k+275) + 5778*dirac(k+276) + 5671*dirac(k+277) + 5565*dirac(k+278)
            coeff -= 5460*dirac(k+279) + 5356*dirac(k+280) + 5253*dirac(k+281) + 5151*dirac(k+282) + 5050*dirac(k+283) + 4950*dirac(k+284)
            coeff -= 4851*dirac(k+285) + 4753*dirac(k+286) + 4656*dirac(k+287) + 4560*dirac(k+288) + 4465*dirac(k+289) + 4371*dirac(k+290)
            coeff -= 4278*dirac(k+291) + 4186*dirac(k+292) + 4095*dirac(k+293) + 4005*dirac(k+294) + 3916*dirac(k+295) + 3828*dirac(k+296)
            coeff -= 3741*dirac(k+297) + 3655*dirac(k+298) + 3570*dirac(k+299) + 3486*dirac(k+300) + 3403*dirac(k+301) + 3321*dirac(k+302)
            coeff -= 3240*dirac(k+303) + 3160*dirac(k+304) + 3081*dirac(k+305) + 3003*dirac(k+306) + 2926*dirac(k+307) + 2850*dirac(k+308)
            coeff -= 2775*dirac(k+309) + 2701*dirac(k+310) + 2628*dirac(k+311) + 2556*dirac(k+312) + 2485*dirac(k+313) + 2415*dirac(k+314)
            coeff -= 2346*dirac(k+315) + 2278*dirac(k+316) + 2211*dirac(k+317) + 2145*dirac(k+318) + 2080*dirac(k+319) + 2016*dirac(k+320)
            coeff -= 1953*dirac(k+321) + 1891*dirac(k+322) + 1830*dirac(k+323) + 1770*dirac(k+324) + 1711*dirac(k+325) + 1653*dirac(k+326)
            coeff -= 1596*dirac(k+327) + 1540*dirac(k+328) + 1485*dirac(k+329) + 1431*dirac(k+330) + 1378*dirac(k+331) + 1326*dirac(k+332)
            coeff -= 1275*dirac(k+333) + 1225*dirac(k+334) + 1176*dirac(k+335) + 1128*dirac(k+336) + 1081*dirac(k+337) + 1035*dirac(k+338)
            coeff -= 990*dirac(k+339) + 946*dirac(k+340) + 903*dirac(k+341) + 861*dirac(k+342) + 820*dirac(k+343) + 780*dirac(k+344)
            coeff -= 741*dirac(k+345) + 703*dirac(k+346) + 666*dirac(k+347) + 630*dirac(k+348) + 595*dirac(k+349) + 561*dirac(k+350)
            coeff -= 528*dirac(k+351) + 496*dirac(k+352) + 465*dirac(k+353) + 435*dirac(k+354) + 406*dirac(k+355) + 378*dirac(k+356)
            coeff -= 351*dirac(k+357) + 325*dirac(k+358) + 300*dirac(k+359) + 276*dirac(k+360) + 253*dirac(k+361) + 231*dirac(k+362)
            coeff -= 210*dirac(k+363) + 190*dirac(k+364) + 171*dirac(k+365) + 153*dirac(k+366) + 136*dirac(k+367) + 120*dirac(k+368)
            coeff -= 105*dirac(k+369) + 91*dirac(k+370) + 78*dirac(k+371) + 66*dirac(k+372) + 55*dirac(k+373) + 45*dirac(k+374)
            coeff -= 36*dirac(k+375) + 28*dirac(k+376) + 21*dirac(k+377) + 15*dirac(k+378) + 10*dirac(k+379) + 6*dirac(k+380)
            coeff -= 3*dirac(k+381) + dirac(k+382)
            
            self.qj[8][k + abs(a)] = -1/1048576 * coeff 
            
        # plt.figure(figsize=(12, 8))
        # plt.subplot(2,1,1)
        # plt.plot(self.qj[1], label="Q1")
        # plt.plot(self.qj[2], label="Q2")
        # plt.plot(self.qj[3], label="Q3")
        # plt.plot(self.qj[4], label="Q4")
        # plt.plot(self.qj[5], label="Q5")
        # plt.plot(self.qj[6], label="Q6")
        # plt.plot(self.qj[7], label="Q7")
        # plt.plot(self.qj[8], label="Q8")
        # plt.legend()
        # plt.grid()
        
        for i in range(1, 9):
            sig = self.qj[i]
            N = len(sig)
            freq = np.fft.rfftfreq(N, d = 1/self.original_fs)
            fft_vals = np.fft.rfft(sig)
            magnitude = np.abs(fft_vals)/N
            magnitude /= np.max(magnitude) + 1e-12
            plt.subplot(2,1,2)
            plt.plot(freq, magnitude, label=f"Q{i}")
            plt.legend()
        plt.tight_layout()
        plt.show()
                
    def filter_response(self):
        fs = self.original_fs
        g = self.g_coeffs
        h = self.h_coeffs
               
        max_freq_index = 128 * int(np.round(fs / 2))
        array_size = max(20000, max_freq_index + 1)
        Hw = np.zeros(array_size)
        Gw = np.zeros(array_size)

        for i_freq in range(0, array_size):
            reG, imG, reH, imH = 0.0, 0.0, 0.0, 0.0
            for k_idx, k in enumerate(self.n_range):
                angle = k * 2 * np.pi * i_freq / fs
                reG += g[k_idx] * np.cos(angle)
                imG -= g[k_idx] * np.sin(angle)
                reH += h[k_idx] * np.cos(angle)
                imH -= h[k_idx] * np.sin(angle)
            Hw[i_freq] = np.sqrt((reH**2) + (imH**2))
            Gw[i_freq] = np.sqrt((reG**2) + (imG**2))

        # --- 2. Calculate Q ---
        n_plot_points = int(np.round(fs / 2)) + 1
        Q = np.zeros((8, n_plot_points))
        i_vals_plot = np.arange(0, n_plot_points)

        for i in i_vals_plot:
                # Need to check bounds because index can exceed array_size
            Q[0][i] = Gw[i] if i < array_size else 0
            Q[1][i] = Gw[2*i] * Hw[i] if 2*i < array_size and i < array_size else 0
            Q[2][i] = Gw[4*i] * Hw[2*i] * Hw[i] if 4*i < array_size and 2*i < array_size and i < array_size else 0
            Q[3][i] = Gw[8*i] * Hw[4*i] * Hw[2*i] * Hw[i] if 8*i < array_size and 4*i < array_size and 2*i < array_size and i < array_size else 0
            Q[4][i] = Gw[16*i] * Hw[8*i] * Hw[4*i] * Hw[2*i] * Hw[i] if 16*i < array_size and 8*i < array_size and 4*i < array_size and 2*i < array_size and i < array_size else 0
            Q[5][i] = Gw[32*i] * Hw[16*i] * Hw[8*i] * Hw[4*i] * Hw[2*i] * Hw[i] if 32*i < array_size and 16*i < array_size and 8*i < array_size and 4*i < array_size and 2*i < array_size and i < array_size else 0
            Q[6][i] = Gw[64*i] * Hw[32*i] * Hw[16*i] * Hw[8*i] * Hw[4*i] * Hw[2*i] * Hw[i] if 64*i < array_size and 32*i < array_size and 16*i < array_size and 8*i < array_size and 4*i < array_size and 2*i < array_size and i < array_size else 0
            Q[7][i] = Gw[128*i] * Hw[64*i] * Hw[32*i] * Hw[16*i] * Hw[8*i] * Hw[4*i] * Hw[2*i] * Hw[i] if 128*i < array_size and 64*i < array_size and 32*i < array_size and 16*i < array_size and 8*i < array_size and 4*i < array_size and 2*i < array_size and i < array_size else 0
        
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')
        
        for j in range(8):
            ax.plot(i_vals_plot, Q[j], label=f"Q{j+1}")
        ax.set_title(f'DWT Cascaded Filter Response (fs = {self.fs:.1f} Hz)', color='black')
        ax.set_xlabel('Frequency (Hz)', color='black')
        ax.set_ylabel('Magnitude', color='black')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(colors='black')
        ax.set_facecolor('white')
        fig.tight_layout()
        fig.show()
            
    def applying(self, signal, specific_j=None, factor=None):
        coeffs = {}
        if specific_j is not None:
            # Get the actual filter coefficients (non-zero part)
            qj_filter = self.qj[specific_j]
            conv_result = convolve(signal, qj_filter)

            if factor is not None:
                conv_result = downSample(conv_result, factor)
                print(f"Downsampled by a factor of {factor}.")
    
            coeffs[specific_j] = conv_result
            print(f"Applied only Q{specific_j} filter.")
        return coeffs
    