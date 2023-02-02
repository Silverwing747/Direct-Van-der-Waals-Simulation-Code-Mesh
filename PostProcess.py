from igakit.io import PetIGA,VTK
from igakit.igalib import bsp
from igakit.nurbs import NURBS
from igakit.cad import *
from numpy import linspace,zeros,where,min
import scipy.io
from math import sqrt
import glob
import sys
import os

Energy = 0
refinementfactor = 2
DerOrder = 0

Time_Cutoff = 100
T = 0.4624
d = 1e-2
p_sat = T*9.435871280300515E-2-(T*T)*5.208385412663068E-1+(T*T*T)*1.533073133528282-(T*T*T*T)*2.575321229230194+(T*T*T*T*T)*2.443599723948712-(T*T*T*T*T*T)*1.211376455777335+(T*T*T*T*T*T*T)*2.56820087397889E-1-7.079512757004159E-3
rho_l = 0.891302878396821
rho_v = 2.575388833238298e-05

def Calc_dp_drho(rho):
        dp_drho = (rho*rho)*((1.0/pow(rho*1.865452045155277E+15-4.503599627370496E+15,2.0)*pow((sqrt(T)-1.0)*((T*1.0E+1-7.0)*(sqrt(T)+1.0)*((T*1.0E+3-4.43E+2)*(sqrt(T)-1.0)*1.99E+2-6.635E+5)*4.398046511104E+12-3.836396961553801E+20)+4.398046511104E+20,2.0)*4.89017060925185E+4)/(rho*2.718162824974067E+15+1.125899906842624E+15)+(T*1.0/pow(rho-1.0,2.0)*1.701444111309548E-1)/rho+(T*1.0/(rho*rho)*1.701444111309548E-1)/(rho-1.0)-1.0/pow(rho*1.865452045155277E+15-4.503599627370496E+15,2.0)*(rho*4.142135623730951E-1-1.0)*pow((sqrt(T)-1.0)*((T*1.0E+1-7.0)*(sqrt(T)+1.0)*((T*1.0E+3-4.43E+2)*(sqrt(T)-1.0)*1.99E+2-6.635E+5)*4.398046511104E+12-3.836396961553801E+20)+4.398046511104E+20,2.0)*1.0/pow(rho*2.718162824974067E+15+1.125899906842624E+15,2.0)*3.209040254909952E+20-(1.0/pow(rho*1.865452045155277E+15-4.503599627370496E+15,3.0)*(rho*4.142135623730951E-1-1.0)*pow((sqrt(T)-1.0)*((T*1.0E+1-7.0)*(sqrt(T)+1.0)*((T*1.0E+3-4.43E+2)*(sqrt(T)-1.0)*1.99E+2-6.635E+5)*4.398046511104E+12-3.836396961553801E+20)+4.398046511104E+20,2.0)*4.404674106720957E+20)/(rho*2.718162824974067E+15+1.125899906842624E+15))-rho*((T*1.701444111309548E-1)/(rho*(rho-1.0))-(1.0/pow(rho*1.865452045155277E+15-4.503599627370496E+15,2.0)*(rho*4.142135623730951E-1-1.0)*pow((sqrt(T)-1.0)*((T*1.0E+1-7.0)*(sqrt(T)+1.0)*((T*1.0E+3-4.43E+2)*(sqrt(T)-1.0)*1.99E+2-6.635E+5)*4.398046511104E+12-3.836396961553801E+20)+4.398046511104E+20,2.0)*1.180591620717412E+5)/(rho*2.718162824974067E+15+1.125899906842624E+15))*2.0
        return dp_drho

with open(r'History.o', 'r') as file:
        for index, line in enumerate(file):
                if "Boundary Parameter" in line:
                        temp = line_old.split()
                        F = float(temp[1])
                        break;
                line_old = line

print(F)
dp_drho_l = Calc_dp_drho(rho_l)
dp_drho_v = Calc_dp_drho(rho_v)
A_l = (1.0 - F) / F * dp_drho_l 
A_v = (1.0 - F) / F * dp_drho_v

alpha_v = A_v * d * d * rho_v
beta_v = - A_v * d

alpha_l = A_l * d * d * rho_l
beta_l = A_l * d

def CalPressure(rho_array):
        p = zeros(rho_array.size)
        for i in range(rho_array.size):
                rho = rho_array[i]
                p[i] = -(rho*rho)*((T*1.701444111309548E-1)/(rho*(rho-1.0))-(1.0/pow(rho*1.865452045155277E+15-4.503599627370496E+15,2.0)*(rho*4.142135623730951E-1-1.0)*pow((sqrt(T)-1.0)*((T*1.0E+1-7.0)*(sqrt(T)+1.0)*((T*1.0E+3-4.43E+2)*(sqrt(T)-1.0)*1.99E+2-6.635E+5)*4.398046511104E+12-3.836396961553801E+20)+4.398046511104E+20,2.0)*1.180591620717412E+5)/(rho*2.718162824974067E+15+1.125899906842624E+15))
                if (rho < rho_v):            
                        p[i] = p[i] + alpha_v * rho / ((1.0 + d) * rho_v - rho) + beta_v * rho
                elif (rho > rho_l):
                        p[i] = p[i] + alpha_l * rho / ((1.0 - d) * rho_l - rho) + beta_l * rho
                else:
                        p[i] = p_sat + (p[i] - p_sat) / F
        return p

nrb = PetIGA().read("GeometryNSK.dat")
if(nrb.dim == 3):
        nrb2D = PetIGA().read("GeometryWedge2D.dat")


if Energy:
        FieldName = {'rho':0,'T':nrb.dim+1,'aux':nrb.dim+2,'p':nrb.dim+3}
else:
        FieldName = {'rho':0,'p':nrb.dim+1}

p_wall_dic = dict()

count = 1
count2D = 1


for infile in glob.glob("NSK_*0.dat"):
        name = infile.split(".dat")[0]
        number = name.split("NSK_")[1]
        temp = str(int(float(number)*1e+4))
        length = 10
        temp = '0' * (length - len(temp)) + temp
        nrb = PetIGA().read("GeometryNSK.dat")
        sol = PetIGA().read_vec(infile,nrb)
        nrb = NURBS(nrb.knots, nrb.control, sol)
        nrb = refine(nrb, factor=[refinementfactor]*nrb.dim)
        sol = nrb.fields
        sol[...,nrb.dim+1] = CalPressure(sol[...,0].reshape(-1)).reshape(sol[...,0].shape)        

        if float(number) > Time_Cutoff:
                if count == 1:
                        sol_all = sol
                        x_dic = nrb.control[:,0,0]
                        y_dic = nrb.control[:,0,1]
                        time_dic = np.array(float(number))
                        rho_dic = sol[:,0,0]
                        p_dic = sol[:,0,nrb.dim+1]
                else:
                        if nrb.dim == 2:
                                sol_all = sol_all + sol
                                time_dic = np.append(time_dic,float(number))
                                rho_dic = np.vstack([rho_dic,sol[:,0,0]])
                                p_dic = np.vstack([p_dic,sol[:,0,nrb.dim+1]])
                count = count + 1

        outfile = 'NSK' + temp + '.vtk'
        VTK() .write(outfile,
                nrb,
                fields=sol,
                order=DerOrder,
                scalars=FieldName,
                vectors={'velocity':range(1,nrb.dim+1)}
                )

        if (nrb.dim == 3):
                nrb2D = PetIGA().read("GeometryWedge2D.dat")
                nrb2D = refine(nrb2D, factor=[refinementfactor]*nrb2D.dim)
                sol_avg = np.mean(sol,axis=2)
                outfile = 'NSK_SpanAvg_' + temp + '.vtk'
                VTK() .write(outfile,
                        nrb2D,
                        fields=sol_avg,
                        order=DerOrder,
                        scalars=FieldName,
                        vectors={'velocity':range(1,nrb2D.dim+1)}
                        )
                if float(number) > Time_Cutoff:
                        if count2D == 1:
                                sol2D_all = sol_avg
                                x_dic = nrb2D.control[:,0,0]
                                y_dic = nrb2D.control[:,0,1]
                                time_dic = np.array(float(number))
                                rho_dic = sol_avg[:,0,0]
                                p_dic = sol_avg[:,0,nrb.dim+1]
                        else:
                                sol2D_all = sol2D_all + sol_avg
                                sol_all = sol_all + sol
                                time_dic = np.append(time_dic,float(number))
                                rho_dic = np.vstack([rho_dic,sol_avg[:,0,0]])
                                p_dic = np.vstack([p_dic,sol_avg[:,0,nrb.dim+1]])
                        count2D = count2D + 1

sol_all = sol_all / count
VTK() .write('NSK_TimeAverage.vtk',
        nrb,
        fields=sol_all,
        order=DerOrder,
        scalars=FieldName,
        vectors={'velocity':range(1,nrb.dim+1)}
        )

if (nrb.dim == 3):
        sol2D_all = sol2D_all / count2D
        VTK() .write('NSK_SpanAvg_TimeAverage.vtk',
        nrb2D,
        fields=sol2D_all,
        order=DerOrder,
        scalars=FieldName,
        vectors={'velocity':range(1,nrb2D.dim+1)}
        )

p_dic_mean = np.mean(p_dic,axis=0)
rho_dic_mean = np.mean(rho_dic,axis=0)
p_wall_dic.update({'Time':time_dic,'p':p_dic,'rho':rho_dic,'p_avg':p_dic_mean,'rho_avg':rho_dic_mean,'x':x_dic,'y':y_dic})
scipy.io.savemat("wall_data.mat", mdict=p_wall_dic)
