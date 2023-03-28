# Author: Kaifeng ZHU
# First Creation: 2023/3/15
# This file is for calculating and plotting for wake traverse experiment.

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import copy
import seaborn as sns
from scipy.integrate import quad
from decimal import Decimal
from scipy.optimize import fsolve

class Traverse:
    """
    This class is for calculating and plotting for wake traverse experiment.
    """
    def __init__(self) -> None:
        self.cal_df = pd.DataFrame()
        self.const_df = pd.DataFrame()
        self.u1 = 0
        

    def import_dfs(self, path):
        """
        This function is for reading excel file.
        """
        path = os.path.abspath(path)
        self.cal_df = pd.read_excel(path, sheet_name="Measurement Data")
        self.const_df = pd.read_excel(path, sheet_name="Experimental Condition")
        self.const_df.set_index('Parameter', inplace=True)
        self.u1 = self.const_df.loc['Free Steam Velocity']['Value']

    def calculate_2_2_y(self):
        """
        This function is for calculating the Task 2.2's y values.
        """
        def rel(u2):
            u1 = self.u1
            result = (u2 / u1) * (1 - u2 / u1)
            return result
        self.cal_df['u2/u1(1-u2/u1)'] = self.cal_df['Velocity'].apply(rel)

        

    def calculate_2_2_y_uncertainty(self):
        """
        This function is for calculating the uncertainty of u2/u1(1-u2/u1).
        """
        u1 = self.const_df.loc['Free Steam Velocity', 'Value']
        d_u1 = self.const_df.loc['Free Steam Velocity Uncertainty', 'Value']
        def unc(cal_df):
            # Calculate the uncertainty for u2/u1 first.
            # d_u2_u1 = np.sqrt((cal_df['Velocity'] * d_u1) ** 2 + (cal_df['Velocity uncertainty'] * u1) ** 2)
            # Calculate the uncertainty for (u2/u1)^2.
            # d_u2_u1_2 = 2 * d_u2_u1 * (cal_df['Velocity'] / u1)
            # Calculate the uncertainty.
            # cal_df['u2/u1(1-u2/u1) uncertainty'] = np.sqrt(d_u2_u1 ** 2 + d_u2_u1_2 ** 2)
            # cal_df['u2/u1(1-u2/u1) uncertainty'] = (1 - 2 * cal_df['Velocity'] / u1) * d_u2_u1

            u2 = cal_df['Velocity']
            d_u2 = cal_df['Velocity uncertainty']
            cal_df['u2/u1(1-u2/u1) uncertainty'] = np.sqrt(((u1 - 2 * u2) * d_u2 / u1 ** 2) ** 2 + ((2 * u2 ** 2 - u2 * u1) * d_u1 / u1 ** 3) ** 2)
            return cal_df
        
        self.cal_df = self.cal_df.apply(unc, axis=1)
        print(self.cal_df)


    def plotting_2_2_y(self):
        """
        This function is for plotting the Task 2.2. (trapezodial rule)
        """
        sns.set_theme()
        # sns.lineplot(x=self.cal_df['l/d'], y=self.cal_df['u2/u1(1-u2/u1)'])
        # sns.scatterplot(x=self.cal_df['l/d'], y=self.cal_df['u2/u1(1-u2/u1)'])
        plt.plot(self.cal_df['l/d'], self.cal_df['u2/u1(1-u2/u1)'], marker='o')
        # sns.histplot(data=self.cal_df, x='l/d', y='u2/u1(1-u2/u1)')
        new_df = self.cal_df.set_index('l/d')
        for i in self.cal_df['l/d']:
            if new_df.loc[i, 'u2/u1(1-u2/u1)'] > 0:
                plt.axvline(i, ymin=0.0, linestyle='dashed')
                # print(new_df.loc[i, 'u2/u1(1-u2/u1)'])
                # ymax=new_df.loc[i, 'u2/u1(1-u2/u1)']


        plt.xlabel('l/d')
        plt.ylabel('U$_2$/U$_1$(1-U$_2$/U$_1$)')
        plt.title('U$_2$/U$_1$(1-U$_2$/U$_1$) vs l/d (trapezoidal)')
        plt.savefig('trapezoidal.svg')
        plt.show()

    def plotting(self):
        
        
        sns.set_theme()

        # sns.lmplot(data=self.cal_df, x='l/d', y='u2/u1(1-u2/u1)')
        # sns.kdeplot(data=self.cal_df, x='l/d', y='u2/u1(1-u2/u1)')

        plot_df = self.cal_df.iloc[5:23]
        coeff = np.polyfit(plot_df['l/d'], plot_df['u2/u1(1-u2/u1)'], 2)
        fit_y = coeff[0] * plot_df['l/d']**2 + coeff[1]* plot_df['l/d'] + coeff[2]
        correlation = np.corrcoef(plot_df['u2/u1(1-u2/u1)'], fit_y) [0, 1]
        r2 = correlation ** 2

        # Calculate the area under the curve.
        def integrand(x):
            return coeff[0] * x **2 + coeff[1] * x + coeff[2]
        area = quad(integrand, -1.44, 1.28)
        print(area)

        print(r2)

        print(coeff)
        print(plot_df)

        # sns.scatterplot(x=self.cal_df['l/d'], y=self.cal_df['u2/u1(1-u2/u1)'])
        # plt.scatter(x=self.cal_df['l/d'], y=self.cal_df['u2/u1(1-u2/u1)'])
        plt.errorbar(plot_df['l/d'], plot_df['u2/u1(1-u2/u1)'], yerr=plot_df['u2/u1(1-u2/u1) uncertainty'], ecolor='k', elinewidth=0.5,
                     marker='s', mfc='orange', mec='k', mew=1, ms=5, alpha=1, capsize=5, capthick=3, linestyle='none')
        
        plt.plot(plot_df['l/d'], fit_y, 'k-', label=f'y = {Decimal(coeff[0]).quantize(Decimal("0.00"))}x$^2$ + {Decimal(coeff[1]).quantize(Decimal("0.00"))}x + {Decimal(coeff[2]).quantize(Decimal("0.00"))}\nR$^2$ = {Decimal(r2).quantize(Decimal("0.000"))}')

        
        

        # plt.plot(self.cal_df['l/d'], y_list)

        plt.xlabel('l/d')
        plt.ylabel('U$_2$/U$_1$(1-U$_2$/U$_1$)')
        plt.title('U$_2$/U$_1$(1-U$_2$/U$_1$) vs l/d')
        plt.legend()
        plt.savefig('quadratic.svg')
        plt.show()


    def calculate_area_trapez(self):
        """
        This function is using trapezoidal rule to calculate th area of u2/u1(1-u2/u1)
        """
        # u2/u1(1-u2/u1) cannot be smaller than 0.
        def cor(i):
            if i < 0:
                i = 0
            return i
        self.cal_df['u2/u1(1-u2/u1)'] = self.cal_df['u2/u1(1-u2/u1)'].apply(cor)
        

        postive_df = self.cal_df[self.cal_df['Pitot Position']>=0]
        negative_df = self.cal_df[self.cal_df['Pitot Position']<=0]

        postive_result = np.trapz(postive_df['u2/u1(1-u2/u1)'], postive_df['l/d'])
        negative_result = np.trapz(negative_df['u2/u1(1-u2/u1)'], negative_df['l/d'])
        overall_result = postive_result + negative_result
        self.const_df.loc['Trapez Area postive l/d', 'Value'] = postive_result
        self.const_df.loc['Trapez Area negative l/d', 'Value'] = abs(negative_result)
        self.const_df.loc['Trapez Area overall', 'Value'] = overall_result

    def calculate_trapez_zones(self):
        """
        This function is for calculating every trapezoid under the curve.
        """
        cal_df = self.cal_df[['l/d', 'u2/u1(1-u2/u1)']]
        cal_df = cal_df[cal_df['u2/u1(1-u2/u1)'] >= 0]
        cal_df.reset_index(drop='index', inplace=True)
        trapez_df = pd.DataFrame()
        
        for index in range(23):
            df = cal_df.loc[index:(index+1)]
            area = (df.loc[index, 'u2/u1(1-u2/u1)'] + df.loc[index+1, 'u2/u1(1-u2/u1)']) * (df.loc[index+1, 'l/d'] - df.loc[index, 'l/d']) * 0.5
            trapez_df.loc[index, 'Trapezoid Number'] = index+1
            trapez_df.loc[index, 'l/d 1'] = df.loc[index, 'l/d']
            trapez_df.loc[index, 'l/d 2'] = df.loc[index+1, 'l/d']
            trapez_df.loc[index, 'u2/u1(1-u2/u1) 1'] = df.loc[index, 'u2/u1(1-u2/u1)']
            trapez_df.loc[index, 'u2/u1(1-u2/u1) 2'] = df.loc[index+1, 'u2/u1(1-u2/u1)']
            trapez_df.loc[index, 'Area'] = area

        # Drop Area=0 Zone
        self.trapez_df = trapez_df[trapez_df['Area']>0]
        self.trapez_df.reset_index(drop='index', inplace=True)
        print(self.trapez_df)

    def drag_coeff_integral(self):
        """
        This function is for calculating the drag coefficient using integral.
        """
        plot_df = self.cal_df.iloc[5:23]
        coeff = np.polyfit(plot_df['l/d'], plot_df['u2/u1(1-u2/u1)'], 2)
        fit_y = coeff[0] * plot_df['l/d']**2 + coeff[1]* plot_df['l/d'] + coeff[2]
        correlation = np.corrcoef(plot_df['u2/u1(1-u2/u1)'], fit_y) [0, 1]
        r2 = correlation ** 2

        integral_result = fsolve(lambda x: coeff[0] * x **2 + coeff[1] * x + coeff[2], 0)

        delta = coeff[1] ** 2 - 4 * coeff[0] * coeff[2]
        result1 = (-coeff[1] + np.sqrt(delta)) / (2 * coeff[0])
        result2 = (-coeff[1] - np.sqrt(delta)) / (2 * coeff[0])
        print([result1, result2])

        # print(coeff)
        # print(integral_result)

        # Calculate the area under the curve.
        def integrand(x):
            return coeff[0] * x **2 + coeff[1] * x + coeff[2]
        area = quad(integrand, result1, result2)
        print(area)
            
    def is_in_uncertainty_range(self):
        """
        This function is to judge the fitting value is in the uncertainty range or not.
        """
        plot_df = self.cal_df.iloc[5:23]
        plot_df = plot_df[['Pitot Position', 'l/d', 'u2/u1(1-u2/u1)', 'u2/u1(1-u2/u1) uncertainty']]
        print(plot_df)

        coeff = np.polyfit(plot_df['l/d'], plot_df['u2/u1(1-u2/u1)'], 2)


        plot_df['fitting y'] = coeff[0] * plot_df['l/d']**2 + coeff[1]* plot_df['l/d'] + coeff[2]

        print(plot_df)

        correlation = np.corrcoef(plot_df['u2/u1(1-u2/u1)'], plot_df['fitting y']) [0, 1]
        print(f'correlation: {correlation}')

        def in_range(df):
            lower = df['u2/u1(1-u2/u1)'] - df['u2/u1(1-u2/u1) uncertainty']
            upper = df['u2/u1(1-u2/u1)'] + df['u2/u1(1-u2/u1) uncertainty']
            if (df['fitting y'] >= lower) & (df['fitting y'] <= upper):
                df['in the range'] = 'Yes'
            else:
                df['in the range'] = 'No'
            return df
        plot_df = plot_df.apply(in_range, axis=1)
        print(plot_df)
        self.uncertainty_range = plot_df




        


    # def calculate_drag_coeff_integral(self):
    #     """
    #     This function is for calculating the drag coefficient using integral.
    #     """
    #     # def integral(df):
    #     #     upp = df['l/d']
    #     #     low = -df['l/d']
    #     #     def eq(x):
    #     #         return 1 * df['u2/u1(1-u2/u1)']
    #     #     res, err = quad(eq, low, upp)
    #     #     return res
    #     # test_df = self.cal_df.apply(integral, axis=1)
    #     # print(test_df)
    #     def c_d(cal_df):
    #         cal_df['drag coefficient'] = cal_df['u2/u1(1-u2/u1)'] * cal_df['l/d'] * 4
    #         return cal_df
    #     self.cal_df = self.cal_df.apply(c_d, axis=1)
    #     print(self.cal_df)

    def output_cal_df(self, output_name):
        """
        This function is for output cal_df to excel
        """
        org_df = self.cal_df[['Pitot Position', 'l/d', 'Pitot Pressure', 'Static Pressure', 'delta P', 'Velocity', 'u2/u1(1-u2/u1)']]
        uncertainty_df = self.cal_df[['head uncertainty', 'P uncertainty', 'delta P uncertainty', 'Velocity uncertainty', 'u2/u1(1-u2/u1) uncertainty']]
        
        with pd.ExcelWriter(output_name) as writer:
            org_df.to_excel(writer, index=False, sheet_name='origin')
            self.trapez_df.to_excel(writer, index=False, sheet_name='trapez')
            uncertainty_df.to_excel(writer, index=False, sheet_name='uncertainty')
            self.uncertainty_range.to_excel(writer, index=False, sheet_name='uncertainty range')
            
            # self.suspend_df.to_excel(writer, index=False, sheet_name='Suspend')






c = Traverse()
c.import_dfs(r"C:\Users\Lenovo\OneDrive - The University of Nottingham Ningbo China\Architectural Environment Engineering\Thermofluids 1\Lab\data.xlsx")

c.calculate_2_2_y()
c.calculate_2_2_y_uncertainty()
c.is_in_uncertainty_range()

c.drag_coeff_integral()

c.calculate_trapez_zones()

# c.plotting_2_2_y()

c.calculate_area_trapez()
# c.plotting()
c.output_cal_df('python result.xlsx')

# c.calculate_drag_coeff_integral()


# y =[2, 4, 6]
# x =[-1, -3, -10]
# np.trapz(y, x)

# res, err = quad(np.cos, 0, 2*np.pi)

