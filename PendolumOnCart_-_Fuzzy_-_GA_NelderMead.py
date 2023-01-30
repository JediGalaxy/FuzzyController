# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 16:50:10 2019

@author: Aquilon
"""
import matplotlib.pyplot as plt
import pylab
import math
import numpy as np
from scipy.optimize import minimize
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import sys

class Pendulum:
    def __init__(self):
        # характеристики маятника
        self.m = 1 #суммарная масса (исходный 100)
        self.mb = 0.1 # масса маятника (исходный 50)
        self.g = 9.8 # гравитационная постоянная
        self.Q = -10 # угол наклона (исходный -5)
        self.Q1 = 0 # угловая скорость
        self.Q2 = 0 # угловое ускорение
        self.l = 1 # длина маятника (исходный 10) 1 - удобный
        self.X = 0 # позиция тележки в t
        self.X1 = 0 # скорость тележки
        self.X2 = 0 # ускорение движения тележки
        self.f = 0 # сила прилагаемая к тележке (исходный 2)
        self.t = 100 # время (исходный 100)
        self.dt = 0.01 # единица времени
        self.b = 0 # переменная цикла
        self.Xg = 0
        self.n = 0
        self.er = 0
        #параметры для управления PID регулятором
        self.Qg = 0 # желаемое положение угла (в градусах)
        self.e0 = 0 # эта переменная для хранения невязки в предыдущий 
                    # момент времени - нужно для рассчета дифференциальной
                    # части PID регулятора
        self.s = 0
        self.VarGrFew = 0 # Правая точка переменной 0 
        self.VarGrLittle1 = -2.5 # Левая точка переменной -20
        self.VarGrLittle2 = 2.5 # Правая точка переменной 20
        self.VarGrLot = 0 # Левая точка переменной 0
        self.VarUsFew = 0 # Правая точка переменной 0
        self.VarUsLittle1 = -0.5 # Левая точка переменной -0.5
        self.VarUsLittle2 = 0.5 # Правая точка переменной -0.5
        self.VarUsLot = 0 # Левая точка переменной 0
        self.VarPtFew = 0 # Правая точка переменной 0
        self.VarPtLittle1 = -5 # Левая точка переменной -1
        self.VarPtLittle2 = 5 # Правая точка переменной 1
        self.VarPtLot = 0 # Левая точка переменной 0
        self.VarStFew = 0 # Правая точка переменной 0
        self.VarStLittle1 = -0.5 # Левая точка переменной -0.5
        self.VarStLittle2 = 0.5 # Правая точка переменной 0.5
        self.VarStLot = 0 # Левая точка переменной 0
        self.VarOutFew = 0 # Правая точка переменной 0
        self.VarOutLittle1 = -10 # Левая точка переменной -50
        self.VarOutLittle2 = 10 # Правая граница переменной 50
        self.VarOutLot = 0 # Левая точка переменной 0

        self.Outp = 150 # 50
        
        self.VarPtMid = 0  
        self.Out=[]        
        self.Out.append('few') # few
        self.Out.append('little') # little
        self.Out.append('little') # little
        self.Out.append('little') # little
        self.Out.append('lot') # lot
        
        self.Out.append('few') # few
        self.Out.append('few') # few
        self.Out.append('little') # little
        self.Out.append('lot') # lot
        self.Out.append('lot') # lot  
        # ['lot', 'little', 'lot', 'few', 'lot', 'lot', 'lot', 'little', 'lot', 'little']  термы ГА
        # ['few', 'lot', 'few', 'few', 'lot', 'little', 'few', 'few', 'little', 'little']
        self.qq = [] # массив значений угла маятника
        self.tt = [] # массив значений времени
        self.zz = [] # нулевая ось
        self.ff = [] # сила прилагаемая к маятнику
        self.xx = [] # позиция тележки
        self.O1 = []
        self.O2 = []
#=================Параметры ГА=================================================
        self.n_attribute=20                              #количество столбцов(признаков) в выборке
        self.population=10                               #размер популяции индивидов для ГА (кол-во строк)
        self.generation=5                               #количество поколений  
        self.genotype=np.empty((1,21), float)     #генотип для ГА(размер строки соответствует кол-ву столбцов(признаков) исходной выборки)
        self.genotype[0,:]=0
        self.individual=np.empty((self.population,
                                  self.n_attribute), float)         #массив индивидов
        self.error_function=np.empty((self.population,1), float)    #массив значения ошибки по каждому индивиду в популяции
        self.best=np.empty((self.generation,self.n_attribute+1), float) 
        self.fit=np.empty((self.population,1), float)               #инициализация массива для фитнес функции
        self.Best_individ=np.empty((self.generation,self.n_attribute+2), float)               #инициализация массива для фитнес функции
        self.GEN=np.empty((1,20), float)
        self.parent=np.empty((self.population*2,
                        self.n_attribute), float)                   #массив родителей(выбирается N индивидов, среди них выбирается 1 с наибольшей fitness )
        self.child=np.empty((self.population,
                             self.n_attribute), float)              #массив потомков
        self.new_population=np.empty((self.population,
                                      self.n_attribute), float)     #массив популяции полученной после мутации

        self.d = 0      

        self.iteration = 0
#------------------------------------------------------------------------------        
    def Drop(self, Mas = np.array([]), fl_show=False):
        self.iteration+=1
        print("\rIteration: {}".format(self.iteration), end="")
        # print("Итерация - ", self.iteration)
        i = 0
        
        while i < self.n_attribute:
            if (self.genotype[0,i]==0) & (self.genotype[0,i+1]==0):
                self.Out.append('few')
            if ((self.genotype[0,i]==1) & (self.genotype[0,i+1]==0)) | ((self.genotype[0,i]==0) & (self.genotype[0,i+1]==1)):
                self.Out.append('little')
            if (self.genotype[0,i]==1) & (self.genotype[0,i+1]==1):
                self.Out.append('lot')            
            i+=2
            
        if len(Mas)==12:
            self.VarGrFew = Mas[0]
            self.VarGrLittle1 = Mas[1]
            self.VarGrLittle2 = Mas[2]
            self.VarGrLot = Mas[3]
            self.VarUsFew = Mas[4]
            self.VarUsLittle1 = Mas[5]
            self.VarUsLittle2 = Mas[6]
            self.VarUsLot = Mas[7]
            self.VarOutFew = Mas[8]
            self.VarOutLittle1 = Mas[9]
            self.VarOutLittle2 = Mas[10]
            self.VarOutLot = Mas[11]

        FLC = self.Controller()
        er = 0
        Q2 = self.Q2
        Q1 = self.Q1
        Q = self.Q
        X2 = 0
        X1 = 0
        X = self.X
        f = self.f
        b = self.b
        k = 0
        
        self.qq = [] # массив значений угла маятника
        self.tt = [] # массив значений времени
        self.zz = [] # нулевая ось
        self.ff = [] # сила прилагаемая к маятнику
        self.xx = [] # позиция тележки
        self.O1 = []
        self.O2 = []
        Pm = np.random.uniform(low=-1, high=1)
        MasQ = 0
        MasX = 0
        while b < self.t:
            QRad=math.radians(Q)
            self.qq.append(Q) 
            self.tt.append(b) 
            self.zz.append(0)
            self.ff.append(f)
            self.xx.append(X)
            self.O1.append(-5)
            self.O2.append(5)
            
            # Q2 - угловое ускорение, Q1 - угловая скорость, Q - угол
            Q2 = ((self.m*self.g*math.sin(QRad)-math.cos(QRad)*(f+self.mb*self.l*math.pow(Q1, 2)*math.sin(QRad)))/
                       ((4/3)*self.m*self.l-self.mb*self.l*(math.cos(QRad)**2)))+Pm
            Q1 = Q2*self.dt + Q1
            Q = math.degrees(Q1*self.dt + QRad)
            MasQ+=Q
            
            X2 = ((f+self.mb*self.l*((Q1**2)*math.sin(QRad-Q2*math.cos(QRad)))/self.m))
            X1 = X2*self.dt+X1
            X = X1*self.dt+X
            MasX+=X
            
            # FLC.input['gr']=math.degrees(self.Qg-QRad)
            FLC.input['gr']=Q
            FLC.input['us']=Q1
            FLC.input['pt']=X
            FLC.input['st']=X1
            FLC.compute()
            f = FLC.output['outp']

            k += 1
            er += (0.8*(self.Xg-X)**2)+(0.2*(self.Qg-Q)**2)
            er=er/k
            
            b=b+self.dt
            # print("\rTime: {}".format(b), end="")
          
        if fl_show==True: 
            plt.close("all")
            plt.figure('Обратный маятник')
            pylab.ylim(-30, 30)
            plt.ylabel('угол отклонения') 
            plt.xlabel('время')
            plt.plot(self.tt, self.O1, 'g--')
            plt.plot(self.tt, self.O2, 'g--')
            plt.plot(self.tt, self.zz, 'green')
            plt.plot(self.tt, self.qq, 'red')
            
            plt.figure('Тележка')
            pylab.ylim(-30, 30)
            plt.ylabel('позиция тележки') 
            plt.xlabel('время')
            plt.plot(self.tt, self.O1, 'g--')
            plt.plot(self.tt, self.O2, 'g--')
            plt.plot(self.tt, self.zz, 'green')
            plt.plot(self.tt, self.xx, 'blue')
            
            plt.figure('Система')
            pylab.ylim(-30, 30)
            plt.ylabel('Показатели') 
            plt.xlabel('время')
            plt.plot(self.tt, self.O1, 'g--')
            plt.plot(self.tt, self.O2, 'g--')
            plt.plot(self.tt, self.zz, 'green')
            plt.plot(self.tt, self.xx, 'blue')
            plt.plot(self.tt, self.qq, 'red')

            print("Отклонение угла         - ", MasQ/k)            
            print("Отклонение позиции      - ", MasX/k)
            print("Максимальное по углу    - ", np.max(np.absolute(self.qq)))
            print("Максимальное по позиции - ", np.max(np.absolute(self.xx)))

        return er
#------------------------------------------------------------------------------        
    def Controller(self):
        # print('Инициализация контроллера')
        gr = ctrl.Antecedent(np.arange(-90, 90, 0.01), 'gr')
        us = ctrl.Antecedent(np.arange(-90, 90, 0.01), 'us')
        pt = ctrl.Antecedent(np.arange(-100, 100, 0.01), 'pt')
        st = ctrl.Antecedent(np.arange(-10, 10, 0.01), 'st')
        outp = ctrl.Consequent(np.arange(-self.Outp, self.Outp, 0.01), 'outp')

        VarGrFew = self.VarGrFew
        VarGrLittle1 = min(self.VarGrLittle1, self.VarGrLittle2, 0)        
        VarGrLittle2 = max(self.VarGrLittle1, self.VarGrLittle2, 0)
        VarGrLot = self.VarGrLot
        
        VarUsFew = max(self.VarUsFew, 0)
        VarUsLittle1 = min(self.VarUsLittle1, self.VarUsLittle2, 0) 
        VarUsLittle2 = max(self.VarUsLittle1, self.VarUsLittle2, 0)
        VarUsLot = min(self.VarUsLot, 0)
    
        VarPtFew = self.VarPtFew
        VarPtLittle1 = min(self.VarPtLittle1, self.VarPtLittle2, 0)
        VarPtLittle2 = max(self.VarPtLittle1, self.VarPtLittle2, 0)
        VarPtLot = self.VarPtLot

        VarStFew = max(self.VarStFew, 0)
        VarStLittle1 = min(self.VarStLittle1, self.VarStLittle2, 0)
        VarStLittle2 = max(self.VarStLittle1, self.VarStLittle2, 0)
        VarStLot = min(self.VarStLot, 0)
        
        VarOutFew = self.VarOutFew
        VarOutLittle1 = min(self.VarOutLittle1, self.VarOutLittle2, 0)
        VarOutLittle2 = max(self.VarOutLittle1, self.VarOutLittle2, 0)
        VarOutLot = self.VarOutLot

        # Зададим нечеткие множества

        gr['few'] = fuzz.trimf(gr.universe, [-90, -90, VarGrFew])
        gr['little'] = fuzz.trimf(gr.universe, [VarGrLittle1, 0, VarGrLittle2])
        gr['lot'] = fuzz.trimf(gr.universe, [VarGrLot, 90, 90])

        us['few'] = fuzz.trimf(us.universe, [-90, -90, VarUsFew])
        us['little'] = fuzz.trimf(us.universe, [VarUsLittle1, 0, VarUsLittle2])
        us['lot'] = fuzz.trimf(us.universe, [VarUsLot, 90, 90])
        
        pt['few'] = fuzz.trimf(pt.universe, [-100, -100, VarPtFew])
        pt['little'] = fuzz.trimf(pt.universe, [VarPtLittle1, self.VarPtMid, VarPtLittle2])
        pt['lot'] = fuzz.trimf(pt.universe, [VarPtLot, 100, 100])

        st['few'] = fuzz.trimf(st.universe, [-10, -10, VarStFew])
        st['little'] = fuzz.trimf(st.universe, [VarStLittle1, 0, VarStLittle2])
        st['lot'] = fuzz.trimf(st.universe, [VarStLot, 10, 10])

        outp['few'] = fuzz.trimf(outp.universe, [-self.Outp, -self.Outp, VarOutFew])
        outp['little'] = fuzz.trimf(outp.universe, [VarOutLittle1, 0, VarOutLittle2])
        outp['lot'] = fuzz.trimf(outp.universe, [VarOutLot, self.Outp, self.Outp])
               
        # Зададим базу правил
        
        rule1 = ctrl.Rule(gr['few'], outp[self.Out[0]]) 
        rule2 = ctrl.Rule((gr['little'] & us['few']), outp[self.Out[1]])
        rule3 = ctrl.Rule((gr['little'] & us['little']), outp[self.Out[2]])
        rule4 = ctrl.Rule((gr['little'] & us['lot']), outp[self.Out[3]])
        rule5 = ctrl.Rule(gr['lot'], outp[self.Out[4]])
        
        rule6 = ctrl.Rule((pt['few'] & gr['little']), outp[self.Out[5]])
        rule7 = ctrl.Rule((pt['few'] & st['lot'] & gr['little']), outp[self.Out[6]])
        rule8 = ctrl.Rule((pt['little'] & st['lot'] & gr['little']), outp[self.Out[7]])       
        rule9 = ctrl.Rule((pt['lot'] & st['lot'] & gr['little']), outp[self.Out[8]])
        rule10 = ctrl.Rule((pt['lot'] & gr['little']), outp[self.Out[9]])

        
        tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5,
                                           rule6, rule7, rule8, rule9, rule10])

        FLC = ctrl.ControlSystemSimulation(tipping_ctrl)
        return FLC
#------------------------------------------------------------------------------
    def Optimize(self, Mas = np.array([])):
#        
#        self.Out[0]=Out[0,0]
#        self.Out[1]=Out[0,1]
#        self.Out[2]=Out[0,2]
#        self.Out[3]=Out[0,3]
#        self.Out[4]=Out[0,4]
#        self.Out[5]=Out[0,5]
#        self.Out[6]=Out[0,6]
#        self.Out[7]=Out[0,7]
#        self.Out[8]=Out[0,8]

        if len(Mas)!= 12:
            Mas = [self.VarGrFew, self.VarGrLittle1, self.VarGrLittle2, self.VarGrLot,
                   self.VarUsFew, self.VarUsLittle1, self.VarUsLittle2, self.VarUsLot,
                   self.VarOutFew, self.VarOutLittle1, self.VarOutLittle2, self.VarOutLot]    
        
        
        res = minimize(self.Drop, Mas, method='Nelder-Mead', tol=1e-1)
        
        self.Er = self.Drop(res.x, fl_show=True)        
        print("качество управления = {0}".format(self.Er))
        
        return self.Er
#------------------------------------------------------------------------------
        
#------------------------------------------------------------------------------        
    def GA_initialization(self):                        #инициализация начальной популяции индивидов
        i=0
        j=0
        while i<self.population:                        #инициализация массива индивидов случайными значениями 0 и 1
            j=0
            while j<self.n_attribute:
                self.individual[i,j]=np.random.random_integers(0,1)  
                j+=1
            i+=1
#------------------------------------------------------------------------------
def SettingGA():                                   
    
    while S.d<S.generation:
        i=0
        j=0
        while i<S.population:                               #прогон нейросети по 1йпопуляции индивидов(результат->получение ошибок по каждому индивиду)
            j=0        
            while j<S.n_attribute:
                S.genotype[0,j]=S.individual[i,j]
                j+=1
            ER=S.Drop()                                        #ошибка по результату прогона
            S.error_function[i,0]=ER                        #запись ошибки в массив
            i+=1
        
        i=0
        while i<S.population:                               #вычисление фитнес функции для каждого индивида
            S.fit[i,0]=1/(1+S.error_function[i,0])
            if S.fit[i,0]==1:
                S.GEN[0,:]=S.individual[i,:]
                S.best[S.d,0:S.n_attribute]=S.individual[i,0:S.n_attribute]
                S.best[S.d,S.n_attribute]=S.error_function[i,0]
                print('Поколение', S.d)             
                print('Ген лучшего индивида',S.GEN)
                print('Фитнесс функция',S.fit[i,0])
                print('Ошибка на тестовых данных для индивида',S.error_function[i,0])
                return S.best
                sys.exit()   
            i+=1
        
        
        b=0
        Max=0
        while b<S.population:                                           #для запоминания лучших индивидов в каждом поколении
            if S.fit[b,0]>Max:
                Max=S.fit[b,0]
                S.best[S.d,0:S.n_attribute]=S.individual[b,0:S.n_attribute]
                S.best[S.d,S.n_attribute]=S.fit[b,0]
            b+=1
            
#--------------------Селекция------------------------------------------------------------------------------------------------ 
        n=(S.population/100)*5                              #кол-во индивидов среди которых происходит отбор(5% от общего количества )
        if n<=2:
           N=2
        else:
            N=int(n)                                        #округление влево до целого
              
                     
        c=np.empty((N,S.n_attribute+1), float)              #массив цикла, в него случ образом отбираются N индивидов из individual
        c[:,:]=0
        i=0
        j=0
        rand=0
        while i<S.population*2:                             #*2 тк создаем пара родителей состоит из 2х индивидов
            j=0
            MAX=0
            while j<N:
                rand=np.random.randint(0, S.population-1)   #генерируем число, которое обозначает индекс-строки массива individual
                c[j,0:S.n_attribute]=S.individual[rand,:]
                c[j,S.n_attribute]=S.fit[rand,0]            #ставим в соответствии индивиду его fitness функцию
                if c[j,S.n_attribute]>MAX:                  #ищем наибольшую fithess среди этих индивидов
                    S.parent[i,0:S.n_attribute]=c[j,        #массив родителей(выбирается N индивидов, среди них выбирается 1 с наибольшей fitness )
                          0:S.n_attribute]                  #записываем индивида с наибольшим fitness в массив родителей 
                    MAX=c[j,S.n_attribute] 
                j+=1
            i+=1
         
#----------------Скрещивание(одноточечное)--------------------------------------------------------------------------------------------     
        i=0
        while i<S.population:                               #создаем потомков путем скрещивания
            point_break=np.random.randint(0,S.n_attribute-1)#точка разрыва для гена
            dominant_parent=np.random.randint(0,1)          #выбираем доминирующего родителя из пары индивидов
            
            if dominant_parent==0:                          #если доминирует 1й родитель(до (.) разрыва от 1го родителя, остальное от 2го)
                S.child[i,0:point_break]=S.parent[i*2,0:point_break]
                S.child[i,point_break:S.n_attribute]=S.parent[i*2+1,point_break:S.n_attribute]
            else:                                           #если доминирует 2й родитель(до (.) разрыва от 2го родителя, остальное от 1го)
                S.child[i,0:point_break]=S.parent[i*2+1,0:point_break]
                S.child[i,point_break:S.n_attribute]=S.parent[i*2,point_break:S.n_attribute]
            i+=1
#----------------Мутация------------------------------------------------------------------------------------------------    
        coefficient_mutation=1/S.n_attribute                            #коэффициент мутации
        i=0
        j=0
        while i<S.population:
            j=0
            while j<S.n_attribute:
                random_value=np.random.uniform(0, 1)
                if random_value<coefficient_mutation:                   #генерация случ величины, сравнивается с коэф мутации
                    if S.child[i,j]==0:
                        S.new_population[i,j]=1                         #если была 1 в гене, то заменяем на 0
                    if S.child[i,j]==1:
                        S.new_population[i,j]=0                         #если был 0 в гене, то заменяем на 1
                else:
                    S.new_population[i,j]=S.child[i,j]                  #иначе записываем в новый ген информацию о старом
                j+=1
            i+=1
                
        S.individual=S.new_population                                  #смена популяции 
           
        S.d+=1
   
    return S.best 
#------------------------------------------------------------------------------
   
S = Pendulum()
S.GA_initialization()
# print("Посчитал ГА")
Best=np.empty((S.generation,S.n_attribute+1), float) 
Best[0:S.generation,0:S.n_attribute+1]=0
Best=SettingGA()
BR=np.empty((1,S.n_attribute), float) 
BR[0,0:S.n_attribute]=0
i=0
j=0
Max=0
while i<S.generation:
    if Best[i,20]>Max:
        Max=Best[i,20]
        BR[0,0:S.n_attribute]=Best[i,0:20]
    i+=1
Out=[]
i = 0 
while i < S.n_attribute:
    if (BR[0,i]==0) & (BR[0,i+1]==0):
        Out.append('few')
    if ((BR[0,i]==1) & (BR[0,i+1]==0)) | ((BR[0,i]==0) & (BR[0,i+1]==1)):
        Out.append('little')
    if (BR[0,i]==1) & (BR[0,i+1]==1):
        Out.append('lot')            
    i+=2
print("Заработал НелдерМид")    
# S.Optimize()
# S.Drop(fl_show=True)
print("\n Оптимизированные заключения")
print(Out) 
#print('Ген лучшего индивида по поколениям(поколение- № строки, последнее значение строки - ошибка на тестовых данных)', Best)