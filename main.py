# 导入模块
import gurobipy as grb
import numpy as np
import pandas as pd

#定义模型参数类

class Parameters:
    def __init__(self,k,sigma,cost,reinsurance,
                 freq,rate,Resource,MaxPay,payrate,
                 alpha,P0,require,Base,period=1):
        self.k=k
        self.sigma=sigma
        self.cost=cost
        self.reinsurance=reinsurance
        self.freq=freq
        self.rate=rate
        self.period=period
        self.Resource=Resource
        self.MaxPay=MaxPay
        self.payrate=payrate
        self.alpha=alpha
        self.P0=P0
        self.require=require
        self.Base=Base
    def calculate_parameters(self):
        #风险调节因子τ的计算
        τ=self.k*np.exp(self.sigma)
        # 目标函数常数项
        # C=lnτ-ln(cost)-ln(reinsurance)-ln(10*freq)-ln(exp(-rate*period))
        C=np.log(τ)-np.log(self.cost)-np.log(self.reinsurance)
        -np.log(10*self.freq)-np.log(np.exp(-self.rate*self.period))
        # 约束条件常数项1：Pmax=ln(MaxPay)
        Pmax=self.MaxPay
        # 约束条件常数项2：Rmax=ln(Resource)
        Rmax=self.Resource
        # 约束条件常数项3：PAYRATEmin=ln(payrate)
        PAYRATEmin=self.payrate
        # 约束条件常数项4： alpha=ln(alpha)
        alpha=self.alpha
        # alpha=np.log(self.alpha)
        # 约束条件常数项5：p0=lnP0
        p0=self.P0
        # Require=lnRequire
        Require=self.require
        # freq=lnfreq
        freq=self.freq
        Base=self.Base
        sigma=np.exp(-self.rate*self.period)
        return τ,C,Pmax,Rmax,PAYRATEmin,alpha,p0,Require,freq,Base,sigma

    def model(self):
        τ,C,Pmax,Rmax,PAYRATEmin,alpha,p0,Require,freq,Base,sigma=self.calculate_parameters()
        m=grb.Model()
        #定义变量
        x=m.addVar(name="x",lb=0)
        y=m.addVar(name="y",lb=0)
        z=m.addVar(name="z",lb=PAYRATEmin,ub=1)
        pi=m.addVar(name="pi")
        # 添加约束
        m.addConstr(x<=Pmax,name='c1')
        m.addConstr(x*y<=Rmax,name='c2')
        m.addConstr(z<=1,name='c3')
        m.addConstr(z>=PAYRATEmin,name='c4')
        # m.addConstr(2*sigma*x-y+z==-6,name='c5')
        m.addConstr(x>=p0*z,name='c5')
        m.addConstr(pi==τ*x*y-(self.cost+self.reinsurance)*y-10*self.freq*y*z-10*self.freq*sigma*z,name='c6')
        m.addConstr(y<=Require,name='c7')
        m.addConstr(x*y<=100*τ*freq*Base,name='c8')
        m.addConstr(x*y>=freq*p0*z,name='c9')

        m.setObjective(pi,grb.GRB.MAXIMIZE)
        m.optimize()
        if m.status == grb.GRB.OPTIMAL:
            m.printAttr('X')
        # 对结果取指数
        x_value = x.X
        y_value = y.X
        z_value = z.X
        pi_value =pi.X
        print('最优解的目标函数值为：',pi_value)
        print(f"Solution values: x={x_value}, y={y_value}, z={z_value}")
        return x_value,y_value,z_value,pi_value
        
        
param=Parameters(k=0.03,sigma=0.5,cost=0.061,
                    reinsurance=0.07,freq=0.27,rate=0.05,
                    Resource=800000000,MaxPay=600000,
                    payrate=0.01,alpha=-0.05,period=1,
                    Base=100000000,require=36000,P0=1000000)
list={}
list=param.calculate_parameters()
print(list)
param.model()       



