import time,warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern,WhiteKernel,ConstantKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.exceptions import ConvergenceWarning

LABELS={'local_gp':'局部ARD-Matérn高斯过程','krr':'RBF核岭回归','extra_trees':'极端随机树集成','bayes_last_layer':'贝叶斯末层神经网络（冻结特征后的近似BNN）'}
LABELS['global_gp']='全样本ARD-Matérn高斯过程'
def eligible(n,d,o,include_global_gp=False):
    alg=['extra_trees','bayes_last_layer'];reasons={}
    if n<=1800 and d<=30 and o<=16:alg.insert(0,'local_gp')
    else:reasons['local_gp']='样本/维数/输出规模较大，逐指标局部GP训练成本超出首版默认候选范围。'
    if n<=3000:alg.insert(1,'krr')
    else:reasons['krr']='核矩阵随样本量二次增长，默认跳过超过3000组的密集KRR。'
    if include_global_gp:
        if n<=600 and d<=20 and o<=8:alg.append('global_gp')
        else:reasons['global_gp']='全样本GP限于600样本、20输入、8输出以内的显式启用任务。'
    return alg,reasons

class Predictor:
    def __init__(self,algorithm,center=None,neighbors=120):self.algorithm=algorithm;self.center=center;self.neighbors=neighbors
    def fit(self,x,y):
        self.n_available=len(x);ix=np.arange(len(x))
        if self.algorithm=='local_gp':ix=np.argsort(np.linalg.norm(x-self.center,axis=1))[:self.neighbors]
        self.xs=StandardScaler().fit(x[ix]);self.ys=StandardScaler().fit(y[ix]);xx=self.xs.transform(x[ix]);yy=self.ys.transform(y[ix]);self.n_fit=len(ix);self.warnings=[]
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always',ConvergenceWarning)
            if self.algorithm in ('local_gp','global_gp'):
                self.models=[]
                for j in range(y.shape[1]):
                    kernel=ConstantKernel(1,(.01,100))*Matern(np.ones(x.shape[1]),(.02,30),nu=2.5)+WhiteKernel(.01,(1e-5,.5))
                    gp=GaussianProcessRegressor(kernel=kernel,alpha=1e-8,random_state=13);gp.fit(xx,yy[:,j]);self.models.append(gp)
            elif self.algorithm=='krr':self.model=KernelRidge(alpha=.03,kernel='rbf',gamma=1/x.shape[1]).fit(xx,yy)
            elif self.algorithm=='extra_trees':self.model=ExtraTreesRegressor(n_estimators=120,min_samples_leaf=2,max_features=1.,n_jobs=4,random_state=13).fit(xx,yy)
            elif self.algorithm=='bayes_last_layer':
                self.net=MLPRegressor(hidden_layer_sizes=(64,32),activation='tanh',alpha=.01,max_iter=250,early_stopping=len(xx)>=40,random_state=13).fit(xx,yy if yy.shape[1]>1 else yy.ravel())
                features=self.features(xx);self.models=[BayesianRidge().fit(features,yy[:,j]) for j in range(yy.shape[1])]
            else:raise ValueError(self.algorithm)
            self.warnings=[str(w.message) for w in records]
        return self
    def features(self,x):
        for w,b in zip(self.net.coefs_[:-1],self.net.intercepts_[:-1]):x=np.tanh(x@w+b)
        return x
    def predict(self,x):
        xx=self.xs.transform(x)
        if self.algorithm in ['local_gp','global_gp','bayes_last_layer']:
            features=self.features(xx) if self.algorithm=='bayes_last_layer' else xx
            values=[g.predict(features,return_std=True) for g in self.models];mean=np.column_stack([v[0] for v in values]);std=np.column_stack([v[1] for v in values])*self.ys.scale_
        elif self.algorithm=='extra_trees':
            a=np.array([np.asarray(t.predict(xx)).reshape(len(xx),-1) for t in self.model.estimators_]);mean=a.mean(0);std=a.std(0)*self.ys.scale_
        else:
            mean=np.asarray(self.model.predict(xx)).reshape(len(xx),-1);std=np.zeros_like(mean)
        return self.ys.inverse_transform(mean),std

def batched_bnn_predict(train_x,train_y,query_x,prior_std=.1,lr=.05,decay=.999,max_steps=3000,patience=200,samples=200,seed=0,device='auto',kl_weight=1.,min_steps=500):
    """Variational-inference BNN of Liu et al. (IEEE TAP 2022, Sec. III-B), one independent network per query point.

    train_x (B,tau,d), train_y (B,tau,m), query_x (B,d). Structure d -> 2d -> max(d,2m) -> m, Gaussian prior
    std 0.1, Adam lr 0.05 with 0.999 per-step decay and early stopping; the B networks are trained in parallel.
    Prediction mean/std come from weight sampling (eqs. 13-15). Hidden activation (tanh) is not stated in the paper.
    """
    import torch
    import torch.nn.functional as F
    dev=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device)
    torch.manual_seed(seed)
    X=torch.as_tensor(np.asarray(train_x),dtype=torch.float32,device=dev);Y=torch.as_tensor(np.asarray(train_y),dtype=torch.float32,device=dev)
    Q=torch.as_tensor(np.asarray(query_x),dtype=torch.float32,device=dev)[:,None,:]
    B,_,d=X.shape;m=Y.shape[2]
    xm=X.mean(1,keepdim=True);xs=X.std(1,keepdim=True).clamp_min(1e-6);ym=Y.mean(1,keepdim=True);ys=Y.std(1,keepdim=True).clamp_min(1e-6)
    Xn=(X-xm)/xs;Yn=(Y-ym)/ys;Qn=(Q-xm)/xs
    sizes=[d,2*d,max(d,2*m),m];layers=[]
    for a,b in zip(sizes[:-1],sizes[1:]):
        layers.append([(torch.randn(B,a,b,device=dev)/np.sqrt(a)).requires_grad_(),torch.full((B,a,b),-5.,device=dev,requires_grad=True),
                       torch.zeros(B,1,b,device=dev,requires_grad=True),torch.full((B,1,b),-5.,device=dev,requires_grad=True)])
    log_noise=torch.full((B,1,m),-1.,device=dev,requires_grad=True)
    opt=torch.optim.Adam([p for layer in layers for p in layer]+[log_noise],lr=lr);sched=torch.optim.lr_scheduler.ExponentialLR(opt,gamma=decay)
    def forward(h):
        for k,(wm,wr,bm,br) in enumerate(layers):
            h=torch.bmm(h,wm+F.softplus(wr)*torch.randn_like(wm))+bm+F.softplus(br)*torch.randn_like(bm)
            if k<len(layers)-1:h=torch.tanh(h)
        return h
    def kl():
        total=0
        for wm,wr,bm,br in layers:
            for mu,rho in ((wm,wr),(bm,br)):
                s=F.softplus(rho);total=total+(torch.log(prior_std/s)+(s**2+mu**2)/(2*prior_std**2)-.5).sum(dim=(1,2))
        return total
    best=float('inf');stall=0
    for step in range(max_steps):
        opt.zero_grad()
        nll=(.5*((Yn-forward(Xn))**2*torch.exp(-2*log_noise)+2*log_noise+np.log(2*np.pi))).sum(dim=(1,2))
        objective=(nll+kl_weight*kl()).sum()  # negative evidence lower bound (eq. 11); the paper does not state KL scaling
        objective.backward();opt.step();sched.step()
        value=objective.item()
        if value<best-1e-4*abs(best):best=value;stall=0
        else:stall+=1
        if step>=min_steps and stall>=patience:break
    with torch.no_grad():
        draws=torch.stack([forward(Qn)[:,0,:] for _ in range(samples)])*ys[:,0,:]+ym[:,0,:]
    return draws.mean(0).cpu().numpy(),draws.std(0).cpu().numpy(),dict(bnn_steps=step+1,device=str(dev),prediction_samples=samples)
