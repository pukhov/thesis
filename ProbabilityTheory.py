import numpy as np
import scipy
import scipy.optimize
import scipy.special
import scipy.misc
import scipy.integrate as integrate
import scipy.interpolate as interpolate

EPS = 1e-9


class Function:
    def __init__(self, 
                 lambda_f,
                 bounds):
        self.lambda_f = lambda_f
        self.left_bound = bounds[0] - EPS
        self.right_bound = bounds[1] + EPS
    
    def __call__(self, x):
        x = np.asarray(x)

        if np.logical_or((x < self.left_bound).any(), (x > self.right_bound).any()):
            raise Exception("Wrong x")            

        return self.lambda_f(x)

class CDF:
    def __init__(self, 
                 lambda_f = lambda x : x,
                 bounds = [0, 1]):
        self.left_bound = bounds[0]
        self.right_bound = bounds[1]
        
        self.lambda_f = lambda_f
        
    def __call__(self, x):
        x = np.asarray(x)
        try:
            return np.where(x < self.left_bound,
                              0,
                              np.where(x > self.right_bound, 1, self.lambda_f(x)))
        except:
#             print("Slow function called in CDF")
            ans = []
            for x_ in x:
                if x_ < self.left_bound:
                    ans.append(0)
                elif x_ > self.right_bound:
                    ans.append(1)
                else:
                    ans.append(self.lambda_f(x_))
            return np.array(ans)
        
    
class PDF:
    def __init__(self, 
                 lambda_f = lambda x : x,
                 bounds = [0, 1]):
        self.left_bound = bounds[0]
        self.right_bound = bounds[1]
        
        self.lambda_f = lambda_f
        
    def __call__(self, x):
        x = np.asarray(x)
        
        return np.where(np.logical_or(x < self.left_bound, x > self.right_bound),
                          0,
                          self.lambda_f(x))

class UniformCDF(CDF):
    def __init__(self,  bounds=[0, 1]):
        l = bounds[0]
        r = bounds[1]
        super().__init__(lambda x : (x - l) / (r - l), bounds)
        
class UniformPDF(PDF):
    def __init__(self,  bounds=[0, 1]):
        l = bounds[0]
        r = bounds[1]
        super().__init__(lambda x : 1. / (r - l), bounds)

def Derivative(function_with_call):
    derivative = lambda x  : scipy.misc.derivative(function_with_call, x,  dx=1e-8)
    return Function(lambda_f=derivative,
                          bounds=[function_with_call.left_bound,
                                  function_with_call.right_bound])
    
def CDF_to_PDF(function_with_call):
    assert isinstance(function_with_call, CDF)
    derivative = lambda x  : scipy.misc.derivative(function_with_call, x,  dx=1e-8)

    return PDF(lambda_f=derivative,
                          bounds=[function_with_call.left_bound,
                                  function_with_call.right_bound])

class Power:
    def __init__(self, f, n):
        self.f = f
        self.n = n
        
    def __call__(self,  x):
        return np.power(self.f(x), self.n) 

def k_statistics_PDF(k, n, F, f=None):
    if not f:
        f = CDF_to_PDF(F)
        
    lambda_f = lambda x : scipy.special.comb(n, k) * k * (F(x) ** (k - 1)) * ((1 - F(x)) ** (n - k)) * f(x)
    return PDF(lambda_f = lambda_f, bounds=[F.left_bound, F.right_bound])


def k_statistics_CDF(k, n, F):
    
    def lambda_f(x):
        ans = 0
        for i in range(k, n+1):
            ans += scipy.special.comb(n, i) * (F(x) ** i) * ((1 - F(x)) ** (n - i))
        return ans    
        
    return CDF(lambda_f = lambda_f, bounds=[F.left_bound, F.right_bound])


def Integrate(f, a, b, n=1000):
    x = np.linspace(a, b, n)
    y = [integrate.quad(f, a, t)[0]
                   for t in x
    ]
    
    return interpolate.interp1d(x, y, fill_value="extrapolate", assume_sorted=True)

def PDF_to_CDF(f):
    assert isinstance(f, PDF), "f must be PDF"
    lambda_ = Integrate(f, f.left_bound + EPS, f.right_bound - EPS)
    
    
    return CDF(lambda_,
                    [f.left_bound, f.right_bound])


def ExpectedValue(PDF_or_CDF):
    if isinstance(PDF_or_CDF, UniformPDF) or isinstance(PDF_or_CDF, UniformCDF):
        print("uniform pdf called")
        return (PDF_or_CDF.left_bound + PDF_or_CDF.right_bound) / 2
    elif isinstance(PDF_or_CDF, PDF):
        PDF_of_f = PDF_or_CDF
        return integrate.quad(lambda x : x * PDF_of_f(x),
                              PDF_of_f.left_bound, PDF_of_f.right_bound)[0]
    elif isinstance(PDF_or_CDF, CDF):
        PDF_of_f = CDF_to_PDF(PDF_or_CDF)
        return integrate.quad(lambda x : x * PDF_of_f(x),
                              PDF_of_f.left_bound, PDF_of_f.right_bound)[0]
    