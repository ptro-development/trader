import numpy as np
from scipy.stats.stats import pearsonr

def _chk_asarray(a, axis):
    if axis is None:
        a = np.ravel(a)
        outaxis = 0
    else:
        a = np.asarray(a)
        outaxis = axis
    if a.ndim == 0:
        a = np.atleast_1d(a)
    return a, outaxis

def _sum_of_squares(a, axis=0):
    a, axis = _chk_asarray(a, axis)
    return np.sum(a*a, axis)

def par(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    mx = x.mean()
    my = y.mean()
    print "mx", mx, "my", my
    xm, ym = x - mx, y - my
    print "xm", xm, "ym", ym
    print "np.multiply", np.multiply(xm, ym)
    # this multiply two arrays and then reduces them by sum
    r_num = np.add.reduce(xm * ym)
    print "r_num", r_num
    # this does sum([x0*x0, x1*x1, x2*x2])
    print "_sum_of_squares(xm)", _sum_of_squares(xm)
    r_den = np.sqrt(_sum_of_squares(xm) * _sum_of_squares(ym))
    r = r_num / r_den

    # Presumably, if abs(r) > 1, then it is only some small artifact of floating
    # point arithmetic.
    #r = max(min(r, 1.0), -1.0)
    #df = n - 2
    #if abs(r) == 1.0:
    #    prob = 0.0
    #else:
    #    t_squared = r**2 * (df / ((1.0 - r) * (1.0 + r)))
    #    prob = _betai(0.5*df, 0.5, df/(df+t_squared))

    #return r, prob
    return r

x = [1,2,3,4]
y = [6,3,1,5]
print par(x, y)
print pearsonr(x, y)
"""
[pwrap@localhost open_cl]$ python par.py
mx 2.5 my 3.75
xm [-1.5 -0.5  0.5  1.5] ym [ 2.25 -0.75 -2.75  1.25]
r_num -2.5
-0.29111125487
"""
