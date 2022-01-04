import numpy as _np
import matplotlib.pyplot as plt


def _moment(x, k):
    m1 = x.mean()
    out = ((x - m1) ** k).sum() / len(x)
    return out


def _unbiased_skewness(x):
    sd = _np.sqrt(_moment(x, 2))
    n = len(x)
    gamma1 = _moment(x, 3) / sd ** 3
    skewness = _np.sqrt(n * (n - 1)) * gamma1 / (n - 2)
    return skewness


def _unbiased_kurtosis(x):
    n = len(x)
    var = _moment(x, 2)
    gamma2 = _moment(x, 4) / var ** 2
    kurtosis = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * gamma2 - 3 * (n - 1)) + 3
    return kurtosis


def _unbiased_std(x):
    return x.std(ddof=1)


def _sample_skewness(x):
    sd = _np.sqrt(_moment(x, 2))
    skewness = _moment(x, 3) / sd ** 3
    return skewness


def _sample_kurtosis(x):
    var = _moment(x, 2)
    kurtosis = _moment(x, 4) / var ** 2
    return kurtosis


def _sample_std(x):
    return _np.sqrt(_moment(x, 2))


def _discrete_plot(ax, kurtmax):
    rmax = 100
    rmin = -rmax
    # negbin dist
    p = _np.exp(-10)
    lr = _np.arange(rmin, rmax + 0.1, 0.1)
    r = _np.exp(lr)
    s2a = (2 - p) ** 2 / (r * (1 - p))
    ya = 3 + 6 / r + p ** 2 / (r * (1 - p))
    p = 1 - _np.exp(-10)
    lr = _np.arange(rmax, rmin - 0.1, -0.1)
    r = _np.exp(lr)
    s2b = (2 - p) ** 2 / (r * (1 - p))
    yb = 3 + 6 / r + p ** 2 / (r * (1 - p))
    s2 = _np.append(s2a, s2b)
    y = _np.append(ya, yb)
    xy = _np.array([s2, y])
    ax.fill(xy[0, :], xy[1, :], "lightgrey", label="negative binomial")

    # if (!is.null(boot)) {
    # legend(xmax*0.2,ymax*0.98,pch=1,legend="bootstrapped values",
    #     bty="n",cex=0.8,col=boot.col)
    # }

    # poisson dist
    llambda = _np.arange(rmin, rmax + 0.1, 0.1)
    lamb = _np.exp(llambda)
    s2 = 1 / lamb
    y = 3 + 1 / lamb
    ax.plot(s2, y, "--", label="Poisson")
    ax.plot(0, 3, "*", label="normal")
    return ax


def _cont_plot(ax, kurtmax):
    rmax = 100
    rmin = -rmax
    # beta dist
    # fmt: off
    p=_np.exp(rmin)
    lq=_np.arange(rmin,rmax+0.1,0.1)
    q=_np.exp(lq)
    s2a=(4*(q-p)**2*(p+q+1))/((p+q+2)**2*p*q)
    ya=(3*(p+q+1)*(p*q*(p+q-6)+2*(p+q)**2)/(p*q*(p+q+2)*(p+q+3))) #kurtmax-
    p=_np.exp(rmax)
    lq=_np.arange(rmin,rmax+0.1,0.1)
    q=_np.exp(lq)
    s2b=(4*(q-p)**2*(p+q+1))/((p+q+2)**2*p*q)
    yb=(3*(p+q+1)*(p*q*(p+q-6)+2*(p+q)**2)/(p*q*(p+q+2)*(p+q+3))) #kurtmax-
    s2=_np.append(s2a,s2b)
    y=_np.append(ya,yb)
    xy = _np.array([s2,y])
    ax.fill(xy[0,:],xy[1,:], 'lightgrey', label='beta')
    # gamma dist
    lshape=_np.arange(rmin,rmax+0.1,0.1)
    shape=_np.exp(lshape)
    s2=4/shape
    y=(3+6/shape)
    ax.plot(s2,y,"--", label='gamma')
    # lnorm dist
    # I have no idea why this needs to be such a large
    # range? especially since we're then taking the exp
    # and the 4th power - quickly maxes out number length
    # override for now...
    # lshape=_np.arange(rmin,rmax+0.1,0.1)
    # shape=_np.exp(lshape)
    # es2=_np.exp(shape**2)
    es2=_np.arange(0,rmax+0.1,0.1)
    s2=(es2+2)**2*(es2-1)
    y=(es2**4+2*es2**3+3*es2**2-3)
    # added to only show +ve x once I overrode above
    y = y[s2>=0]
    s2 = s2[s2>=0]
   
    ax.plot(s2,y,":", label='lognormal')
    # fmt: on
    ax.plot(0, 3, "*", label="normal")
    ax.plot(0, 9 / 5, "^", label="uniform")
    ax.plot(2 ** 2, 9, "X", label="exponential")
    ax.plot(0, 4.2, "P", label="logistic")
    return ax


def _boot(boot):
    if boot < 10:
        raise ValueError("boot value is less than 10")
    pass


def describe_distribution(x, discrete=False, boot=None, method="unbiased", graph=True):

    if method == "unbiased":
        skewness = _unbiased_skewness
        kurtosis = _unbiased_kurtosis
        std = _unbiased_std
    else:
        skewness = _sample_skewness
        kurtosis = _sample_kurtosis
        std = _sample_std

    res = dict(
        min=x.min(),
        max=x.max(),
        median=_np.median(x),
        mean=x.mean(),
        sd=std(x),
        skewness=skewness(x),
        kurtosis=kurtosis(x),
        method=method,
    )

    skewdata = res["skewness"]
    kurtdata = res["kurtosis"]

    if graph:
        if boot is not None:
            _boot(boot)
        else:
            kurtmax = max(10, _np.ceil(kurtdata))
            xmax = max(4, _np.ceil(skewdata ** 2))

        ymax = kurtmax  # in orginal code is kurtmax-1 but ugly graph with high kurt
        fig, ax = plt.subplots()
        ax.set_ylim((ymax + 1, 0))
        ax.set_xlim((-0.1, xmax + 0.1))
        ax.set_title("Cullen and Frey Graph")
        ax.set_ylabel("kurtosis")
        ax.set_xlabel("square of skewness")
        ax.xaxis.set_ticks(_np.arange(0, xmax + 1, 1))
        ax.yaxis.set_ticks(_np.arange(0, ymax + 1, 1))

        if discrete == False:
            ax = _cont_plot(ax, kurtmax)
        else:
            ax = _discrete_plot(ax, kurtmax)

        ax.plot([skewdata ** 2], [kurtdata], "o", label="observation")
        ax.legend()
    if graph == True:
        return fig
    else:
        return res


if __name__ == "__main__":
    x = _np.array([1, 4, 7, 9, 15, 20, 54])
    n = len(x)

    describe_distribution(x, method="sample")
