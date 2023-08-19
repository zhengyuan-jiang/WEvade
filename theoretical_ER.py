from scipy import stats


# HiDDeN watermarking method's (n=30) theoretical lower bound of Evasion Rate with respect to Detection Threshold \tau.
def HiDDeN_theoretical():
    tau_list = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55]
    theo_er = []
    epsilon = 0.01
    n = 30
    for tau in tau_list:
        X = int((tau-epsilon) * n)
        print(stats.binom.cdf(X, 30, 0.5))
        theo_er.append(stats.binom.cdf(X, 30, 0.5)*2-1)

    print(theo_er)


# UDH watermarking method's (n=256) theoretical lower bound of Evasion Rate with respect to Detection Threshold \tau.
def UDH_theoretical():
    tau_list = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55]
    theo_er = []
    epsilon = 0.01
    n = 256
    for tau in tau_list:
        X = int((tau-epsilon) * n)
        print(stats.binom.cdf(X, 256, 0.5))
        theo_er.append(stats.binom.cdf(X, 256, 0.5)*2-1)

    print(theo_er)


if __name__ == '__main__':
    HiDDeN_theoretical()
    UDH_theoretical()