import numpy as np

T = 1000
x0 = 5
beta_curr = 0.00001
prev_x = x0

for t in range(1, T):
    mu, sigma = np.sqrt(1 - beta_curr) * prev_x, beta_curr * 1
    xt = np.random.normal(mu, sigma, 1)
    print(f"t= {t}, xt= {xt}")
    prev_x = xt
    beta_curr += 0.00001


