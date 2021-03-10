import numpy as np
N = 50
inn = np.random.rand(999, 1)
a = inn.shape[0]
c = int(a/N)
print(c)
out = []
for i in range(N):
    aux = inn[i*c:(i+1)*c]
    if i == N-1:
        aux = inn[i*c:]
    out.append(np.mean(np.array(aux).reshape(-1,1)))

print(out)
print(len(out))