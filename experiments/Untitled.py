#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 1 - 120
# 01 16 31 46
# 02 17 32 47


# In[28]:


import numpy as np


# In[64]:


# criando sequencia da passagem vertical das questoes
# 1, ..., 15, 61, ..., 75, 16, ..., 30, ...
seq = []
q = 1
for _ in range(1, 121):
    seq.append(q)
    if q % 15 == 0:
        if q > 60:
            q -= 59
        else:
            q += 46
    else:
        q += 1

# dividindo sequencia em 4 grupos
# cada grupo é replicado em 5 vezes conseguintes, criando a sequencia final de 600 posicoes
nseq = []
for chunk in np.array_split(seq, 4):
    a = np.concatenate([chunk for _ in range(0, 5)])
    nseq += list(a)

# identificando alternativa a cada valor da sequencia
alt_i = 0
alt = "ABCDE"
for i in range(len(nseq)):
    nseq[i] = (nseq[i], alt[alt_i])

    if (i + 1) % 30 == 0:
        if alt_i == 4:
            alt_i = 0
        else:
            alt_i += 1


# In[65]:


nseq


# In[45]:


a = chunks[0]
a = np.concatenate([a for i in range(0, 5)])
a


# In[ ]:


# In[ ]:
