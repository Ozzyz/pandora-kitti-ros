import numpy as np
import os

KtoP = np.array([
    [0, 1, 0, 0],
    [-1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

PtoK = np.array([
    [0, -1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
    ])

RT = np.array([
    [-0.9990905,  0.0155914,  0.0396873, 0.02067],
    [-0.0396262,  0.0042089, -0.9992057, -0.095356],
    [-0.0157461, -0.9998696, -0.0035873, -0.108349],
    [0, 0, 0, 1]
])

RTInv = np.linalg.inv(RT)

R = np.array([
    [-1, 0, 0, 0],
    [0, 0, -1, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1]
])

RInv = np.linalg.inv(R)

velo_to_cam = np.array([
    [0, -1, 0, 0],
    [0, 0, -1, 0], 
    [1, 0, 0, 0],
    [0, 0, 0, 1]
])

combined = np.dot(PtoK, np.dot(R, np.dot(RTInv, KtoP)))
v2c = np.dot(velo_to_cam, combined)

v2c_str = ""

for v in v2c.flatten():
    v2c_str += str(v) + " "

print(v2c_str)

K = np.array([
    [1461.317641253868, 0.0, 650.5533608051251, 0],
    [0.0, 1447.137849649878, 355.3033160801929, 0],
    [0.0, 0.0, 1.0, 0.0],
    [0, 0, 0, 1]
])

P2 = K
print(P2)

P2_str = ""
for v in P2.flatten():
    P2_str += str(v) + " "

print(P2_str)