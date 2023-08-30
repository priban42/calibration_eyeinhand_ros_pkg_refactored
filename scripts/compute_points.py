#!/usr/bin/env python
import numpy as np
from numpy import pi
import os
#import tf.transformations as tft
import tf.transformations as tft

os.chdir(os.path.dirname(__file__))

def main():
    base_xyz = [-pi, 0, 0]
    points = [{'type':'plain', 'center':[0,0,0], 'offset': [0.4, 0, 0.6], 'orientation': 0},
              # {'type':'plain', 'center':[0,0,0], 'offset': [0.3, -0.4, 0.6], 'orientation': 0},
              # {'type':'plain', 'center':[0,0,0], 'offset': [0.3, 0.4, 0.6], 'orientation': 0},
              # {'type':'plain', 'center':[0,0,0], 'offset': [0.6, -0.4, 0.4], 'orientation': 0},
              # {'type':'plain', 'center':[0,0,0], 'offset': [0.6, 0.4, 0.4], 'orientation': 0},

              # {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.5, 'alpha': 3*pi/4, 'beta': pi/12, 'orientation':0},
              # {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.5, 'alpha': 3*pi/4, 'beta': pi/6, 'orientation':-pi/4},
              # {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.5, 'alpha': 3*pi/4, 'beta': pi/4, 'orientation':0},
              {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.5, 'alpha': 0, 'beta': 0, 'orientation':-pi/4},
              {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.5, 'alpha': pi/2, 'beta': pi/12, 'orientation':0},
              {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.5, 'alpha': pi/2, 'beta': pi/6, 'orientation':-pi/4},
              {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.5, 'alpha': pi/2, 'beta': pi/4, 'orientation':0},
              # {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.5, 'alpha': pi/4, 'beta': pi/12, 'orientation':0},
              # {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.5, 'alpha': pi/4, 'beta': pi/6, 'orientation':-pi/4},
              # {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.5, 'alpha': pi/4, 'beta': pi/4, 'orientation':0},

              {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.5, 'alpha': 0, 'beta': pi/6, 'orientation':0},
              {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.5, 'alpha': 0, 'beta': pi/12, 'orientation':pi/4},

              # {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.5, 'alpha': -pi/4, 'beta': pi/12, 'orientation':0},
              # {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.5, 'alpha': -pi/4, 'beta': pi/6, 'orientation':pi/4},
              # {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.5, 'alpha': -pi/4, 'beta': pi/4, 'orientation':0},
              {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.5, 'alpha': -pi/2, 'beta': pi/12, 'orientation':0},
              {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.5, 'alpha': -pi/2, 'beta': pi/6, 'orientation':pi/4},
              {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.5, 'alpha': -pi/2, 'beta': pi/4, 'orientation':0},
              # {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.5, 'alpha': -3*pi/4, 'beta': pi/12, 'orientation':0},
              # {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.5, 'alpha': -3*pi/4, 'beta': pi/6, 'orientation':pi/4},
              # {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.5, 'alpha': -3*pi/4, 'beta': pi/4, 'orientation':0},

              {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.5, 'alpha': 0, 'beta': -pi/24, 'orientation':0},
              {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.5, 'alpha': 0, 'beta': -pi/12, 'orientation':0},

              # Lower level
              {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.4, 'alpha': 0, 'beta': 0, 'orientation':0},
              {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.4, 'alpha': pi/2, 'beta': pi/12, 'orientation':pi/4},
              # {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.4, 'alpha': pi/4, 'beta': pi/24, 'orientation':0},

              {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.4, 'alpha': 0, 'beta': pi/6, 'orientation':0},
              {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.4, 'alpha': 0, 'beta': pi/12, 'orientation':0},

              # {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.4, 'alpha': -pi/4, 'beta': pi/12, 'orientation':0},
              {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.4, 'alpha': -pi/2, 'beta': pi/24, 'orientation':pi/4},

              {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.4, 'alpha': 0, 'beta': -pi/24, 'orientation':0},

              #Higher leve6
              {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.65, 'alpha': 0, 'beta': 0, 'orientation':0},
              # {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.65, 'alpha': 3*pi/4, 'beta': pi/12, 'orientation':0},
              {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.65, 'alpha': pi/2, 'beta': pi/12, 'orientation':0},
              {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.65, 'alpha': pi/2, 'beta': pi/6, 'orientation':pi/4},
              # {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.65, 'alpha': pi/4, 'beta': pi/12, 'orientation':0},

              {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.65, 'alpha': 0, 'beta': pi/12, 'orientation':0},
              {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.65, 'alpha': 0, 'beta': pi/24, 'orientation':0},

              # {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.65, 'alpha': -pi/4, 'beta': pi/12, 'orientation':0},
              {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.65, 'alpha': -pi/2, 'beta': pi/12, 'orientation':0},
              {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.65, 'alpha': -pi/2, 'beta': pi/6, 'orientation':pi/4},
              # {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.65, 'alpha': -3*pi/4, 'beta': pi/12, 'orientation':0},

              {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.65, 'alpha': 0, 'beta': -pi/24, 'orientation':0},
              {'type':'sphere','center':[0.4, 0, 0] , 'radius': 0.65, 'alpha': 0, 'beta': -pi/12, 'orientation':pi/4},

              # TESTING
              # {'type':'plain', 'center':[0,0,0], 'offset': [0.55, 0, 0.5], 'orientation': 0},
              # {'type':'plain', 'center':[0,0,0], 'offset': [0.4, 0.3, 0.5], 'orientation': pi/2},
              # {'type':'plain', 'center':[0,0,0], 'offset': [0.4, -0.2, 0.5], 'orientation': 0},
              # {'type':'plain', 'center':[0,0,0], 'offset': [0.4, -0.3, 0.5], 'orientation': pi/4},
              # {'type':'plain', 'center':[0,0,0], 'offset': [0.7, -0.1, 0.43], 'orientation': 0},
              # {'type':'plain', 'center':[0,0,0], 'offset': [0.4, 0.1, 0.43], 'orientation': pi/2},
              # {'type':'sphere','center':[0.3, 0, 0] , 'radius': 0.5, 'alpha': 0, 'beta': -pi/12, 'orientation':0},
              # {'type':'sphere','center':[0.3, 0, 0] , 'radius': 0.55, 'alpha': pi/2, 'beta': -pi/12, 'orientation':pi/3},
              # {'type':'sphere','center':[0.3, 0, 0] , 'radius': 0.6, 'alpha': -pi/2, 'beta': -pi/12, 'orientation':0},
              # {'type':'sphere','center':[0.3, 0, 0] , 'radius': 0.7, 'alpha': -3*pi/4, 'beta': -pi/12, 'orientation':0},
              # {'type':'sphere','center':[0.3, 0, 0] , 'radius': 0.7, 'alpha': 3*pi/4, 'beta': -pi/12, 'orientation':0},
              # {'type':'sphere','center':[0.3, 0, 0] , 'radius': 0.6, 'alpha': pi/4*0.75, 'beta': -pi/6, 'orientation':0},
    ]

    f = open("../positions/demo.txt", 'w')


    for point in points:
        if point['type']=='sphere':
            x = point['center'][0]+point['radius'] * np.cos(point['alpha']) * np.sin(point['beta'])
            y = point['center'][1]+point['radius'] * np.sin(point['alpha']) * np.sin(point['beta'])
            z = point['center'][2]+point['radius'] * np.cos(point['beta'])
            beta_x = point['beta']*np.sin(point['alpha'])
            beta_y = -1*point['beta']*np.cos(point['alpha'])
            q = tft.quaternion_from_euler(base_xyz[0], base_xyz[1]+beta_y, base_xyz[2]+beta_x  + point['orientation'], 'rxyz')
            pose = np.array([x, y, z, q[0], q[1], q[2], q[3]])
        elif point['type']=='plain':
            x = point['center'][0] + point['offset'][0]
            y = point['center'][1] + point['offset'][1]
            z = point['center'][2] + point['offset'][2]
            q = tft.quaternion_from_euler(base_xyz[0], base_xyz[1], base_xyz[2]+point['orientation'], 'rxyz')
            pose = np.array([x, y, z, q[0], q[1], q[2], q[3]])
        else:
            continue
        for p in pose:
            f.write(str(p))
            f.write(' ')
        f.write('\n')
    f.close()

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1
if __name__ == '__main__':
    main()
