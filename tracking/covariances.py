

class NuScenesCovariances:
    P = {
        'bicycle': {'x': 0.05390982, 'y': 0.05039431, 'z': 0.01863044, 'yaw': 1.29464435,
                    'l': 0.02713823, 'w': 0.01169572, 'h': 0.01295084,
                    'dx': 0.04560422, 'dy': 0.04097244, 'dz': 0.01725477, 'dyaw': 1.21635902},
        'bus': {'x': 0.17546469, 'y': 0.13818929, 'z': 0.05947248, 'yaw': 0.1979503,
                'l': 0.78867322, 'w': 0.05507407, 'h': 0.06684149,
                'dx': 0.13263319, 'dy': 0.11508148, 'dz': 0.05033665, 'dyaw': 0.22529652},
        'car': {'x': 0.08900372, 'y': 0.09412005, 'z': 0.03265469, 'yaw': 1.00535696,
                'l': 0.10912802, 'w': 0.02359175, 'h': 0.02455134,
                'dx': 0.08120681, 'dy': 0.08224643, 'dz': 0.02266425, 'dyaw': 0.99492726},
        'motorcycle': {'x': 0.04052819, 'y': 0.0398904, 'z': 0.01511711, 'yaw': 1.06442726,
                       'l': 0.03291016, 'w': 0.00957574, 'h': 0.0111605,
                       'dx': 0.0437039, 'dy': 0.04327734, 'dz': 0.01465631, 'dyaw': 1.30414345},
        'pedestrian': {'x': 0.03855275, 'y': 0.0377111, 'z': 0.02482115, 'yaw': 2.0751833,
                       'l': 0.02286483, 'w': 0.0136347, 'h': 0.0203149,
                       'dx': 0.04237008, 'dy': 0.04092393, 'dz': 0.01482923, 'dyaw': 2.0059979},
        'trailer': {'x': 0.23228021, 'y': 0.22229261, 'z': 0.07006275, 'yaw': 1.05163481,
                    'l': 1.37451601, 'w': 0.06354783, 'h': 0.10500918,
                    'dx': 0.2138643, 'dy': 0.19625241, 'dz': 0.05231335, 'dyaw': 0.97082174},
        'truck': {'x': 0.14862173, 'y': 0.1444596, 'z': 0.05417157, 'yaw': 0.73122169,
                  'l': 0.69387238, 'w': 0.05484365, 'h': 0.07748085,
                  'dx': 0.10683797, 'dy': 0.10248689, 'dz': 0.0378078, 'dyaw': 0.76188901}
    }

    Q = {
        'bicycle': {'x': 1.98881347e-02, 'y': 1.36552276e-02, 'z': 5.10175742e-03, 'yaw': 1.33430252e-01,
                    'l': 0, 'w': 0, 'h': 0,
                    'dx': 1.98881347e-02, 'dy': 1.36552276e-02, 'dz': 5.10175742e-03, 'dyaw': 1.33430252e-01},
        'bus': {'x': 1.17729925e-01, 'y': 8.84659079e-02, 'z': 1.17616440e-02, 'yaw': 2.09050032e-01,
                'l': 0, 'w': 0, 'h': 0,
                'dx': 1.17729925e-01, 'dy': 8.84659079e-02, 'dz': 1.17616440e-02, 'dyaw': 2.09050032e-01},
        'car': {'x': 1.58918523e-01, 'y': 1.24935318e-01, 'z': 5.35573165e-03, 'yaw': 9.22800791e-02,
                'l': 0, 'w': 0, 'h': 0,
                'dx': 1.58918523e-01, 'dy': 1.24935318e-01, 'dz': 5.35573165e-03, 'dyaw': 9.22800791e-02},
        'motorcycle': {'x': 3.23647590e-02, 'y': 3.86650974e-02, 'z': 5.47421635e-03, 'yaw': 2.34967407e-01,
                       'l': 0, 'w': 0, 'h': 0,
                       'dx': 3.23647590e-02, 'dy': 3.86650974e-02, 'dz': 5.47421635e-03, 'dyaw': 2.34967407e-01},
        'pedestrian': {'x': 3.34814566e-02, 'y': 2.47354921e-02, 'z': 5.94592529e-03, 'yaw': 4.24962535e-01,
                       'l': 0, 'w': 0, 'h': 0,
                       'dx': 3.34814566e-02, 'dy': 2.47354921e-02, 'dz': 5.94592529e-03, 'dyaw': 4.24962535e-01},
        'trailer': {'x': 4.19985099e-02, 'y': 3.68661552e-02, 'z': 1.19415050e-02, 'yaw': 5.63166240e-02,
                    'l': 0, 'w': 0, 'h': 0,
                    'dx': 4.19985099e-02, 'dy': 3.68661552e-02, 'dz': 1.19415050e-02, 'dyaw': 5.63166240e-02},
        'truck': {'x': 9.45275998e-02, 'y': 9.45620374e-02, 'z': 8.38061721e-03, 'yaw': 1.41680460e-01,
                  'l': 0, 'w': 0, 'h': 0,
                  'dx': 9.45275998e-02, 'dy': 9.45620374e-02, 'dz': 8.38061721e-03, 'dyaw': 1.41680460e-01}
    }

    R = {
        'bicycle': {'x': 0.05390982, 'y': 0.05039431, 'z': 0.01863044, 'yaw': 1.29464435,
                    'l': 0.02713823, 'w': 0.01169572, 'h': 0.01295084},
        'bus': {'x': 0.17546469, 'y': 0.13818929, 'z': 0.05947248, 'yaw': 0.1979503,
                'l': 0.78867322, 'w': 0.05507407, 'h': 0.06684149},
        'car': {'x': 0.08900372, 'y': 0.09412005, 'z': 0.03265469, 'yaw': 1.00535696,
                'l': 0.10912802, 'w': 0.02359175, 'h': 0.02455134},
        'motorcycle': {'x': 0.04052819, 'y':0.0398904, 'z': 0.01511711, 'yaw': 1.06442726,
                       'l': 0.03291016, 'w':0.00957574, 'h': 0.0111605},
        'pedestrian': {'x': 0.03855275, 'y': 0.0377111, 'z': 0.02482115, 'yaw': 2.0751833,
                       'l': 0.02286483, 'w': 0.0136347, 'h': 0.0203149},
        'trailer': {'x': 0.23228021, 'y': 0.22229261, 'z': 0.07006275, 'yaw': 1.05163481,
                    'l': 1.37451601, 'w': 0.06354783, 'h': 0.10500918},
        'truck': {'x': 0.14862173, 'y': 0.1444596, 'z': 0.05417157, 'yaw': 0.73122169,
                  'l': 0.69387238, 'w': 0.05484365, 'h': 0.07748085}
    }
