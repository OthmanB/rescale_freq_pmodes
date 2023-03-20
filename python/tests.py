import numpy as np

fl0=[2495.06,2691.95, 2896.78 , 3096.4 ,3296.63 ,3491.31, 3690.47, 3890.79, 4091.62 ,4286.91 ,4484.15 ,4685.95, 4888.46, 5086.44, 5280.94,
      5484.5, 5680.22, 5879.27 , 6077.7 , 6281.7]

n=np.asarray([12, 13 ,14 ,15 ,16, 17, 18,19, 20 ,21 ,22, 23, 24 ,25, 26, 27 ,28 ,29 ,30 ,31])
Dnu=199.077
epsilon=0.541345
d0l= -2.45919
O2=np.asarray([-1.6273 , -3.81776,   1.93901 ,  2.47712  , 3.63662 ,-0.763757 ,-0.678889 , 0.561839 ,  2.31551    , -1.47  ,-3.30598,
     -0.584018 ,  2.84581  , 1.75473  ,-2.82109 ,  1.65873  ,-1.70207 , -1.72038 ,-2.37137 ,  2.55115])

nu=(n+epsilon)*Dnu + O2