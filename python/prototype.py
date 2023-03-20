import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored

def freq_ref():
    fl0=[1287.7378, 1391.6472, 1494.9822, 1598.7040, 1700.9117, 1802.3264,1904.5894, 2007.5698,
        2110.8952, 2214.2624, 2317.2959, 2420.9247, 2524.9409, 2629.3219, 2734.3085, 2839.1342,
        2944.6692, 3048.5000]
    fl1=[1334.1481,
            1437.7227,
            1541.9739,
            1644.9908,
            1747.1738,
            1849.0120,
            1952.0236,
            2055.4881,
            2159.1390,
            2262.5176,
            2366.1802,
            2470.2896,
            2574.7077,
            2679.4912,
            2784.2952,
            2890.0152,
            2995.1098]
    fl2=[1279.5453,
            1383.9922,
            1487.7844,
            1591.2323,
            1694.0395,
            1795.8284,
            1898.3264,
            2001.7400,
            2105.3332,
            2208.9378,
            2312.4974,
            2416.3217,
            2520.5174,
            2624.8247,
            2730.0274,
            2836.0081,
            2942.1468,
            3046.5000]
    fl3=[1427.44580,
            1530.34899,
            1632.59738,
            1735.49943,
            1838.42350,
            1941.60645,
            2045.80496,
            2149.91448,
            2253.57221,
            2357.48682,
            2461.82262,
            2566.53772,
            2670.15963,
            2774.16196]
    Amp_l0=[ 0.43002206, 0.57919905, 1.0561653, 1.2720441, 1.8095614, 3.1660368,
        4.6474355, 6.7184649, 8.7121034, 7.6455132, 4.8872829, 2.6154045, 1.0578662,
        0.45623723, 0.22393687, 0.084068007, 0.041473324, 0.010000000]
    Amp_l1=[0.62326858, 
        0.83951082 ,
        1.5326199 ,
        1.8440937 ,
        2.6226760 ,
        4.5863581 ,
        6.7292735 ,
        9.7313251 ,
        12.627343 ,
        11.072803 ,
        7.0797880 ,
        3.7891896 ,
        1.5327583 ,
        0.66107875 ,
        0.32461789 ,
        0.12177273 ,
        0.060119340 ]
    Amp_l2=[ 0.28319571,
        0.38247189,
        0.69652430,
        0.83868864,
        1.1919224,
        2.0867683,
        3.0624058,
        4.4289445,
        5.7391074,
        5.0402307,
        3.2178249,
        1.7239335,
        0.69726538,
        0.30092180,
        0.14755212,
        0.055328129,
        0.027332655,
        0.00500000 ]
    Amp_l3=[0.09032,
            0.14734,
            0.11641,
            0.17580,
            0.30536,
            0.45811,
            0.61920,
            1.00981,
            0.84437,
            0.51441,
            0.22659,
            0.09991,
            0.04100,
            0.02104]
    return fl0, fl1, fl2, fl3, Amp_l0, Amp_l1, Amp_l2, Amp_l3

def rescale_freqs(Dnu_star, epsilon_star, fl0_ref, fl1=[], fl2=[], fl3=[], dl1_star=None, dl2_star=None, dl3_star=None):
    '''
        A prototype of rescaling function that. It decompose the frequency by identifying all of the terms of the asymptotic
        in order to isolate the residual error term. Then it uses this residual error (with a proper rescaling) to generate a new set 
        of frequencies following the asymptotic with the desired Dnu_star, epsilon_star, dl1_star (d01), dl2_star (d02), dl3_star (d13).
        Finally, a consistency check is made by re-decomposing the rescaled frequency so that you can check that the rescaling process
        worked as expected
    '''
    # Compute Dnu and epsilon
    n_ref=np.linspace(0, len(fl0_ref)-1, len(fl0_ref))
    f=np.polyfit(n_ref, fl0_ref, 1)
    Dnu_ref=f[0]
    epsilon_ref, n0_ref=np.modf(f[1]/f[0])

    # Rescaling and epsilon shifting l=0
    fl0_star=(fl0_ref/Dnu_ref - epsilon_ref + epsilon_star) * Dnu_star
    n_star=np.linspace(0, len(fl0_star)-1, len(fl0_star))
    fstar=np.polyfit(n_star, fl0_star, 1)
    Dnu_l=fstar[0]
    epsilon_star, n0_star=np.modf(fstar[1]/fstar[0])
    n_star=np.linspace(n0_star, n0_star + len(fl0_star)-1, len(fl0_star))
    
    if fl1 !=[]:
        if dl1_star == None:
            print("Error: fl1 and dl1_star must be jointly set! Otherwise do not pass them as arguments")
            print("       Cannot rescale")
            exit()
        l=1
        # Rescaling the fl_ref:
    #       1. Extract all of asymtptotic elements. In particular nl and O2_l
        nl1, Dnu_l1, epsilon_l1, d01, O2_l1=decompose_nu_nl(l, fl0_ref, fl=fl1, Cfactor=0.25, verbose=False)
    #       2. Reconstruct a new relation using the requested Dnu_star and epsilon_star along with nl and O2_l*Dnu_star/Dnu_ref
        fl1_star=(nl1 + epsilon_star + l/2)*Dnu_star + dl1_star + O2_l1*Dnu_star/Dnu_ref
    else:
        fl1_star=[]
    #
    if fl2 !=[]:
        if dl2_star == None:
            print("Error: fl2 and dl2_star must be jointly set! Otherwise do not pass them as arguments")
            print("       Cannot rescale")
            exit()
        l=2
        # Rescaling the fl_ref:
    #       1. Extract all of asymtptotic elements. In particular nl and O2_l
        nl2, Dnu_l2, epsilon_l2, d02, O2_l2=decompose_nu_nl(l, fl0_ref, fl=fl2, Cfactor=0.25, verbose=False)
    #       2. Reconstruct a new relation using the requested Dnu_star and epsilon_star along with nl and O2_l*Dnu_star/Dnu_ref
        fl2_star=(nl2 + epsilon_star + l/2)*Dnu_star + dl2_star + O2_l2*Dnu_star/Dnu_ref
    else:
        fl2_star=[]
    #
    if fl3 !=[]:
        if dl3_star == None:
            print("Error: fl3 and dl3_star must be jointly set! Otherwise do not pass them as arguments")
            print("       Cannot rescale")
            exit()
        l=3
        # Rescaling the fl_ref:
    #       1. Extract all of asymtptotic elements. In particular nl and O2_l
        nl3, Dnu_l3, epsilon_l3, d03, O2_l3=decompose_nu_nl(l, fl0_ref, fl=fl3, Cfactor=0.25, verbose=False)
    #       2. Reconstruct a new relation using the requested Dnu_star and epsilon_star along with nl and O2_l*Dnu_star/Dnu_ref
        fl3_star=(nl3 + epsilon_star + l/2)*Dnu_star + dl3_star + O2_l3*Dnu_star/Dnu_ref
    else:
        fl3_star=[]
    print(colored("       ----- Reference star -----", "blue"))
    print("Dnu_star/Dnu_ref = ", Dnu_star/Dnu_ref)
    print("Dnu_ref: ", Dnu_ref)
    print("epsilon_ref: ", epsilon_ref)
    print(colored('       --- Test l=0 ---', "blue"))
#    print("epsilon_l0:", epsilon_star)
#    print("Dnu_l0    :", fstar[0])
#    print("nl0 : ", n_star)
    k=decompose_nu_nl(0, fl0_star, fl=[], Cfactor=0.25, verbose=False) #nl, Dnu_l, epsilon_l, d0l, O2_l
    print(' Dnu_in    : {0:0.4f}   Dnu    : {1:0.4f}    delta: {2:0.5f}'.format(Dnu_star, k[1], Dnu_star-k[1]))
    print(' epsilon_in: {0:0.6f}   epsilon: {1:0.6f}    delta: {2:0.7f}'.format(epsilon_star, k[2], epsilon_star-k[2]))
    print(' d0l_in    : {0:0.6f}   d0l    : {1:0.6f}    delta: {2:0.7f}'.format(0, k[3], 0-k[3]))
    print(' O2l_in: ')
    print('     Not calculated')
    print(' O2l: ')
    print('     ', k[4])
    print('---')
    print(colored('       --- Test l=1 ---', "blue"))
#    print("nl1 : ", nl1)
#    print("O2_l1 : ", O2_l1)
#    print("fl1_star :", fl1_star)
    k=decompose_nu_nl(1, fl0_star, fl=fl1_star, Cfactor=0.25, verbose=False) #nl, Dnu_l, epsilon_l, d0l, O2_l
    print(' Dnu_in    : {}   Dnu    : {}    delta: {}'.format(Dnu_star, k[1], Dnu_star-k[1]))
    print(' epsilon_in: {}   epsilon: {}    delta: {}'.format(epsilon_star, k[2], epsilon_star-k[2]))
    print(' d0l_in    : {}   d0l    : {}    delta: {}'.format(dl1_star, k[3], dl1_star-k[3]))
    print(' O2l_in: ')
    print('     ', O2_l1)
    print(' new O2l*Dnu_ref/Dnu_star: ')
    print('     ', k[4]*Dnu_ref/Dnu_star)
    print('---')
    print(colored('       --- Test l=2 ---', "blue"))
#    print("nl2 : ", nl2)
#    print("O2_l2 : ", O2_l2)
#    print("fl2_star :", fl2_star)
    k=decompose_nu_nl(2, fl0_star, fl=fl2_star, Cfactor=0.25, verbose=False)
    print(' Dnu_in    : {0:0.4f}   Dnu    : {1:0.4f}    delta: {2:0.5f}'.format(Dnu_star, k[1], Dnu_star-k[1]))
    print(' epsilon_in: {0:0.6f}   epsilon: {1:0.6f}    delta: {2:0.7f}'.format(epsilon_star, k[2], epsilon_star-k[2]))
    print(' d0l_in    : {0:0.6f}   d0l    : {1:0.6f}    delta: {2:0.7f}'.format(dl2_star, k[3], dl2_star-k[3]))
    print(' O2l_in: ')
    print('     ', O2_l2)
    print(' new O2l*Dnu_ref/Dnu_star: ')
    print('     ', k[4]*Dnu_ref/Dnu_star)
    print('---')
    print(colored('       --- Test l=3 ---', "blue"))
#    print("nl3 : ", nl3)
#    print("O2_l3 : ", O2_l3)
#    print("fl3_star :", fl3_star)
    k=decompose_nu_nl(3, fl0_star, fl=fl3_star, Cfactor=0.25, verbose=False)
    print(' Dnu_in    : {0:0.4f}   Dnu    : {1:0.4f}    delta: {2:0.5f}'.format(Dnu_star, k[1], Dnu_star-k[1]))
    print(' epsilon_in: {0:0.6f}   epsilon: {1:0.6f}    delta: {2:0.7f}'.format(epsilon_star, k[2], epsilon_star-k[2]))
    print(' d0l_in    : {0:0.6f}   d0l    : {1:0.6f}    delta: {2:0.7f}'.format(dl3_star, k[3], dl3_star-k[3]))
    print(' O2l_in: ')
    print('     ', O2_l3)
    print(' new O2l*Dnu_ref/Dnu_star: ')
    print('     ', k[4]*Dnu_ref/Dnu_star)
    print(' n_ref   n_star   O2_ref   O2_star  ratio')
    for i in range(len(O2_l3)):
        print("{0:0.0f}  {1:0.4f}  {2:0.4f}  {3:0.4f}   {4:0.4f}".format(n_ref[i]+n0_ref, k[0][i], O2_l3[i], k[4][i], k[4][i]/O2_l3[i]))
    print('---')
    return fl0_star, fl1_star, fl2_star, fl3_star

def decompose_nu_nl(l, fl0, fl=[], Cfactor=0.25, verbose=False):
    '''
        Function that takes the p modes frequencies and decompose them
        ie, identifies, Dnu, epsilon, n, d0l and O(2) (higher order) terms
        This allows to extract the different contribution to the modes 
        and use the following properties:
            - Dnu and epsilon are determined by the l=0 frequencies
            - d0l + O(2) and the radial order n are determined by solving analytically the linear fit
                to the frequencies at the fixed pre-determined value of the slope = Dnu
            - O(2) is determined by assuming it to be a 0-mean perturbation around d0l
    '''
    if fl == []: # In that case, we decompose for the l=0
        fl=fl0
        l=0
    #
    # Compute Dnu and epsilon
    n_ref=np.linspace(0, len(fl0)-1, len(fl0))
    f=np.polyfit(n_ref, fl0, 1)
    Dnu=f[0]
    epsilon, n0_0=np.modf(f[1]/f[0])
    # Least square on l=1 frequencies with a fix slope
    #    1. Isolate d0l + O(2)
    func=fl - (epsilon + l/2)*Dnu # This is Dnu_ref.n + d0l + O(2)
    #    2. Determine a first guess for n
    #e0, n_l=np.modf(func/Dnu_ref) # This does not always works when e0 is close to 0 or 1
    e0, n0_l=np.modf(func[0]/Dnu)
    n_l=np.linspace(n0_l, n0_l + len(func)-1, len(func))
    #    3. Make a list of adjacent potential n and find the one that ensure |Dnu_ref| >> |d0l| + O(2) and d0l < 0
    n_all=np.zeros((4,len(n_l)))
    n_all[0,:]=n_l-np.repeat(1, len(n_l))
    n_all[1,:]=n_l
    n_all[2,:]=n_l+np.repeat(1,len(n_l))
    n_all[3,:]=n_l+np.repeat(2,len(n_l))
    sol=np.zeros(4)
    for n_s in range(4):
        sol[n_s]=np.mean(func) - Dnu*np.mean(n_all[n_s,:])
        #print(func - n_all[n_s,:]*Dnu_ref)
    print("func = ", func)
    print("sol = ", sol)
    #print('--')
    # For debug only
    '''
    for i in range(len(fl)-1):
        print(fl[i+1]-fl[i])
    plt.plot(n_all[2,:], func, marker='o')
    for line in range(12,30):
        plt.plot([line,line],[0, 3000], color='gray')
    plt.show()
    exit()
    '''
    # EndFor debug
    #n_s_optimum=np.where(np.bitwise_and(sol <= 0, np.abs(sol) < Dnu_ref*Cfactor))[0] # Identify the negative solutions. This is to restrictive
    n_s_optimum=np.where(np.abs(sol) < Dnu*Cfactor)[0] # Identify solutions close to 0 as |d0l|<<|Dnu_ref|
    n_best=n_all[n_s_optimum,:].flatten() #This is the list of identified n following the conditions in the where
    d0l=sol[n_s_optimum] # This is the best solution for d0l + O(2)
    if len(d0l) > 1:
        print('Error: multiple solutions of d0l found')
        print('       Debug required')
        print('      d0l =', d0l)
        exit()
    else:
        d0l=d0l[0]
    #    4. Identify the residual variation by assuming that O(2) is a random term of 0-mean
    O2_term=func- n_best*Dnu - np.repeat(d0l, len(n_best))
    #print(func)
    #print(func- n_best*Dnu_ref)
    #exit()
    if verbose == True:
        print('--')
        print('Best list of n matching conditions |Dnu_ref| >> |d0l| + O(2) and d0l < 0:')
        print(n_best)
        print('---')
        print('Identified best solution for d0l:')
        print(d0l)
        print('---')
        print('Residual of the constrained fit:')
        print(O2_term)
    #print("mean(O2):", np.mean(O2_term))
    #plt.plot(fl1, O2_term)
    #plt.show()
    #exit()
    return n_best, Dnu, epsilon, d0l, O2_term


def test(Dnu_star=80, epsilon_star=0.5, dl1_star=0, dl2_star=0, dl3_star=0):
    l=1
    fl0, fl1, fl2,fl3, Amp_l0, Amp_l1, Amp_l2, Amp_l3=freq_ref()
    fl0_star, fl1_star, fl2_star, fl3_star=rescale_freqs(Dnu_star, epsilon_star, fl0, fl1=fl1, fl2=fl2, fl3=fl3, 
                                                        dl1_star=dl1_star, dl2_star=dl2_star, dl3_star=dl3_star)
    exit()

def test_decompose():
    fl0, fl1, fl2, fl3, Amp_l0, Amp_l1, Amp_l2, Amp_l3=freq_ref()
    n,Dnu,epsilon, d0l, O2_term=decompose_nu_nl(0, fl0, fl=fl0, Cfactor=0.25, verbose=True)
    d=np.gradient(fl0)
    fl0_rebuilt=(n + epsilon)*Dnu  + O2_term
    fl0_rebuilt_deriv=(n+ epsilon)*np.median(d) + (d-np.median(d))
    print("fl0 =", fl0)
    print("fl0_rebuilt =", fl0_rebuilt)
    print("fl0_rebuilt_deriv = ", fl0_rebuilt_deriv)
    plt.plot(fl0, np.median(d)-d, label='O2_deriv')
    plt.plot([fl0[0],fl0[1]], [0, 0], linestyle='--')
    plt.plot(fl0, O2_term, label='O2_term')
    plt.legend()
    plt.show()