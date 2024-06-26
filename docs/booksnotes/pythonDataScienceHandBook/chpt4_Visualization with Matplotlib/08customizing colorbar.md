================
by Jawad Haider

# **Chpt 4 - Visualization with Matplotlib**

# 08 -  Customizing Colorbars
------------------------------------------------------------------------

- <a href="#customizing-colorbars"
  id="toc-customizing-colorbars">Customizing Colorbars</a>
  - <a href="#customizing-colorbars-1"
    id="toc-customizing-colorbars-1">Customizing Colorbars</a>
  - <a href="#choosing-the-colormap" id="toc-choosing-the-colormap">Choosing
    the colormap</a>
    - <a href="#color-limits-and-extensions"
      id="toc-color-limits-and-extensions">Color limits and extensions</a>
  - <a href="#discrete-colorbars" id="toc-discrete-colorbars">Discrete
    colorbars</a>

------------------------------------------------------------------------

# Customizing Colorbars

Plot legends identify discrete labels of discrete points. For continuous
labels based on the color of points, lines, or regions, a labeled
colorbar can be a great tool. In Mat‐ plotlib, a colorbar is a separate
axes that can provide a key for the meaning of colors in a plot.

``` python
import matplotlib.pyplot as plt
plt.style.use('classic')
%matplotlib inline
import numpy as np
```

``` python
x=np.linspace(0,10,1000)
I = np.sin(x)*np.cos(x[:,np.newaxis])
```

``` python
np.cos(x[:,np.newaxis])
```

    array([[ 1.00000000e+00],
           [ 9.99949900e-01],
           [ 9.99799606e-01],
           [ 9.99549133e-01],
           [ 9.99198505e-01],
           [ 9.98747758e-01],
           [ 9.98196937e-01],
           [ 9.97546097e-01],
           [ 9.96795304e-01],
           [ 9.95944632e-01],
           [ 9.94994167e-01],
           [ 9.93944004e-01],
           [ 9.92794249e-01],
           [ 9.91545016e-01],
           [ 9.90196431e-01],
           [ 9.88748629e-01],
           [ 9.87201754e-01],
           [ 9.85555963e-01],
           [ 9.83811419e-01],
           [ 9.81968298e-01],
           [ 9.80026785e-01],
           [ 9.77987073e-01],
           [ 9.75849367e-01],
           [ 9.73613882e-01],
           [ 9.71280841e-01],
           [ 9.68850478e-01],
           [ 9.66323038e-01],
           [ 9.63698772e-01],
           [ 9.60977944e-01],
           [ 9.58160826e-01],
           [ 9.55247701e-01],
           [ 9.52238861e-01],
           [ 9.49134607e-01],
           [ 9.45935251e-01],
           [ 9.42641112e-01],
           [ 9.39252521e-01],
           [ 9.35769817e-01],
           [ 9.32193350e-01],
           [ 9.28523478e-01],
           [ 9.24760567e-01],
           [ 9.20904997e-01],
           [ 9.16957152e-01],
           [ 9.12917429e-01],
           [ 9.08786232e-01],
           [ 9.04563975e-01],
           [ 9.00251081e-01],
           [ 8.95847982e-01],
           [ 8.91355120e-01],
           [ 8.86772945e-01],
           [ 8.82101915e-01],
           [ 8.77342499e-01],
           [ 8.72495174e-01],
           [ 8.67560426e-01],
           [ 8.62538748e-01],
           [ 8.57430645e-01],
           [ 8.52236627e-01],
           [ 8.46957216e-01],
           [ 8.41592940e-01],
           [ 8.36144337e-01],
           [ 8.30611953e-01],
           [ 8.24996341e-01],
           [ 8.19298066e-01],
           [ 8.13517698e-01],
           [ 8.07655815e-01],
           [ 8.01713006e-01],
           [ 7.95689865e-01],
           [ 7.89586997e-01],
           [ 7.83405013e-01],
           [ 7.77144531e-01],
           [ 7.70806181e-01],
           [ 7.64390596e-01],
           [ 7.57898419e-01],
           [ 7.51330302e-01],
           [ 7.44686901e-01],
           [ 7.37968884e-01],
           [ 7.31176922e-01],
           [ 7.24311697e-01],
           [ 7.17373896e-01],
           [ 7.10364214e-01],
           [ 7.03283355e-01],
           [ 6.96132027e-01],
           [ 6.88910947e-01],
           [ 6.81620838e-01],
           [ 6.74262431e-01],
           [ 6.66836464e-01],
           [ 6.59343679e-01],
           [ 6.51784829e-01],
           [ 6.44160671e-01],
           [ 6.36471968e-01],
           [ 6.28719491e-01],
           [ 6.20904016e-01],
           [ 6.13026327e-01],
           [ 6.05087214e-01],
           [ 5.97087471e-01],
           [ 5.89027900e-01],
           [ 5.80909308e-01],
           [ 5.72732510e-01],
           [ 5.64498325e-01],
           [ 5.56207577e-01],
           [ 5.47861097e-01],
           [ 5.39459722e-01],
           [ 5.31004293e-01],
           [ 5.22495658e-01],
           [ 5.13934670e-01],
           [ 5.05322185e-01],
           [ 4.96659067e-01],
           [ 4.87946184e-01],
           [ 4.79184410e-01],
           [ 4.70374621e-01],
           [ 4.61517701e-01],
           [ 4.52614537e-01],
           [ 4.43666022e-01],
           [ 4.34673051e-01],
           [ 4.25636526e-01],
           [ 4.16557353e-01],
           [ 4.07436441e-01],
           [ 3.98274704e-01],
           [ 3.89073060e-01],
           [ 3.79832432e-01],
           [ 3.70553744e-01],
           [ 3.61237927e-01],
           [ 3.51885914e-01],
           [ 3.42498642e-01],
           [ 3.33077052e-01],
           [ 3.23622088e-01],
           [ 3.14134698e-01],
           [ 3.04615831e-01],
           [ 2.95066442e-01],
           [ 2.85487487e-01],
           [ 2.75879926e-01],
           [ 2.66244723e-01],
           [ 2.56582842e-01],
           [ 2.46895251e-01],
           [ 2.37182922e-01],
           [ 2.27446827e-01],
           [ 2.17687942e-01],
           [ 2.07907245e-01],
           [ 1.98105716e-01],
           [ 1.88284337e-01],
           [ 1.78444091e-01],
           [ 1.68585966e-01],
           [ 1.58710948e-01],
           [ 1.48820028e-01],
           [ 1.38914196e-01],
           [ 1.28994445e-01],
           [ 1.19061768e-01],
           [ 1.09117162e-01],
           [ 9.91616222e-02],
           [ 8.91961465e-02],
           [ 7.92217334e-02],
           [ 6.92393823e-02],
           [ 5.92500934e-02],
           [ 4.92548678e-02],
           [ 3.92547068e-02],
           [ 2.92506125e-02],
           [ 1.92435873e-02],
           [ 9.23463398e-03],
           [-7.75244699e-04],
           [-1.07850457e-02],
           [-2.07937660e-02],
           [-3.08004029e-02],
           [-4.08039535e-02],
           [-5.08034156e-02],
           [-6.07977872e-02],
           [-7.07860669e-02],
           [-8.07672539e-02],
           [-9.07403481e-02],
           [-1.00704350e-01],
           [-1.10658262e-01],
           [-1.20601085e-01],
           [-1.30531825e-01],
           [-1.40449485e-01],
           [-1.50353072e-01],
           [-1.60241594e-01],
           [-1.70114060e-01],
           [-1.79969480e-01],
           [-1.89806868e-01],
           [-1.99625237e-01],
           [-2.09423604e-01],
           [-2.19200987e-01],
           [-2.28956405e-01],
           [-2.38688883e-01],
           [-2.48397444e-01],
           [-2.58081116e-01],
           [-2.67738928e-01],
           [-2.77369913e-01],
           [-2.86973105e-01],
           [-2.96547543e-01],
           [-3.06092267e-01],
           [-3.15606321e-01],
           [-3.25088751e-01],
           [-3.34538607e-01],
           [-3.43954943e-01],
           [-3.53336815e-01],
           [-3.62683283e-01],
           [-3.71993410e-01],
           [-3.81266263e-01],
           [-3.90500914e-01],
           [-3.99696437e-01],
           [-4.08851910e-01],
           [-4.17966417e-01],
           [-4.27039043e-01],
           [-4.36068881e-01],
           [-4.45055025e-01],
           [-4.53996574e-01],
           [-4.62892633e-01],
           [-4.71742311e-01],
           [-4.80544720e-01],
           [-4.89298979e-01],
           [-4.98004210e-01],
           [-5.06659542e-01],
           [-5.15264107e-01],
           [-5.23817042e-01],
           [-5.32317492e-01],
           [-5.40764603e-01],
           [-5.49157530e-01],
           [-5.57495432e-01],
           [-5.65777473e-01],
           [-5.74002823e-01],
           [-5.82170659e-01],
           [-5.90280161e-01],
           [-5.98330518e-01],
           [-6.06320922e-01],
           [-6.14250574e-01],
           [-6.22118677e-01],
           [-6.29924445e-01],
           [-6.37667095e-01],
           [-6.45345851e-01],
           [-6.52959943e-01],
           [-6.60508609e-01],
           [-6.67991093e-01],
           [-6.75406644e-01],
           [-6.82754520e-01],
           [-6.90033984e-01],
           [-6.97244307e-01],
           [-7.04384767e-01],
           [-7.11454648e-01],
           [-7.18453241e-01],
           [-7.25379846e-01],
           [-7.32233768e-01],
           [-7.39014320e-01],
           [-7.45720824e-01],
           [-7.52352607e-01],
           [-7.58909005e-01],
           [-7.65389360e-01],
           [-7.71793024e-01],
           [-7.78119354e-01],
           [-7.84367718e-01],
           [-7.90537488e-01],
           [-7.96628046e-01],
           [-8.02638783e-01],
           [-8.08569096e-01],
           [-8.14418391e-01],
           [-8.20186082e-01],
           [-8.25871590e-01],
           [-8.31474347e-01],
           [-8.36993790e-01],
           [-8.42429367e-01],
           [-8.47780532e-01],
           [-8.53046751e-01],
           [-8.58227495e-01],
           [-8.63322245e-01],
           [-8.68330490e-01],
           [-8.73251730e-01],
           [-8.78085470e-01],
           [-8.82831226e-01],
           [-8.87488523e-01],
           [-8.92056894e-01],
           [-8.96535881e-01],
           [-9.00925037e-01],
           [-9.05223919e-01],
           [-9.09432099e-01],
           [-9.13549155e-01],
           [-9.17574673e-01],
           [-9.21508251e-01],
           [-9.25349494e-01],
           [-9.29098017e-01],
           [-9.32753446e-01],
           [-9.36315413e-01],
           [-9.39783561e-01],
           [-9.43157544e-01],
           [-9.46437023e-01],
           [-9.49621670e-01],
           [-9.52711165e-01],
           [-9.55705199e-01],
           [-9.58603471e-01],
           [-9.61405692e-01],
           [-9.64111581e-01],
           [-9.66720867e-01],
           [-9.69233287e-01],
           [-9.71648591e-01],
           [-9.73966536e-01],
           [-9.76186890e-01],
           [-9.78309431e-01],
           [-9.80333945e-01],
           [-9.82260231e-01],
           [-9.84088095e-01],
           [-9.85817354e-01],
           [-9.87447834e-01],
           [-9.88979373e-01],
           [-9.90411816e-01],
           [-9.91745021e-01],
           [-9.92978853e-01],
           [-9.94113189e-01],
           [-9.95147916e-01],
           [-9.96082929e-01],
           [-9.96918136e-01],
           [-9.97653452e-01],
           [-9.98288803e-01],
           [-9.98824127e-01],
           [-9.99259369e-01],
           [-9.99594485e-01],
           [-9.99829443e-01],
           [-9.99964218e-01],
           [-9.99998798e-01],
           [-9.99933178e-01],
           [-9.99767366e-01],
           [-9.99501377e-01],
           [-9.99135239e-01],
           [-9.98668988e-01],
           [-9.98102670e-01],
           [-9.97436344e-01],
           [-9.96670075e-01],
           [-9.95803940e-01],
           [-9.94838026e-01],
           [-9.93772430e-01],
           [-9.92607258e-01],
           [-9.91342628e-01],
           [-9.89978665e-01],
           [-9.88515508e-01],
           [-9.86953301e-01],
           [-9.85292203e-01],
           [-9.83532378e-01],
           [-9.81674005e-01],
           [-9.79717268e-01],
           [-9.77662364e-01],
           [-9.75509498e-01],
           [-9.73258887e-01],
           [-9.70910757e-01],
           [-9.68465341e-01],
           [-9.65922886e-01],
           [-9.63283645e-01],
           [-9.60547884e-01],
           [-9.57715877e-01],
           [-9.54787907e-01],
           [-9.51764268e-01],
           [-9.48645263e-01],
           [-9.45431204e-01],
           [-9.42122413e-01],
           [-9.38719222e-01],
           [-9.35221972e-01],
           [-9.31631013e-01],
           [-9.27946706e-01],
           [-9.24169418e-01],
           [-9.20299529e-01],
           [-9.16337427e-01],
           [-9.12283508e-01],
           [-9.08138179e-01],
           [-9.03901855e-01],
           [-8.99574960e-01],
           [-8.95157929e-01],
           [-8.90651203e-01],
           [-8.86055234e-01],
           [-8.81370484e-01],
           [-8.76597420e-01],
           [-8.71736521e-01],
           [-8.66788276e-01],
           [-8.61753178e-01],
           [-8.56631733e-01],
           [-8.51424454e-01],
           [-8.46131863e-01],
           [-8.40754490e-01],
           [-8.35292874e-01],
           [-8.29747562e-01],
           [-8.24119109e-01],
           [-8.18408081e-01],
           [-8.12615048e-01],
           [-8.06740592e-01],
           [-8.00785301e-01],
           [-7.94749771e-01],
           [-7.88634608e-01],
           [-7.82440424e-01],
           [-7.76167840e-01],
           [-7.69817485e-01],
           [-7.63389994e-01],
           [-7.56886012e-01],
           [-7.50306190e-01],
           [-7.43651188e-01],
           [-7.36921673e-01],
           [-7.30118318e-01],
           [-7.23241806e-01],
           [-7.16292826e-01],
           [-7.09272073e-01],
           [-7.02180252e-01],
           [-6.95018073e-01],
           [-6.87786253e-01],
           [-6.80485517e-01],
           [-6.73116597e-01],
           [-6.65680231e-01],
           [-6.58177165e-01],
           [-6.50608149e-01],
           [-6.42973943e-01],
           [-6.35275311e-01],
           [-6.27513025e-01],
           [-6.19687862e-01],
           [-6.11800607e-01],
           [-6.03852050e-01],
           [-5.95842988e-01],
           [-5.87774222e-01],
           [-5.79646561e-01],
           [-5.71460820e-01],
           [-5.63217820e-01],
           [-5.54918385e-01],
           [-5.46563347e-01],
           [-5.38153544e-01],
           [-5.29689819e-01],
           [-5.21173018e-01],
           [-5.12603997e-01],
           [-5.03983613e-01],
           [-4.95312730e-01],
           [-4.86592217e-01],
           [-4.77822948e-01],
           [-4.69005801e-01],
           [-4.60141660e-01],
           [-4.51231412e-01],
           [-4.42275952e-01],
           [-4.33276176e-01],
           [-4.24232986e-01],
           [-4.15147288e-01],
           [-4.06019993e-01],
           [-3.96852014e-01],
           [-3.87644272e-01],
           [-3.78397687e-01],
           [-3.69113187e-01],
           [-3.59791703e-01],
           [-3.50434167e-01],
           [-3.41041518e-01],
           [-3.31614697e-01],
           [-3.22154648e-01],
           [-3.12662319e-01],
           [-3.03138662e-01],
           [-2.93584631e-01],
           [-2.84001182e-01],
           [-2.74389277e-01],
           [-2.64749878e-01],
           [-2.55083951e-01],
           [-2.45392466e-01],
           [-2.35676391e-01],
           [-2.25936703e-01],
           [-2.16174375e-01],
           [-2.06390387e-01],
           [-1.96585719e-01],
           [-1.86761353e-01],
           [-1.76918273e-01],
           [-1.67057466e-01],
           [-1.57179921e-01],
           [-1.47286626e-01],
           [-1.37378573e-01],
           [-1.27456754e-01],
           [-1.17522165e-01],
           [-1.07575800e-01],
           [-9.76186559e-02],
           [-8.76517305e-02],
           [-7.76760224e-02],
           [-6.76925312e-02],
           [-5.77022572e-02],
           [-4.77062016e-02],
           [-3.77053657e-02],
           [-2.77007519e-02],
           [-1.76933624e-02],
           [-7.68420006e-03],
           [ 2.32573223e-03],
           [ 1.23354315e-02],
           [ 2.23438947e-02],
           [ 3.23501191e-02],
           [ 4.23531021e-02],
           [ 5.23518413e-02],
           [ 6.23453348e-02],
           [ 7.23325814e-02],
           [ 8.23125803e-02],
           [ 9.22843315e-02],
           [ 1.02246836e-01],
           [ 1.12199095e-01],
           [ 1.22140112e-01],
           [ 1.32068891e-01],
           [ 1.41984436e-01],
           [ 1.51885755e-01],
           [ 1.61771855e-01],
           [ 1.71641745e-01],
           [ 1.81494437e-01],
           [ 1.91328943e-01],
           [ 2.01144278e-01],
           [ 2.10939459e-01],
           [ 2.20713504e-01],
           [ 2.30465433e-01],
           [ 2.40194270e-01],
           [ 2.49899039e-01],
           [ 2.59578769e-01],
           [ 2.69232489e-01],
           [ 2.78859232e-01],
           [ 2.88458033e-01],
           [ 2.98027932e-01],
           [ 3.07567967e-01],
           [ 3.17077185e-01],
           [ 3.26554632e-01],
           [ 3.35999358e-01],
           [ 3.45410418e-01],
           [ 3.54786867e-01],
           [ 3.64127767e-01],
           [ 3.73432181e-01],
           [ 3.82699178e-01],
           [ 3.91927829e-01],
           [ 4.01117208e-01],
           [ 4.10266396e-01],
           [ 4.19374475e-01],
           [ 4.28440534e-01],
           [ 4.37463662e-01],
           [ 4.46442957e-01],
           [ 4.55377519e-01],
           [ 4.64266452e-01],
           [ 4.73108866e-01],
           [ 4.81903875e-01],
           [ 4.90650597e-01],
           [ 4.99348156e-01],
           [ 5.07995681e-01],
           [ 5.16592305e-01],
           [ 5.25137167e-01],
           [ 5.33629410e-01],
           [ 5.42068184e-01],
           [ 5.50452643e-01],
           [ 5.58781947e-01],
           [ 5.67055261e-01],
           [ 5.75271756e-01],
           [ 5.83430610e-01],
           [ 5.91531004e-01],
           [ 5.99572127e-01],
           [ 6.07553173e-01],
           [ 6.15473343e-01],
           [ 6.23331843e-01],
           [ 6.31127885e-01],
           [ 6.38860689e-01],
           [ 6.46529479e-01],
           [ 6.54133487e-01],
           [ 6.61671951e-01],
           [ 6.69144116e-01],
           [ 6.76549233e-01],
           [ 6.83886561e-01],
           [ 6.91155363e-01],
           [ 6.98354912e-01],
           [ 7.05484486e-01],
           [ 7.12543371e-01],
           [ 7.19530859e-01],
           [ 7.26446251e-01],
           [ 7.33288853e-01],
           [ 7.40057981e-01],
           [ 7.46752954e-01],
           [ 7.53373104e-01],
           [ 7.59917766e-01],
           [ 7.66386284e-01],
           [ 7.72778011e-01],
           [ 7.79092307e-01],
           [ 7.85328537e-01],
           [ 7.91486078e-01],
           [ 7.97564313e-01],
           [ 8.03562632e-01],
           [ 8.09480434e-01],
           [ 8.15317127e-01],
           [ 8.21072126e-01],
           [ 8.26744853e-01],
           [ 8.32334742e-01],
           [ 8.37841230e-01],
           [ 8.43263768e-01],
           [ 8.48601811e-01],
           [ 8.53854824e-01],
           [ 8.59022282e-01],
           [ 8.64103666e-01],
           [ 8.69098468e-01],
           [ 8.74006186e-01],
           [ 8.78826329e-01],
           [ 8.83558414e-01],
           [ 8.88201968e-01],
           [ 8.92756523e-01],
           [ 8.97221626e-01],
           [ 9.01596827e-01],
           [ 9.05881688e-01],
           [ 9.10075781e-01],
           [ 9.14178684e-01],
           [ 9.18189988e-01],
           [ 9.22109289e-01],
           [ 9.25936195e-01],
           [ 9.29670323e-01],
           [ 9.33311299e-01],
           [ 9.36858757e-01],
           [ 9.40312343e-01],
           [ 9.43671709e-01],
           [ 9.46936521e-01],
           [ 9.50106449e-01],
           [ 9.53181178e-01],
           [ 9.56160398e-01],
           [ 9.59043812e-01],
           [ 9.61831130e-01],
           [ 9.64522073e-01],
           [ 9.67116371e-01],
           [ 9.69613765e-01],
           [ 9.72014004e-01],
           [ 9.74316848e-01],
           [ 9.76522066e-01],
           [ 9.78629437e-01],
           [ 9.80638749e-01],
           [ 9.82549803e-01],
           [ 9.84362405e-01],
           [ 9.86076374e-01],
           [ 9.87691540e-01],
           [ 9.89207739e-01],
           [ 9.90624820e-01],
           [ 9.91942641e-01],
           [ 9.93161070e-01],
           [ 9.94279984e-01],
           [ 9.95299273e-01],
           [ 9.96218833e-01],
           [ 9.97038572e-01],
           [ 9.97758408e-01],
           [ 9.98378270e-01],
           [ 9.98898095e-01],
           [ 9.99317830e-01],
           [ 9.99637435e-01],
           [ 9.99856876e-01],
           [ 9.99976133e-01],
           [ 9.99995192e-01],
           [ 9.99914052e-01],
           [ 9.99732722e-01],
           [ 9.99451218e-01],
           [ 9.99069571e-01],
           [ 9.98587817e-01],
           [ 9.98006005e-01],
           [ 9.97324193e-01],
           [ 9.96542450e-01],
           [ 9.95660854e-01],
           [ 9.94679493e-01],
           [ 9.93598466e-01],
           [ 9.92417881e-01],
           [ 9.91137856e-01],
           [ 9.89758520e-01],
           [ 9.88280011e-01],
           [ 9.86702476e-01],
           [ 9.85026074e-01],
           [ 9.83250973e-01],
           [ 9.81377351e-01],
           [ 9.79405396e-01],
           [ 9.77335304e-01],
           [ 9.75167284e-01],
           [ 9.72901553e-01],
           [ 9.70538338e-01],
           [ 9.68077875e-01],
           [ 9.65520411e-01],
           [ 9.62866203e-01],
           [ 9.60115516e-01],
           [ 9.57268626e-01],
           [ 9.54325818e-01],
           [ 9.51287388e-01],
           [ 9.48153638e-01],
           [ 9.44924885e-01],
           [ 9.41601450e-01],
           [ 9.38183667e-01],
           [ 9.34671879e-01],
           [ 9.31066437e-01],
           [ 9.27367703e-01],
           [ 9.23576047e-01],
           [ 9.19691849e-01],
           [ 9.15715499e-01],
           [ 9.11647395e-01],
           [ 9.07487943e-01],
           [ 9.03237562e-01],
           [ 8.98896678e-01],
           [ 8.94465724e-01],
           [ 8.89945145e-01],
           [ 8.85335394e-01],
           [ 8.80636933e-01],
           [ 8.75850233e-01],
           [ 8.70975773e-01],
           [ 8.66014042e-01],
           [ 8.60965536e-01],
           [ 8.55830762e-01],
           [ 8.50610235e-01],
           [ 8.45304477e-01],
           [ 8.39914019e-01],
           [ 8.34439403e-01],
           [ 8.28881176e-01],
           [ 8.23239896e-01],
           [ 8.17516128e-01],
           [ 8.11710445e-01],
           [ 8.05823429e-01],
           [ 7.99855670e-01],
           [ 7.93807766e-01],
           [ 7.87680323e-01],
           [ 7.81473955e-01],
           [ 7.75189283e-01],
           [ 7.68826938e-01],
           [ 7.62387557e-01],
           [ 7.55871785e-01],
           [ 7.49280275e-01],
           [ 7.42613687e-01],
           [ 7.35872690e-01],
           [ 7.29057959e-01],
           [ 7.22170177e-01],
           [ 7.15210034e-01],
           [ 7.08178227e-01],
           [ 7.01075461e-01],
           [ 6.93902448e-01],
           [ 6.86659906e-01],
           [ 6.79348561e-01],
           [ 6.71969145e-01],
           [ 6.64522399e-01],
           [ 6.57009068e-01],
           [ 6.49429905e-01],
           [ 6.41785669e-01],
           [ 6.34077127e-01],
           [ 6.26305051e-01],
           [ 6.18470219e-01],
           [ 6.10573417e-01],
           [ 6.02615435e-01],
           [ 5.94597072e-01],
           [ 5.86519131e-01],
           [ 5.78382421e-01],
           [ 5.70187757e-01],
           [ 5.61935960e-01],
           [ 5.53627858e-01],
           [ 5.45264283e-01],
           [ 5.36846073e-01],
           [ 5.28374071e-01],
           [ 5.19849126e-01],
           [ 5.11272092e-01],
           [ 5.02643829e-01],
           [ 4.93965202e-01],
           [ 4.85237080e-01],
           [ 4.76460337e-01],
           [ 4.67635853e-01],
           [ 4.58764512e-01],
           [ 4.49847203e-01],
           [ 4.40884820e-01],
           [ 4.31878260e-01],
           [ 4.22828426e-01],
           [ 4.13736226e-01],
           [ 4.04602569e-01],
           [ 3.95428371e-01],
           [ 3.86214551e-01],
           [ 3.76962033e-01],
           [ 3.67671743e-01],
           [ 3.58344613e-01],
           [ 3.48981577e-01],
           [ 3.39583574e-01],
           [ 3.30151544e-01],
           [ 3.20686433e-01],
           [ 3.11189189e-01],
           [ 3.01660765e-01],
           [ 2.92102114e-01],
           [ 2.82514195e-01],
           [ 2.72897968e-01],
           [ 2.63254397e-01],
           [ 2.53584448e-01],
           [ 2.43889090e-01],
           [ 2.34169294e-01],
           [ 2.24426035e-01],
           [ 2.14660288e-01],
           [ 2.04873032e-01],
           [ 1.95065249e-01],
           [ 1.85237919e-01],
           [ 1.75392030e-01],
           [ 1.65528565e-01],
           [ 1.55648516e-01],
           [ 1.45752870e-01],
           [ 1.35842619e-01],
           [ 1.25918758e-01],
           [ 1.15982279e-01],
           [ 1.06034179e-01],
           [ 9.60754549e-02],
           [ 8.61071037e-02],
           [ 7.61301246e-02],
           [ 6.61455173e-02],
           [ 5.61542823e-02],
           [ 4.61574206e-02],
           [ 3.61559340e-02],
           [ 2.61508246e-02],
           [ 1.61430949e-02],
           [ 6.13374766e-03],
           [-3.87621418e-03],
           [-1.38857876e-02],
           [-2.38939697e-02],
           [-3.38997577e-02],
           [-4.39021489e-02],
           [-5.39001411e-02],
           [-6.38927325e-02],
           [-7.38789220e-02],
           [-8.38577088e-02],
           [-9.38280931e-02],
           [-1.03789076e-01],
           [-1.13739659e-01],
           [-1.23678846e-01],
           [-1.33605640e-01],
           [-1.43519046e-01],
           [-1.53418073e-01],
           [-1.63301726e-01],
           [-1.73169017e-01],
           [-1.83018957e-01],
           [-1.92850558e-01],
           [-2.02662836e-01],
           [-2.12454807e-01],
           [-2.22225490e-01],
           [-2.31973906e-01],
           [-2.41699079e-01],
           [-2.51400034e-01],
           [-2.61075798e-01],
           [-2.70725403e-01],
           [-2.80347881e-01],
           [-2.89942268e-01],
           [-2.99507604e-01],
           [-3.09042928e-01],
           [-3.18547287e-01],
           [-3.28019728e-01],
           [-3.37459301e-01],
           [-3.46865061e-01],
           [-3.56236066e-01],
           [-3.65571375e-01],
           [-3.74870055e-01],
           [-3.84131173e-01],
           [-3.93353801e-01],
           [-4.02537015e-01],
           [-4.11679896e-01],
           [-4.20781526e-01],
           [-4.29840994e-01],
           [-4.38857392e-01],
           [-4.47829817e-01],
           [-4.56757370e-01],
           [-4.65639155e-01],
           [-4.74474285e-01],
           [-4.83261871e-01],
           [-4.92001036e-01],
           [-5.00690902e-01],
           [-5.09330599e-01],
           [-5.17919262e-01],
           [-5.26456029e-01],
           [-5.34940046e-01],
           [-5.43370462e-01],
           [-5.51746432e-01],
           [-5.60067118e-01],
           [-5.68331685e-01],
           [-5.76539306e-01],
           [-5.84689158e-01],
           [-5.92780425e-01],
           [-6.00812295e-01],
           [-6.08783964e-01],
           [-6.16694633e-01],
           [-6.24543510e-01],
           [-6.32329808e-01],
           [-6.40052747e-01],
           [-6.47711552e-01],
           [-6.55305458e-01],
           [-6.62833702e-01],
           [-6.70295531e-01],
           [-6.77690196e-01],
           [-6.85016957e-01],
           [-6.92275080e-01],
           [-6.99463837e-01],
           [-7.06582509e-01],
           [-7.13630381e-01],
           [-7.20606748e-01],
           [-7.27510910e-01],
           [-7.34342176e-01],
           [-7.41099862e-01],
           [-7.47783289e-01],
           [-7.54391789e-01],
           [-7.60924700e-01],
           [-7.67381366e-01],
           [-7.73761141e-01],
           [-7.80063386e-01],
           [-7.86287469e-01],
           [-7.92432766e-01],
           [-7.98498661e-01],
           [-8.04484548e-01],
           [-8.10389826e-01],
           [-8.16213903e-01],
           [-8.21956196e-01],
           [-8.27616129e-01],
           [-8.33193136e-01],
           [-8.38686657e-01],
           [-8.44096142e-01],
           [-8.49421049e-01],
           [-8.54660845e-01],
           [-8.59815004e-01],
           [-8.64883010e-01],
           [-8.69864355e-01],
           [-8.74758541e-01],
           [-8.79565076e-01],
           [-8.84283479e-01],
           [-8.88913277e-01],
           [-8.93454007e-01],
           [-8.97905213e-01],
           [-9.02266449e-01],
           [-9.06537279e-01],
           [-9.10717274e-01],
           [-9.14806016e-01],
           [-9.18803095e-01],
           [-9.22708110e-01],
           [-9.26520671e-01],
           [-9.30240394e-01],
           [-9.33866908e-01],
           [-9.37399849e-01],
           [-9.40838863e-01],
           [-9.44183606e-01],
           [-9.47433742e-01],
           [-9.50588945e-01],
           [-9.53648900e-01],
           [-9.56613300e-01],
           [-9.59481847e-01],
           [-9.62254255e-01],
           [-9.64930246e-01],
           [-9.67509551e-01],
           [-9.69991913e-01],
           [-9.72377081e-01],
           [-9.74664818e-01],
           [-9.76854895e-01],
           [-9.78947090e-01],
           [-9.80941196e-01],
           [-9.82837012e-01],
           [-9.84634348e-01],
           [-9.86333025e-01],
           [-9.87932871e-01],
           [-9.89433727e-01],
           [-9.90835442e-01],
           [-9.92137877e-01],
           [-9.93340899e-01],
           [-9.94444389e-01],
           [-9.95448237e-01],
           [-9.96352341e-01],
           [-9.97156611e-01],
           [-9.97860966e-01],
           [-9.98465337e-01],
           [-9.98969661e-01],
           [-9.99373890e-01],
           [-9.99677982e-01],
           [-9.99881906e-01],
           [-9.99985643e-01],
           [-9.99989182e-01],
           [-9.99892522e-01],
           [-9.99695674e-01],
           [-9.99398657e-01],
           [-9.99001501e-01],
           [-9.98504245e-01],
           [-9.97906940e-01],
           [-9.97209644e-01],
           [-9.96412429e-01],
           [-9.95515375e-01],
           [-9.94518569e-01],
           [-9.93422114e-01],
           [-9.92226118e-01],
           [-9.90930702e-01],
           [-9.89535995e-01],
           [-9.88042137e-01],
           [-9.86449278e-01],
           [-9.84757577e-01],
           [-9.82967204e-01],
           [-9.81078338e-01],
           [-9.79091169e-01],
           [-9.77005895e-01],
           [-9.74822726e-01],
           [-9.72541880e-01],
           [-9.70163586e-01],
           [-9.67688082e-01],
           [-9.65115616e-01],
           [-9.62446446e-01],
           [-9.59680840e-01],
           [-9.56819074e-01],
           [-9.53861435e-01],
           [-9.50808220e-01],
           [-9.47659734e-01],
           [-9.44416293e-01],
           [-9.41078223e-01],
           [-9.37645857e-01],
           [-9.34119539e-01],
           [-9.30499623e-01],
           [-9.26786471e-01],
           [-9.22980456e-01],
           [-9.19081959e-01],
           [-9.15091370e-01],
           [-9.11009089e-01],
           [-9.06835526e-01],
           [-9.02571099e-01],
           [-8.98216234e-01],
           [-8.93771368e-01],
           [-8.89236948e-01],
           [-8.84613426e-01],
           [-8.79901266e-01],
           [-8.75100941e-01],
           [-8.70212930e-01],
           [-8.65237726e-01],
           [-8.60175824e-01],
           [-8.55027734e-01],
           [-8.49793970e-01],
           [-8.44475058e-01],
           [-8.39071529e-01]])

``` python
plt.imshow(I)
plt.colorbar()
```

    <matplotlib.colorbar.Colorbar at 0x7f5d4c0728b0>

![](08customizing%20colorbar_files/figure-gfm/cell-5-output-2.png)

## Customizing Colorbars

We can specify the colormap using the cmap argument to the plotting
function that is creating the visualization

``` python
plt.imshow(I, cmap='gray')
```

    <matplotlib.image.AxesImage at 0x7f5d3a3e2490>

![](08customizing%20colorbar_files/figure-gfm/cell-6-output-2.png)

All the available colormaps are in the plt.cm namespace; using IPython’s
tab- completion feature will give you a full list of built-in
possibilities:

``` python
#plt.imshow(I, cmap='winter_r')
```

But being able to choose a colormap is just the first step: more
important is how to decide among the possibilities! The choice turns out
to be much more subtle than you might initially expect.

## Choosing the colormap

A full treatment of color choice within visualization is beyond the
scope of this book, but for entertaining reading on this subject and
others,

Broadly, you should be aware of three different categories of
colormaps:  
*Sequential colormaps*  
These consist of one continuous sequence of colors (e.g., binary or
viridis).  
*Divergent colormaps* These usually contain two distinct colors, which
show positive and negative deviations from a mean (e.g., RdBu or
PuOr).  
*Qualitative colormaps*  
These mix colors with no particular sequence (e.g., rainbow or jet).

The jet colormap, which was the default in Matplotlib prior to version
2.0, is an example of a qualitative colormap. Its status as the default
was quite unfortunate, because qualitative maps are often a poor choice
for representing quantitative data. Among the problems is the fact that
qualitative maps usually do not display any uni‐ form progression in
brightness as the scale increases.

``` python
from matplotlib.colors import LinearSegmentedColormap
def grayscale_cmap(cmap):
    """Return a grayscale version of the given colormap"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    # convert RGBA to perceived grayscale luminance
    # cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]
    return LinearSegmentedColormap.from_list(cmap.name + "_gray", colors, cmap.N)
```

``` python
def view_colormap(cmap):
    """Plot a colormap with its grayscale equivalent"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    cmap = grayscale_cmap(cmap)
    grayscale = cmap(np.arange(cmap.N))
    fig, ax = plt.subplots(2, figsize=(6, 2),
    subplot_kw=dict(xticks=[], yticks=[]))
    ax[0].imshow([colors], extent=[0, 10, 0, 1])
    ax[1].imshow([grayscale], extent=[0, 10, 0, 1])
```

``` python
view_colormap('jet')
```

![](08customizing%20colorbar_files/figure-gfm/cell-10-output-1.png)

Notice the bright stripes in the grayscale image. Even in full color,
this uneven bright‐ ness means that the eye will be drawn to certain
portions of the color range, which will potentially emphasize
unimportant parts of the dataset. It’s better to use a color‐ map such
as viridis (the default as of Matplotlib 2.0), which is specifically
construc‐ ted to have an even brightness variation across the range.
Thus, it not only plays well with our color perception, but also will
translate well to grayscale printing

``` python
view_colormap('viridis')
```

![](08customizing%20colorbar_files/figure-gfm/cell-11-output-1.png)

``` python
view_colormap('cubehelix')
```

![](08customizing%20colorbar_files/figure-gfm/cell-12-output-1.png)

``` python
view_colormap('RdBu')
```

![](08customizing%20colorbar_files/figure-gfm/cell-13-output-1.png)

### Color limits and extensions

Matplotlib allows for a large range of colorbar customization. The
colorbar itself is simply an instance of plt.Axes, so all of the axes
and tick formatting tricks we’ve learned are applicable. The colorbar
has some interesting flexibility; for example, we can narrow the color
limits and indicate the out-of-bounds values with a triangular arrow at
the top and bottom by setting the extend property. This might come in
handy, for example, if you’re displaying an image that is subject to
noise

``` python
# make noise in 1% of the image pixel

speckles=(np.random.random(I.shape)<0.01)
I[speckles]=np.random.normal(0,3,np.count_nonzero(speckles))
```

``` python
plt.figure(figsize=(10,3.5))

plt.subplot(1,2,1)
plt.imshow(I, cmap='RdBu')
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(I, cmap='RdBu')
plt.colorbar(extend='both')
plt.clim(-1,1)
```

![](08customizing%20colorbar_files/figure-gfm/cell-15-output-1.png)

## Discrete colorbars

Colormaps are by default continuous, but sometimes you’d like to
represent discrete values. The easiest way to do this is to use the
plt.cm.get_cmap() function, and pass the name of a suitable colormap
along with the number of desired bins

``` python
plt.imshow(I, cmap=plt.cm.get_cmap('Blues',6))
plt.colorbar()
plt.clim(-1,1);
```

![](08customizing%20colorbar_files/figure-gfm/cell-16-output-1.png)
