# Quasiprobabilistic state-overlap estimator for NISQ devices

Code to accompany the article **[Quasiprobabilistic state-overlap estimator for NISQ devices](www.arxiv.org/abs/XXXXX)**, by Leonardo Guerini, Roeland Wiersema, Juan Felipe Carrasquilla, and Leandro Aolita.

All code is written in Python. Here is a short summary of them:
* [scaling_ent](https://github.com/guerinileonardo/overlap/blob/main/scaling_ent.py): calculates the sample complexity scaling of the quasiprobabilistic method for pairs of pure entangled states;
* [scaling_prod](https://github.com/guerinileonardo/overlap/blob/main/scaling_prod.py): calculates the sample complexity scaling of the quasiprobabilistic method for pairs of pure product states;
* [IST_comparison](https://github.com/guerinileonardo/overlap/blob/main/IST_comparison.py): compares the performance of our method with the Improved Swap Test circuit, simulated with the LPDO paradigm within NISQ assumptions;
* [BB_comparison](https://github.com/guerinileonardo/overlap/blob/main/BB_comparison.py): compares the performance of our method with the Bell-basis circuit, simulated with the LPDO paradigm within NISQ assumptions;
* [direct_app_comparison](https://github.com/guerinileonardo/overlap/blob/main/direct_comparison.py): compares the performance of our method with the direct overlap estimator presented by Elben et. al.
