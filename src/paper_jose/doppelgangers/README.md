# Doppelgangers

Code to investigate the behaviour of EOS doppelgangers.

The target is generated from the params (on 03/03/2025):
```python
{'E_sym': 33.431808, 'L_sym': 77.178344, 'K_sym': -129.761344, 'Q_sym': 422.442807, 'Z_sym': -1644.011429, 'E_sat': -16.0, 'K_sat': 285.527411, 'Q_sat': 652.3
66343, 'Z_sat': -1290.138303, 'nbreak': 0.32, 'n_CSE_0': 0.48, 'n_CSE_1': 0.64, 'n_CSE_2': 0.8, 'n_CSE_3': 0.96, 'n_CSE_4': 1.12, 'n_CSE_5': 1.28, 'n_CSE_6':
1.44, 'n_CSE_7': 1.6, 'cs2_CSE_0': 0.5, 'cs2_CSE_1': 0.7, 'cs2_CSE_2': 0.5, 'cs2_CSE_3': 0.4, 'cs2_CSE_4': 0.8, 'cs2_CSE_5': 0.6, 'cs2_CSE_6': 0.9, 'cs2_CSE_7
': 0.8, 'cs2_CSE_8': 0.9}
```
and the resulting output is saved to `my_target_microscopic.dat`, `my_target_macroscopic.dat` and used in the doppelgangers. 

## finetune

This is where we finetune some of the doppelganger EOS to explore for further features. That is, starting from an existing doppelganger EOS we try to enforce other features (MTOV etc). See doppelgangers.py for the source code. 