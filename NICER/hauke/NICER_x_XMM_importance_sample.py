import STU.main_NxX as STU_NxX
import STU.main_NxX_IS as STU_NxX_IS

from xpsi.Sample import importance

names = ['mass', 'radius', 'distance', 'cos_inclination',
         'p__phase_shift', 'p__super_colatitude', 'p__super_radius', 'p__super_temperature',
         's__phase_shift', 's__super_colatitude', 's__super_radius', 's__super_temperature',
         'XTI__alpha',
         'column_density',
         'PN__alpha']

importance(STU_NxX_IS.likelihood,
                  STU_NxX.likelihood,
                  'STU/NICERxXMM/FI_H/run10/samples/nlive40000_eff0.1_noCONST_noMM_noIS_tol-1',
                  names = names,
                  likelihood_change = False,
                  prior_change = True,
                  weight_threshold=1.0e-30,
                  overwrite=True)
