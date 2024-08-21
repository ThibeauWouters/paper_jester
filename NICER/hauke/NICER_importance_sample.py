import STU.main_NICER as STU_NICER
import STU.main_NICER_IS as STU_NICER_IS

from xpsi.Sample import importance_sample

names = ['mass', 'radius', 'distance', 'cos_inclination',
         'p__phase_shift', 'p__super_colatitude', 'p__super_radius', 'p__super_temperature',
         's__phase_shift', 's__super_colatitude', 's__super_radius', 's__super_temperature',
         'XTI__alpha',
         'column_density',]

importance_sample(STU_NICER_IS.likelihood,
                  STU_NICER.likelihood,
                  'STU/NICER/FI_H/run12/samples/nlive40000_eff0.1_noCONST_noMM_noIS_tol-1',
                  names = names,
                  likelihood_change = False,
                  prior_change = True,
                  weight_threshold=1.0e-30,
                  overwrite=True)
