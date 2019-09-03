import unittest
import mcmodel
import mcmodel_util as util

class TestBasic(unittest.TestCase):
    def test_01_version_exists(self):
        # test if __version__ is defined
        self.assertTrue(isinstance(mcmodel.__version__, str))
    def test_02_major_version_is_3(self):        
        self.assertTrue(int(mcmodel.__version__.split('.',1)[0]) == 3)
    def test_03_function_prototype_set_seed(self):
        self.assertTrue(None is mcmodel.set_seed(0x12341234))    
    def test_04_function_parameters_define_phase_function(self):
        mcmodel._clear()
        self.assertTrue(None is mcmodel.define_phase_function([0,1],[0,1]))
        self.assertRaises(ValueError,
                          mcmodel.define_phase_function,
                          [0, 0.5],
                          [0, 1])
        self.assertRaises(ValueError,
                          mcmodel.define_phase_function,
                          [0.5, 1],
                          [0, 1])
    def test_05_function_parameters_simulate(self):
        mcmodel._clear()
        def test():
            # mcmodel.sumulate() tests for a phase function to be defined.
            out = {}
            mcmodel.define_phase_function([0,1],[0,1])
            return mcmodel.simulate(
                    {'angle': 0.1,
                     'azimuth': 0.1,
                     'k': 0.1,
                     'thickness': 0.1},
                    out)        
        self.assertTrue(None is test())
    def test_06_util_make_phase_function(self):
        def test():
            res = util.make_phase_function(
                'Henyey-Greenstein',
                parameters=(0.98, 0),
                angle_steps=4097)
            residual = (res['expected_g'] - res['measured_g']) + 4.8590098598e-6            
            return abs(residual) < 1e-17
    def test_07_function_get_last_particle_track(self):
        mcmodel._clear() # reset global state
        def test():
            mcmodel.set_seed(0x12341234)
            pf = util.make_phase_function(
                 'Henyey-Greenstein',
                 parameters=(0.98, 0),
                 angle_steps=4097)            
            mcmodel.define_phase_function(pf['lookup_cdf'],pf['lookup_phi'])
            out = {}
            mcmodel.simulate({'angle': -3.1415/2, 'azimuth': 0,
                     'k': 1, 'thickness': 1,
                     'do_record_track': True}, out)
            res = mcmodel.get_last_particle_track()
            ok_shape = res.shape == (5,3)
            if not ok_shape:
                print()
                print('Incorrect Shape: %s' % repr(res.shape))
            exp = [1.14870636e-2, 6.29354692e-3, -4.54845133e-1]
            diff = sum(abs(a-b) for a, b in zip(res[2], exp))            
            ok_content = diff < 2e-10
            if not ok_content:
                if ok_shape: print()
                print('Incorrect Diff: %f' % diff)
            return ok_shape and ok_content
            
        self.assertTrue(test())
        
if __name__ == '__main__':
    unittest.main()
    
