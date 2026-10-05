"""Independent CPU tests for the predeclared probe; no simulator or model calls."""
import hashlib
import math
import unittest
import numpy as np
from examples.flow_probe.probe import velocity_change_score, validate_trace, array_sha256

class ScoreTests(unittest.TestCase):
    def test_closed_form_constant(self):
        a=np.full((10,32),2,dtype=np.float32)
        b=np.full((10,32),3,dtype=np.float32)
        result=velocity_change_score(a,b)
        self.assertEqual(result["score"],.5)
        self.assertEqual(result["times"],[1,.5])
        self.assertEqual(result["dimension_slice"],[0,7])
        self.assertEqual(result["coordinate_system"],"normalized_model_actions")

    def test_independent_scalar_oracle(self):
        a=np.arange(320,dtype=np.float32).reshape(10,32)/80+.1
        b=a*.7+np.sin(a)
        pairs=[(float(a[i,j]),float(b[i,j])) for i in range(10) for j in range(7)]
        numerator=math.sqrt(sum((y-x)**2 for x,y in pairs)/70)
        denominator=max(math.sqrt(sum(x*x for x,y in pairs)/70),1e-6)
        self.assertAlmostEqual(velocity_change_score(a,b)["score"],numerator/denominator,places=6)

    def test_zero_and_floor(self):
        a=np.zeros((10,32),np.float32)
        self.assertEqual(velocity_change_score(a,a)["score"],0)
        b=a.copy(); b[:,:7]=1e-6
        self.assertAlmostEqual(velocity_change_score(a,b)["score"],1,places=5)
        self.assertAlmostEqual(velocity_change_score(a,b)["safeguarded_denominator"],1e-6,places=12)

    def test_finite_padding_excluded(self):
        a=np.ones((10,32),np.float32);b=2*a
        expected=velocity_change_score(a,b)
        a[:,7:]=1000;b[:,7:]=-500
        self.assertEqual(velocity_change_score(a,b),expected)

    def test_nonfinite_padding_still_fails_closed(self):
        a=np.ones((10,32),np.float32)
        for value in [np.nan,np.inf,-np.inf]:
            b=a.copy();b[3,20]=value
            with self.assertRaises(ValueError): velocity_change_score(a,b)

    def test_shape_must_be_exact(self):
        a=np.ones((10,32),np.float32)
        for shape in [(10,7),(1,10,32),(9,32),(11,32),(32,10)]:
            with self.assertRaises(ValueError): velocity_change_score(a,np.ones(shape))

    def test_scale_invariance_away_from_floor(self):
        a=np.arange(320,dtype=np.float32).reshape(10,32)/100+1
        b=a+2
        self.assertAlmostEqual(velocity_change_score(a,b)["score"],velocity_change_score(2*a,2*b)["score"],places=6)

    def test_no_input_or_rng_mutation(self):
        a=np.ones((10,32),np.float32);b=2*a
        aa=a.copy();bb=b.copy();before=np.random.get_state()
        first=velocity_change_score(a,b); second=velocity_change_score(a,b)
        after=np.random.get_state()
        np.testing.assert_array_equal(a,aa);np.testing.assert_array_equal(b,bb)
        np.testing.assert_array_equal(before[1],after[1])
        self.assertEqual(before[2:],after[2:]);self.assertEqual(first,second)

    def test_overflow_rejected(self):
        a=np.full((10,32),1e30,dtype=np.float32)
        with self.assertRaises((ValueError,FloatingPointError)): velocity_change_score(a,2*a)

class TraceTests(unittest.TestCase):
    def make_trace(self):
        z=np.arange(320,dtype=np.float32).reshape(1,10,32)/64
        v=np.ones_like(z)*2
        middle=z-.5*v
        second=np.ones_like(z)*3
        return np.stack([z,middle,middle-.5*second]),np.stack([v,second]),np.array([1,.5],np.float32)

    def test_declared_midpoint(self):
        states,vel,times=self.make_trace()
        self.assertEqual(validate_trace(states,vel,times),0)

    def test_wrong_midpoint_and_time_rejected(self):
        states,vel,times=self.make_trace()
        bad=states.copy();bad[1,0,0,0]+=.1
        with self.assertRaises(ValueError): validate_trace(bad,vel,times)
        for times in [np.array([1,0]),np.array([.5,1]),np.array([[1,.5]])]:
            with self.assertRaises(ValueError): validate_trace(states,vel,times)

    def test_nonfinite_and_shape_rejected(self):
        states,vel,times=self.make_trace()
        with self.assertRaises(ValueError): validate_trace(states[:,0],vel,times)
        states[2,0,0,0]=np.nan
        with self.assertRaises(ValueError): validate_trace(states,vel,times)

    def test_array_hash_is_recorded_bytes(self):
        a=np.arange(320,dtype=np.float32).reshape(10,32)
        self.assertEqual(array_sha256(a),hashlib.sha256(a.tobytes()).hexdigest())
        self.assertEqual(array_sha256(a[:,::2]),hashlib.sha256(np.ascontiguousarray(a[:,::2]).tobytes()).hexdigest())

if __name__=="__main__": unittest.main()

