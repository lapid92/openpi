"""Focused independent tests for retrospective fixed-arm evidence."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("rescue_analysis", Path(__file__).with_name("analysis.py"))
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def row(step=1, success=False, case="a", study="old", phase="main"):
    return dict(study=study, phase=phase, physical_case_id=case, benchmark="libero",
                suite="suite", family=case, category="standard", severity=0, condition_id=case,
                seed=1, init_index=0, flow_steps=step, success=success, status="ok",
                checkpoint_sha256="checkpoint", benchmark_commit="benchmark",
                initial_state_sha256="initial", stabilized_state_sha256="stable",
                noise_policy_sha256="noise_policy", execution_sha256="execution",
                source_refs=[dict(path="/raw.jsonl", line=step)],
                chunks=[dict(chunk_index=0, noise_sha256="noise", observation_sha256="observation",
                             action_sha256="action", velocity_evaluations=step)])


def case(pattern, name="a"):
    return analysis.pair_records([row(step, success, name) for step, success in zip(analysis.STEPS, pattern)])[0]


class ClassificationTests(unittest.TestCase):
    def test_all_sixteen_patterns(self):
        import itertools
        for vector in itertools.product((False, True), repeat=4):
            with self.subTest(vector=vector):
                result = analysis.classify(dict(zip(analysis.STEPS, vector)))
                successes = [step for step, success in zip(analysis.STEPS, vector) if success]
                self.assertEqual(result["first_success"], str(successes[0]) if successes else "never")
                self.assertEqual(result["success_vector"], list(vector))
                self.assertEqual(result["nonmonotonic"], any(vector[i] and not vector[j] for i in range(4) for j in range(i+1,4)))

    def test_nonmonotonic_rescue(self):
        result = analysis.classify({1: False, 2: True, 4: False, 10: True})
        self.assertEqual(result["first_success"], "2")
        self.assertFalse(result["persists_at_larger_tested"])
        self.assertEqual(result["rescue_steps"], [2,10])

    def test_partial_first_success_is_unknown(self):
        for ten in (False, True):
            result = analysis.classify({1: False, 10: ten})
            self.assertEqual(result["first_success"], "unknown")
            self.assertEqual(result["success_vector"], [False,None,None,ten])
            self.assertFalse(result["complete_four_arm"])
        self.assertEqual(analysis.classify({1:False,10:True})["first_tested_success"], "10")

    def test_invalid_success_rejected(self):
        for bad in ({1:0}, {3:True}, {1:"false"}):
            with self.assertRaises(ValueError):
                analysis.classify(bad)


class PairingTests(unittest.TestCase):
    def test_exact_duplicate_does_not_inflate_count(self):
        a = row()
        result = analysis.pair_records([a, copy.deepcopy(a), row(10,True)])
        self.assertEqual(len(result),1)
        self.assertEqual(len(result[0]["arms"]["1"]["source_refs"]),2)
        self.assertEqual(analysis.counts(result)["cases"],1)

    def test_conflicting_duplicate_rejected(self):
        for field,value in (("success",True),("checkpoint_sha256","other"),("benchmark_commit","other"),
                            ("noise_policy_sha256","other"),("execution_sha256","other")):
            a,b = row(),row()
            b[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                analysis.pair_records([a,b])

    def test_pair_identity_mismatch_rejected(self):
        for field in ("checkpoint_sha256","benchmark_commit","initial_state_sha256",
                      "stabilized_state_sha256","noise_policy_sha256","execution_sha256"):
            a,b = row(),row(10,True)
            b[field]="wrong"
            with self.subTest(field=field),self.assertRaises(ValueError):
                analysis.pair_records([a,b])

    def test_noise_observation_accounting_mismatch_rejected(self):
        for field,value in (("noise_sha256","wrong"),("observation_sha256","wrong"),("velocity_evaluations",2)):
            a,b = row(),row(10,True)
            b["chunks"][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                analysis.pair_records([a,b])

    def test_errors_never_scored_as_failures(self):
        a=row()
        a["status"]="error"
        with self.assertRaises(ValueError):
            analysis.pair_records([a])

    def test_main_smoke_overlap_deduplicated(self):
        result=analysis.pair_records([row(),row(10,True),row(phase="smoke"),row(10,True,phase="smoke")])
        self.assertEqual(sum(c["include_unique_population"] for c in result),1)
        self.assertTrue(next(c for c in result if c["phase"]=="main")["include_unique_population"])
        summary=analysis.summarize(result,replicates=10)
        self.assertEqual(list(summary),["old/main/libero"])


class RateTests(unittest.TestCase):
    def test_partial_arms_have_eligible_denominators(self):
        cases=[case((False,True,False,True)),case((True,False,True,False),"b")]
        cases+=analysis.pair_records([row(case="c"),row(10,True,case="c")])
        result=analysis.counts(cases)
        self.assertEqual(result["steps"]["2"]["paired_cases"],2)
        self.assertEqual(result["steps"]["2"]["rescue_rate_among_failures"],1)
        self.assertEqual(result["steps"]["10"]["paired_cases"],3)
        self.assertEqual(result["steps"]["10"]["rescues"],2)
        self.assertEqual(result["steps"]["10"]["regressions"],1)
        self.assertEqual(result["first_success_among_one_step_failures"],{"2":1,"unknown":1})

    def test_cluster_intervals_reproducible(self):
        cases=[case((False,True,True,True),"a"),case((True,True,True,True),"b")]
        a=analysis.cluster_intervals(cases,replicates=100,seed=1)
        self.assertEqual(a,analysis.cluster_intervals(cases,replicates=100,seed=1))
        self.assertEqual(a["clusters"],2)
        self.assertEqual(a["intervals"]["first_2_among_all"]["valid_replicates"],100)
        self.assertEqual(a["intervals"]["first_2_among_all"]["lower"],0)
        self.assertLess(a["intervals"]["first_2_among_failures"]["valid_replicates"],100)

    def test_single_cluster_has_no_spurious_interval(self):
        result=analysis.cluster_intervals([case((False,True,True,True))])
        self.assertIsNone(result["intervals"])

if __name__=="__main__":
    unittest.main()

