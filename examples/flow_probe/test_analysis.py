"""Independent oracle tests for ranking, pair labels and declared budget accounting."""
import itertools
import unittest
from examples.flow_probe.analysis import classification, ranking_metrics, budget_metrics

class RankingTests(unittest.TestCase):
    def test_all_pair_labels(self):
        expected={(False,True):"rescue",(True,False):"regression",(True,True):"unchanged_success",(False,False):"unchanged_failure"}
        for outcomes,label in expected.items(): self.assertEqual(classification(*outcomes),label)

    def test_nonmonotonic_pattern_keeps_comparisons_separate(self):
        self.assertEqual([classification(False,x) for x in [True,False,True]],["rescue","unchanged_failure","rescue"])

    def test_constant_score_ties(self):
        r=ranking_metrics([1,1,1,1],["rescue","regression","unchanged_success","unchanged_failure"])
        self.assertEqual(r["ap"],.25);self.assertEqual(r["auroc"],.5)
        self.assertEqual(r["rescue_regression_auroc"],.5)

    def test_perfect_ranking(self):
        r=ranking_metrics([4,3,2,1],["rescue","rescue","regression","unchanged_success"])
        self.assertEqual(r["ap"],1);self.assertEqual(r["auroc"],1)
        self.assertEqual(r["rescue_regression_auroc"],1)

    def test_reverse_ranking_closed_form(self):
        r=ranking_metrics([1,2,3,4],["rescue","rescue","regression","unchanged_success"])
        self.assertAlmostEqual(r["ap"],(1/3+2/4)/2)
        self.assertEqual(r["auroc"],0)

    def test_grouped_tie_ap_not_order_sensitive(self):
        expected=(.5*.5)+(.5*2/3)
        for labels in [ ["rescue","regression","rescue"],["regression","rescue","rescue"] ]:
            r=ranking_metrics([2,2,1],labels)
            self.assertAlmostEqual(r["ap"],expected)
            self.assertEqual(r["auroc"],.25)

    def test_regressions_are_all_case_negatives(self):
        r=ranking_metrics([3,2,1],["regression","rescue","unchanged_failure"])
        self.assertEqual(r["ap"],.5);self.assertEqual(r["rescue_regression_auroc"],0)

    def test_pairwise_auc_independent_enumeration(self):
        scores=[4,2,2,3,0,1];labels=["rescue","rescue","regression","unchanged_success","unchanged_failure","rescue"]
        positives=[s for s,l in zip(scores,labels) if l=="rescue"]
        negatives=[s for s,l in zip(scores,labels) if l!="rescue"]
        expect=sum((p>n)+.5*(p==n) for p,n in itertools.product(positives,negatives))/(len(positives)*len(negatives))
        self.assertAlmostEqual(ranking_metrics(scores,labels)["auroc"],expect)

    def test_undefined_auc_without_classes(self):
        for labels in [["rescue","rescue"],["regression","unchanged_failure"]]:
            r=ranking_metrics([1,2],labels)
            self.assertIsNone(r["auroc"]);self.assertIsNone(r["rescue_regression_auroc"])



class BudgetTests(unittest.TestCase):
    @staticmethod
    def case(identity,score=1,cap=10,s1=False,sk=True,c1=3,ck=4):
        return dict(case_id=identity,score=score,sigma=score,severity=3,cap_chunks=cap,
                    success={"1":s1,"2":sk,"4":sk,"10":sk},
                    chunks={"1":c1,"2":ck,"4":ck,"10":ck},
                    cluster="task/"+identity,condition_id=identity)

    def test_probe_overhead_reserved_for_all(self):
        cases=[self.case(str(i),10-i) for i in range(10)]
        r=budget_metrics(cases,4,"score",.75,2)
        self.assertEqual(r["selected"],1)
        self.assertEqual(r["planned_extra_evaluations"],50)
        self.assertEqual(r["extra_budget"],75)
        self.assertEqual(r["actual_counterfactual_evaluations"],20+9*3+4*4)

    def test_prefix_does_not_skip_expensive_next(self):
        cases=[self.case("a",3,50),self.case("b",2,1),self.case("c",1,1)]
        r=budget_metrics(cases,4,"score",1,0)
        self.assertEqual(r["selected"],0)

    def test_ties_use_identity_not_outcome_or_input_order(self):
        a=self.case("a",1,s1=True,sk=False);b=self.case("b",1)
        for cases in [[a,b],[b,a]]:
            r=budget_metrics(cases,2,"score",.5,0)
            self.assertEqual(r["selected"],1);self.assertEqual(r["regression"],1)
            self.assertEqual(r["rescue"],0)

    def test_actual_lengths_do_not_change_selection(self):
        cases=[self.case(str(i),10-i) for i in range(10)]
        r=budget_metrics(cases,4,"score",.75,2)
        for c in cases:c["chunks"]={"1":50,"2":1,"4":1,"10":1}
        rr=budget_metrics(cases,4,"score",.75,2)
        for key in ["selected","rescue","regression","planned_extra_evaluations"]:
            self.assertEqual(r[key],rr[key])
        self.assertNotEqual(r["actual_counterfactual_evaluations"],rr["actual_counterfactual_evaluations"])

    def test_infeasible_overhead_explicit(self):
        r=budget_metrics([self.case("a",cap=1)],4,"score",1,2)
        self.assertFalse(r["overhead_feasible"]);self.assertEqual(r["selected"],0)
        self.assertEqual(r["planned_extra_evaluations"],2)

    def test_bootstrap_duplicate_occurrences_charged_separately(self):
        a=self.case("a",cap=10,c1=3,ck=4)
        r=budget_metrics([a,a],2,"score",.5,0)
        self.assertEqual(r["selected"],1)
        self.assertEqual(r["actual_counterfactual_evaluations"],4*2+3)

    def test_all_four_selected_labels_and_net(self):
        cases=[self.case(str(i),4-i,s1=a,sk=b) for i,(a,b) in enumerate([(False,True),(True,False),(True,True),(False,False)])]
        r=budget_metrics(cases,2,"score",1,0)
        self.assertEqual(r["selected"],4)
        for label in ["rescue","regression","unchanged_success","unchanged_failure"]:self.assertEqual(r[label],1)
        self.assertEqual(r["precision"],.25);self.assertEqual(r["recall"],1)
        self.assertEqual(r["net"],0)


class IndependentAuditTests(unittest.TestCase):
    def test_independent_point_oracles(self):
        from examples.flow_probe.independent_audit import independent_rank,independent_budget,compare
        scores=[2,2,1,3,0]
        labels=["rescue","regression","rescue","unchanged_failure","unchanged_success"]
        compare(ranking_metrics(scores,labels),independent_rank(scores,labels))
        cases=[BudgetTests.case(str(i),s,cap=10+i,s1=l in ["regression","unchanged_success"],sk=l in ["rescue","unchanged_success"]) for i,(s,l) in enumerate(zip(scores,labels))]
        for k in (2,4,10):
            for budget in (.25,.5,.75,1):
                compare(budget_metrics(cases,k,"score",budget,2),independent_budget(cases,k,"score",budget,2))

    def test_independent_bootstrap_reproduction(self):
        from examples.flow_probe.analysis import bootstrap
        from examples.flow_probe.independent_audit import independent_intervals,compare
        cases=[BudgetTests.case(str(i),i%3,cap=10+i,s1=i%3==0,sk=i%2==0) for i in range(12)]
        for i,c in enumerate(cases):c["cluster"]="family"+str(i//3)
        expected=independent_intervals(cases,"cluster",100,20261005)
        actual=bootstrap(cases,"score",2,"cluster",100,20261005)
        compare(actual,expected)

    def test_mismatch_comparison_fails_closed(self):
        from examples.flow_probe.independent_audit import compare
        with self.assertRaises(ValueError):compare({"precision":.4},{"precision":.5})


if __name__=="__main__":unittest.main()
