import unittest

import pandas as pd

from csi500_daily_alpha_pipeline import _apply_execution_constraints


class DailyTradeConstraintTest(unittest.TestCase):
    def test_unsellable_holding_is_carried_and_replacement_is_limited(self):
        frame = pd.DataFrame(
            {
                "stock_code": ["A", "B", "C", "D"],
                "score": [0.1, 0.2, 0.9, 0.8],
                "tradeable": [False, True, True, True],
                "sellable": [False, True, True, True],
                "execution_ret": [-0.10, 0.01, 0.03, 0.02],
                "limit_down_next_open": [True, False, False, False],
                "suspended_next_open": [False, False, False, False],
            }
        )

        selected, holdings, stats = _apply_execution_constraints(
            frame,
            prev_holdings={"A", "B"},
            top_n=2,
        )

        self.assertEqual(holdings, {"A", "C"})
        self.assertEqual(set(selected["stock_code"]), {"A", "C"})
        self.assertEqual(stats["n_locked"], 1)
        self.assertEqual(stats["n_limit_down_locked"], 1)
        self.assertEqual(stats["n_suspended_locked"], 0)
        self.assertAlmostEqual(stats["turnover"], 0.5)
        self.assertAlmostEqual(stats["gross_ret"], -0.035)

    def test_suspended_holding_is_carried(self):
        frame = pd.DataFrame(
            {
                "stock_code": ["A", "B"],
                "score": [0.0, 1.0],
                "tradeable": [False, True],
                "sellable": [False, True],
                "execution_ret": [0.0, 0.02],
                "limit_down_next_open": [False, False],
                "suspended_next_open": [True, False],
            }
        )

        _, holdings, stats = _apply_execution_constraints(
            frame,
            prev_holdings={"A"},
            top_n=1,
        )

        self.assertEqual(holdings, {"A"})
        self.assertEqual(stats["n_locked"], 1)
        self.assertEqual(stats["n_suspended_locked"], 1)
        self.assertEqual(stats["turnover"], 0.0)


if __name__ == "__main__":
    unittest.main()
