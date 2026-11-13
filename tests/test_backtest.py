"""Tests for the rolling replenishment backtest.

Each test builds a tiny synthetic panel directly, so the semantics under test
are explicit rather than implied by the bundled demo data.
"""

import copy
import csv
import json
import math
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from analysis_app.backtest import (
    CENSORED, EXACT, MISSING, UNIDENTIFIABLE, run_backtest,
)
from analysis_app.config import (
    config_fingerprint, load_config, normalize_config, resolve_cost,
)
from analysis_app.demand_models import (
    exposure_profile, fit_exposure_hurdle, pmf_mean, pmf_quantile,
    predict_window_pmf,
)
from analysis_app.io import load_records

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / 'config' / 'replenishment.json'


def config(**overrides):
    base = {
        'currency': 'CNY',
        'costs': {'shortage_cost_per_unit': 60.0,
                  'holding_cost_per_unit_per_day': 1.5,
                  'order_cost_per_order': 0.0},
        'settings': {'min_history_days': 3, 'recent_window_days': 7,
                     'availability_window_days': 7, 'croston_alpha': 0.1,
                     'min_selection_windows': 2, 'default_model': 'exposure_hurdle',
                     'exclude_promotion_from_training': True,
                     'exclude_promotion_from_evaluation': True,
                     'max_stockout_rate': 0.25, 'stockout_rate_tolerance': 0.05,
                     'min_order_quantile': 0.5, 'max_order_quantile': 0.95,
                     'candidate_quantiles': [0.5, 0.55, 0.6, 0.65, 0.7,
                                             0.75, 0.8, 0.85, 0.9, 0.95],
                     'demand_sensitivity_factors': [0.0, 0.5, 1.0],
                     'report_quantiles': [0.5, 0.8, 0.9, 0.95]},
    }
    for key, value in overrides.items():
        if key in ('costs', 'settings'):
            base[key].update(value)
        else:
            base[key] = value
    return normalize_config(base)


def make_rows(sku, family, days, lead=2, price=10.0, sold=None, hours=None,
              promotion=None):
    """Build a full daily panel for one SKU.

    ``days`` is an iterable of day numbers.  ``sold`` / ``hours`` /
    ``promotion`` are either scalars or dicts keyed by day.
    """
    rows = []
    for day in days:
        rows.append({
            'observation_id': f'{sku}-{day:03d}',
            'sku': sku, 'family': family, 'day': day,
            'sold': float(sold[day] if isinstance(sold, dict) else (sold or 0.0)),
            'available_hours': float(hours[day] if isinstance(hours, dict) else
                                     (24.0 if hours is None else hours)),
            'lead_days': lead, 'price': price,
            'promotion': int(promotion[day] if isinstance(promotion, dict)
                             else (promotion or 0)),
        })
    return rows


def forecast(result, sku, origin):
    matches = [item for item in result['forecasts']
               if item['sku'] == sku and item['origin'] == origin]
    assert matches, f'no forecast for {sku}@{origin}'
    return matches[0]


class ExposureSemanticsTest(unittest.TestCase):
    def test_closed_day_is_not_a_zero_demand_day(self):
        profile = exposure_profile([
            {'available_hours': 0.0}, {'available_hours': 24.0},
            {'available_hours': 12.0}])
        self.assertEqual(profile['closed_days'], 1)
        self.assertEqual(profile['partial_days'], 1)
        self.assertEqual(profile['exposure_units'], 1.5)
        # Closed days stay in the calendar denominator.
        self.assertAlmostEqual(profile['exposure_per_calendar_day'], 0.5)

    def test_partial_day_counts_fractional_exposure(self):
        profile = exposure_profile([{'available_hours': 6.0}])
        self.assertAlmostEqual(profile['exposure_per_calendar_day'], 0.25)
        self.assertEqual(profile['full_days'], 0)

    def test_zero_event_history_is_not_confident_zero(self):
        """A run of zeros with real exposure yields an upper bound, not zero."""
        train = [{'sold': 0.0, 'available_hours': 24.0, 'promotion': 0}
                 for _ in range(4)]
        fit = fit_exposure_hurdle(train, {'croston_alpha': 0.1})
        self.assertTrue(fit['zero_events'])
        self.assertTrue(fit['occurrence_is_upper_bound'])
        # 95% rule-of-three bound on 4 exposure units is 3/4.
        self.assertAlmostEqual(fit['occurrence_upper_bound_per_exposure_unit'], 0.75)
        self.assertGreater(fit['occurrence_per_exposure_unit'], 0.0)

    def test_zero_event_values_are_uncertain_quantity(self):
        """The quantile of a zero-event forecast is not certainly zero."""
        train = [{'sold': 0.0, 'available_hours': 24.0, 'promotion': 0}
                 for _ in range(4)]
        fit = fit_exposure_hurdle(train, {'croston_alpha': 0.1})
        exposure = exposure_profile(train)
        pmf = predict_window_pmf('exposure_hurdle', fit, exposure, 2)
        self.assertGreater(pmf_mean(pmf), 0.0)
        quantity, truncated = pmf_quantile(pmf, 0.95)
        self.assertFalse(truncated)
        self.assertGreater(quantity, 0)


class ClassificationTest(unittest.TestCase):
    def build(self, hours, sold, lead=2):
        rows = make_rows('A', 'F', range(7), lead=lead, sold=sold, hours=hours)
        return run_backtest(rows, config(settings={'min_selection_windows': 999}))

    def test_fully_open_window_is_exact(self):
        result = self.build({day: 24.0 for day in range(7)},
                            {day: 0.0 for day in range(7)})
        self.assertEqual(forecast(result, 'A', 2)['evaluation']['classification'], EXACT)
        self.assertTrue(forecast(result, 'A', 2)['evaluation']['demand_identifiable'])

    def test_partial_window_is_censored_with_lower_bound(self):
        hours = {day: 24.0 for day in range(7)}
        hours[4] = 6.0
        result = self.build(hours, {day: 2.0 for day in range(7)})
        evaluation = forecast(result, 'A', 2)['evaluation']
        self.assertEqual(evaluation['classification'], CENSORED)
        self.assertTrue(evaluation['observed_units_are_lower_bound'])
        self.assertFalse(evaluation['demand_identifiable'])
        self.assertEqual(evaluation['bounds']['demand_lower_bound'], 4.0)
        self.assertAlmostEqual(evaluation['bounds']['lost_exposure_hours'], 18.0)
        self.assertFalse(evaluation['used_for_selection'])

    def test_closed_window_is_unidentifiable_and_never_scored_as_demand(self):
        hours = {day: 24.0 for day in range(7)}
        hours[4] = 0.0
        result = self.build(hours, {day: 2.0 for day in range(7)})
        evaluation = forecast(result, 'A', 2)['evaluation']
        self.assertEqual(evaluation['classification'], UNIDENTIFIABLE)
        self.assertFalse(evaluation['demand_identifiable'])
        self.assertFalse(evaluation['used_for_selection'])

    def test_missing_rows_are_unidentifiable(self):
        rows = [row for row in make_rows('A', 'F', range(7), sold=1.0)
                if row['day'] != 4]
        result = run_backtest(rows, config(settings={'min_selection_windows': 999}))
        evaluation = forecast(result, 'A', 2)['evaluation']
        self.assertEqual(evaluation['classification'], UNIDENTIFIABLE)
        self.assertEqual(evaluation['missing_days'], [4])

    def test_lead_window_is_sku_specific_not_next_day(self):
        rows = (make_rows('A', 'F', range(10), lead=1, sold=1.0)
                + make_rows('B', 'F', range(10), lead=4, sold=1.0))
        result = run_backtest(rows, config(settings={'min_selection_windows': 999}))
        self.assertEqual(forecast(result, 'A', 3)['evaluation']['window_days'], [4])
        self.assertEqual(forecast(result, 'B', 3)['evaluation']['window_days'], [4, 5, 6, 7])
        # The longer lead time must produce a larger expected window demand.
        self.assertGreater(forecast(result, 'B', 3)['models']['exposure_hurdle']['mean_units'],
                           forecast(result, 'A', 3)['models']['exposure_hurdle']['mean_units'])


class PointInTimeTest(unittest.TestCase):
    def build_panel(self):
        rows = []
        for sku, family, lead in (('A', 'F', 2), ('B', 'F', 3)):
            rows += make_rows(sku, family, range(16), lead=lead,
                              sold={day: float(day % 3) for day in range(16)})
        return rows

    def test_future_values_cannot_change_past_origins(self):
        original = self.build_panel()
        changed = copy.deepcopy(original)
        for row in changed:
            if row['day'] >= 8:
                row['sold'] = 999.0
                row['available_hours'] = 0.0
        before = run_backtest(original, config())
        after = run_backtest(changed, config())
        for sku in ('A', 'B'):
            for origin in range(3, 8):
                self.assertEqual(forecast(before, sku, origin)['models'],
                                 forecast(after, sku, origin)['models'])
                self.assertEqual(forecast(before, sku, origin)['recommendation'],
                                 forecast(after, sku, origin)['recommendation'])

    def test_shuffling_future_rows_does_not_change_past_origins(self):
        original = self.build_panel()
        shuffled = copy.deepcopy([row for row in original if row['day'] < 8])
        future = [row for row in original if row['day'] >= 8]
        random.Random(7).shuffle(future)
        before = run_backtest(original, config())
        after = run_backtest(shuffled + future, config())
        for sku in ('A', 'B'):
            for origin in range(3, 8):
                self.assertEqual(forecast(before, sku, origin)['models'],
                                 forecast(after, sku, origin)['models'])

    def test_input_order_does_not_change_result(self):
        original = self.build_panel()
        shuffled = copy.deepcopy(original)
        random.Random(11).shuffle(shuffled)
        self.assertEqual(run_backtest(original, config()), run_backtest(shuffled, config()))

    def test_family_decision_uses_only_resolved_windows(self):
        result = run_backtest(self.build_panel(), config())
        used = [item for item in result['forecasts']
                if item['evaluation']['used_for_selection']]
        self.assertTrue(used)
        for decision in result['decision_timeline']:
            resolved = [item for item in used
                        if item['family'] == decision['family']
                        and item['evaluation']['window_days'][-1] <= decision['decision_origin']]
            # The evidence count is bounded by what was already resolved when
            # the decision was taken; a window straddling the decision origin
            # can never contribute.
            self.assertLessEqual(decision['evidence_windows'], len(resolved))

    def test_origins_are_not_invented_past_the_calendar(self):
        rows = make_rows('A', 'F', range(8), lead=3, sold=1.0)
        result = run_backtest(rows, config())
        origins = [item['origin'] for item in result['forecasts']]
        self.assertEqual(origins, [2, 3, 4])
        self.assertEqual(result['summary']['classification_counts'][UNIDENTIFIABLE], 0)


class FamilySelectionTest(unittest.TestCase):
    def test_expensive_sku_does_not_mask_cheap_stockout_rate(self):
        """A pricey SKU must not dominate selection through money volume."""
        settings = {'min_selection_windows': 2}
        costs = {'shortage_cost_per_unit': {'BIG': 5000.0, '*': 10.0}}
        rows = []
        for sku, family, lead, price, pattern in (
                ('BIG', 'F', 2, 900.0, [1.0, 9.0, 1.0, 9.0, 1.0, 9.0, 1.0, 9.0, 1.0, 9.0]),
                ('SMALL', 'F', 2, 1.0, [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])):
            rows += make_rows(sku, family, range(10), lead=lead, price=price,
                              sold={day: pattern[day] for day in range(10)})
        result = run_backtest(rows, config(costs=costs, settings=settings))
        decision = result['family_decisions'][0]
        self.assertIn('stockout_rate_sku_equal', decision)
        self.assertNotIn('stockout_rate', decision.get('guard', {}))
        # The guard is measured SKU-equal, so both SKUs count once regardless
        # of their unit prices.
        self.assertIn('SMALL', decision['cost_share_by_sku'])
        self.assertIn('BIG', decision['cost_share_by_sku'])

    def test_insufficient_evidence_falls_back_with_explicit_reason(self):
        rows = make_rows('A', 'F', range(6), lead=2, sold=1.0)
        result = run_backtest(rows, config(settings={'min_selection_windows': 99}))
        decision = result['family_decisions'][0]
        self.assertEqual(decision['reason_code'], 'insufficient_identifiable_evidence')
        self.assertIsNone(decision['expected_cost_per_sku_origin'])
        self.assertIn('reason_text', decision)

    def test_selection_reports_alternatives_and_ties(self):
        rows = []
        for sku, family in (('A', 'F'), ('B', 'F')):
            rows += make_rows(sku, family, range(12), lead=2,
                              sold={day: float(day % 2) for day in range(12)})
        result = run_backtest(rows, config())
        decision = result['family_decisions'][0]
        self.assertTrue(decision['alternatives'])
        self.assertIn('tied_candidates', decision['tie'])
        self.assertIn(decision['selected_model'], ('croston_sba', 'exposure_hurdle',
                                                   'exposure_poisson'))
        self.assertGreaterEqual(decision['selected_quantile'], 0.5)
        self.assertLessEqual(decision['selected_quantile'], 0.95)

    def test_stockout_guard_rejects_cheaper_high_stockout_candidate(self):
        """Cheaper on paper but misses too many windows must lose to the guard."""
        rows = make_rows('CHEAP', 'F', range(14), lead=2, price=1.0,
                         sold={day: float(9 if day % 3 == 2 else 0) for day in range(14)})
        rows += make_rows('PRICEY', 'F', range(14), lead=2, price=1000.0, sold=0.0)
        result = run_backtest(rows, config(
            costs={'shortage_cost_per_unit': {'PRICEY': 100000.0, '*': 1.0}}))
        decision = result['family_decisions'][0]
        self.assertLessEqual(decision['stockout_rate_sku_equal'],
                             decision['guard']['max_stockout_rate']
                             + decision['guard']['tolerance'])
        for alternative in decision['alternatives']:
            if not alternative['feasible']:
                self.assertGreater(
                    alternative['stockout_rate_sku_equal'],
                    decision['guard']['max_stockout_rate']
                    + decision['guard']['tolerance'])

    def test_costs_are_echoed_and_fingerprinted(self):
        rows = make_rows('A', 'F', range(8), lead=2, sold=1.0)
        result = run_backtest(rows, config())
        self.assertEqual(result['config']['costs']['shortage_cost_per_unit'], 60.0)
        self.assertEqual(result['config_fingerprint'], config_fingerprint(result['config']))
        self.assertTrue(result['config_fingerprint'])

    def test_resolve_cost_priority(self):
        setting = {'*': 1.0, 'F': 2.0, 'A': 3.0}
        self.assertEqual(resolve_cost(setting, 'A', 'F'), 3.0)
        self.assertEqual(resolve_cost(setting, 'B', 'F'), 2.0)
        self.assertEqual(resolve_cost(setting, 'B', 'G'), 1.0)
        with self.assertRaises(ValueError):
            resolve_cost({'F': 1.0}, 'B', 'G')


class EdgeCaseTest(unittest.TestCase):
    def test_long_run_of_no_exposure_is_unidentifiable_and_flagged(self):
        hours = {day: 24.0 for day in range(12)}
        for day in range(2, 9):
            hours[day] = 0.0
        rows = make_rows('A', 'F', range(12), lead=2,
                         sold={day: 0.0 for day in range(12)}, hours=hours)
        result = run_backtest(rows, config(settings={'min_selection_windows': 999}))
        evaluation = forecast(result, 'A', 3)['evaluation']
        self.assertEqual(evaluation['classification'], UNIDENTIFIABLE)
        self.assertFalse(evaluation['used_for_selection'])
        self.assertGreater(evaluation['bounds']['lost_exposure_hours'], 0.0)
        self.assertTrue(any(item['reason'] == 'closed_days_in_window'
                            for item in result['unidentifiable']))

    def test_true_zero_demand_keeps_exposure_evidence(self):
        """Zero sales on fully open days is evidence, unlike closed days."""
        rows = make_rows('A', 'F', range(10), lead=2, sold=0.0,
                         hours={day: 24.0 for day in range(10)})
        result = run_backtest(rows, config(settings={'min_selection_windows': 999}))
        item = forecast(result, 'A', 4)
        self.assertTrue(item['exposure_evidence'])
        self.assertEqual(item['demand_status'], 'zero_events_exposed')
        self.assertEqual(item['evaluation']['classification'], EXACT)
        self.assertGreater(item['zero_event_upper_bound'], 0.0)
        # The point forecast remains cautious but is not a hard zero.
        self.assertGreater(item['models']['exposure_hurdle']['mean_units'], 0.0)

    def test_new_sku_is_labelled_and_has_no_borrowed_history(self):
        rows = make_rows('NEW', 'F', range(8), lead=2, sold=3.0)
        result = run_backtest(rows, config(settings={'min_selection_windows': 999}))
        first = result['forecasts'][0]
        self.assertTrue(first['is_new_sku'])
        self.assertEqual(first['known_days'], 3)
        self.assertEqual(first['training_rows'], 3)

    def test_post_promotion_window_is_excluded(self):
        promotion = {day: 0 for day in range(12)}
        for day in (5, 6, 7):
            promotion[day] = 1
        rows = make_rows('A', 'F', range(12), lead=2, sold=2.0,
                         promotion=promotion)
        result = run_backtest(rows, config(settings={'min_selection_windows': 999}))
        item = forecast(result, 'A', 5)
        self.assertTrue(item['evaluation']['promotion_in_window'])
        self.assertFalse(item['evaluation']['used_for_selection'])
        self.assertEqual(item['evaluation']['selection_exclusion_reason'], 'promotion_window')
        # Promotion days never enter training either.
        self.assertGreater(item['promotion_rows_excluded_from_training'], 0)

    def test_single_sku_single_day_dataset_has_no_origins(self):
        rows = make_rows('A', 'F', [0], lead=1, sold=1.0)
        result = run_backtest(rows, config())
        self.assertEqual(result['forecasts'], [])

    def test_duplicate_day_rejected(self):
        rows = make_rows('A', 'F', range(5), lead=2, sold=1.0)
        rows.append(dict(rows[0], observation_id='dup'))
        with self.assertRaisesRegex(ValueError, '重复观测'):
            run_backtest(rows, config())

    def test_inconsistent_lead_time_rejected(self):
        rows = make_rows('A', 'F', range(5), lead=2, sold=1.0)
        rows[3]['lead_days'] = 9
        with self.assertRaisesRegex(ValueError, '不一致'):
            run_backtest(rows, config())

    def test_negative_sales_rejected(self):
        rows = make_rows('A', 'F', range(5), lead=1, sold=1.0)
        rows[2]['sold'] = -1.0
        with self.assertRaisesRegex(ValueError, '不能为负'):
            run_backtest(rows, config())

    def test_out_of_range_availability_rejected(self):
        rows = make_rows('A', 'F', range(5), lead=1, sold=1.0)
        rows[2]['available_hours'] = 25.0
        with self.assertRaisesRegex(ValueError, '0 到 24'):
            run_backtest(rows, config())

    def test_gap_in_calendar_is_not_bridged(self):
        """A missing day is never silently treated as an open or closed day."""
        rows = [row for row in make_rows('A', 'F', range(10), lead=1, sold=1.0)
                if row['day'] != 5]
        result = run_backtest(rows, config(settings={'min_selection_windows': 999}))
        window = forecast(result, 'A', 4)['evaluation']
        self.assertEqual(window['classification'], UNIDENTIFIABLE)
        self.assertEqual(window['missing_days'], [5])
        self.assertFalse(window['used_for_selection'])


class SensitivityTest(unittest.TestCase):
    def test_bounds_are_reported_without_claiming_recovery(self):
        hours = {day: 24.0 for day in range(12)}
        hours[5] = 8.0
        rows = make_rows('A', 'F', range(12), lead=3, sold=2.0, hours=hours)
        result = run_backtest(rows, config())
        censored = [item for item in result['unidentifiable']
                    if item['reason'] == 'partial_days_in_window']
        self.assertTrue(censored)
        bounds = censored[0]['bounds']
        self.assertEqual(bounds['demand_lower_bound'], censored[0]['observed_units_lower_bound'])
        self.assertIn('敏感', bounds['assumption'])
        self.assertIn('不是', bounds['assumption'])
        self.assertIn('恢复', bounds['assumption'])
        self.assertTrue(bounds['ceiling_available'])
        self.assertTrue(bounds['upper_bound_is_model_dependent'])

    def test_sensitivity_scenarios_span_lower_and_upper(self):
        hours = {day: 24.0 for day in range(12)}
        hours[6] = 4.0
        rows = make_rows('A', 'F', range(12), lead=3, sold=1.0, hours=hours)
        result = run_backtest(rows, config())
        factors = [row['factor'] for row in result['sensitivity']['F']]
        self.assertEqual(factors, [0.0, 0.5, 1.0])
        costs = [row['expected_cost_per_sku_origin'] for row in result['sensitivity']['F']]
        self.assertEqual(costs, sorted(costs))

    def test_no_exposure_upper_bound_is_not_fabricated(self):
        hours = {day: 24.0 for day in range(12)}
        for day in range(4, 9):
            hours[day] = 0.0
        rows = make_rows('A', 'F', range(12), lead=2, sold=0.0, hours=hours)
        result = run_backtest(rows, config(settings={'min_selection_windows': 999}))
        closed = [item for item in result['unidentifiable']
                  if item['origin'] == 3 and item['reason'] == 'closed_days_in_window']
        self.assertTrue(closed)
        bounds = closed[0]['bounds']
        if not bounds['ceiling_available']:
            self.assertTrue(all(value is None
                                for value in bounds['demand_estimate_ceiling_by_model'].values()))


class ConfigTest(unittest.TestCase):
    def test_repository_config_loads_and_is_explicit(self):
        loaded = load_config(CONFIG)
        self.assertGreater(loaded['costs']['shortage_cost_per_unit'], 0.0)
        self.assertIn('candidate_quantiles', loaded['settings'])
        self.assertEqual(config_fingerprint(loaded), config_fingerprint(normalize_config(json.loads(
            CONFIG.read_text(encoding='utf-8')))))

    def test_missing_cost_rejected(self):
        with self.assertRaisesRegex(ValueError, '缺少成本配置'):
            normalize_config({'costs': {'shortage_cost_per_unit': 1.0}})

    def test_non_positive_shortage_rejected(self):
        with self.assertRaisesRegex(ValueError, '必须为正数'):
            normalize_config({'costs': {'shortage_cost_per_unit': 0.0,
                                        'holding_cost_per_unit_per_day': 1.0}})

    def test_inverted_quantile_range_rejected(self):
        with self.assertRaisesRegex(ValueError, '不能大于'):
            config(settings={'min_order_quantile': 0.9, 'max_order_quantile': 0.5})

    def test_empty_candidate_range_rejected(self):
        with self.assertRaisesRegex(ValueError, '没有任何候选'):
            config(settings={'min_order_quantile': 0.2, 'max_order_quantile': 0.3})

    def test_sensitivity_factors_must_include_one(self):
        with self.assertRaisesRegex(ValueError, '必须包含 1.0'):
            config(settings={'demand_sensitivity_factors': [0.0, 0.5]})


class InternalConsistencyTest(unittest.TestCase):
    def test_reported_means_match_predictive_distributions(self):
        """Every stated mean must reconcile with the pmf that produced it."""
        result = run_backtest(load_records(ROOT / 'data' / 'observations.csv'),
                              load_config(CONFIG))
        for item in result['forecasts']:
            for name, model in item['models'].items():
                stated = model['parameters']['mean_window_units']
                self.assertAlmostEqual(stated, model['mean_units'], places=6,
                                       msg=f'{item["sku"]}@{item["origin"]} {name}')
                self.assertAlmostEqual(
                    model['parameters']['mean_units_per_day'] * item['lead_days'],
                    stated, places=6, msg=f'{item["sku"]}@{item["origin"]} {name}')

    def test_quantiles_are_monotone_in_quantile_level(self):
        result = run_backtest(load_records(ROOT / 'data' / 'observations.csv'),
                              load_config(CONFIG))
        for item in result['forecasts']:
            for model in item['models'].values():
                ordered = [entry['quantity'] for _, entry
                           in sorted(model['quantities'].items(), key=lambda pair: float(pair[0]))]
                self.assertEqual(ordered, sorted(ordered))

    def test_stockout_probability_decreases_with_quantity(self):
        result = run_backtest(load_records(ROOT / 'data' / 'observations.csv'),
                              load_config(CONFIG))
        for item in result['forecasts']:
            for model in item['models'].values():
                tails = [value for _, value in sorted(
                    model['stockout_probability_by_quantile'].items(),
                    key=lambda pair: float(pair[0]))]
                for earlier, later in zip(tails, tails[1:]):
                    self.assertLessEqual(later, earlier + 1e-12)


class CliTest(unittest.TestCase):
    def test_backtest_cli_matches_library(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'backtest.json'
            run = subprocess.run(
                [sys.executable, '-m', 'analysis_app', 'backtest',
                 '--input', str(ROOT / 'data' / 'observations.csv'),
                 '--config', str(CONFIG), '--output', str(path)],
                text=True, capture_output=True, cwd=ROOT)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(path.read_text()), json.loads(run.stdout))

    def test_bad_config_leaves_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / 'bad.json'
            config_path.write_text('{"costs": {}}', encoding='utf-8')
            output = Path(directory) / 'result.json'
            run = subprocess.run(
                [sys.executable, '-m', 'analysis_app', 'backtest',
                 '--input', str(ROOT / 'data' / 'observations.csv'),
                 '--config', str(config_path), '--output', str(output)],
                text=True, capture_output=True, cwd=ROOT)
            self.assertEqual(run.returncode, 2)
            self.assertIn('缺少成本配置', run.stderr)
            self.assertFalse(output.exists())

    def test_summary_command_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'summary.json'
            run = subprocess.run(
                [sys.executable, '-m', 'analysis_app', 'summary',
                 '--input', str(ROOT / 'data' / 'observations.csv'),
                 '--output', str(path)],
                text=True, capture_output=True, cwd=ROOT)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(path.read_text()), json.loads(run.stdout))


if __name__ == '__main__':
    unittest.main()
