"""Point-in-time demand distributions for the rolling replenishment backtest.

Every model is fitted from a slice of one SKU's history that was known at the
forecast origin.  Availability is treated strictly as *sales exposure*:

* ``available_hours == 0`` contributes no exposure and no demand evidence;
* ``0 < available_hours < 24`` contributes fractional exposure (a partial day);
* ``available_hours >= 24`` is a full day.

Observed sales on partial or closed days are **never** scaled up into "true
demand".  Training rates are lower bounds when partial days are present, and
the report says so.

Internally the occurrence rate is expressed per *exposure unit* (24 open
hours), then multiplied by the mean open hours per calendar day observed in
the known history.  That is the only way the models use future availability:
as an assumption that upcoming days resemble the known days.  It is reported.
"""

import math
from statistics import fmean

EXPOSURE_UNIT_HOURS = 24.0
FULL_DAY_HOURS = 24.0
# Upper bound on the whole-unit support of a single-day distribution.
SIZE_SUPPORT = 200
# Pseudo-exposure (in exposure units) used by the hurdle model's smoothing.
HURDLE_PSEUDO_EXPOSURE = 0.5
# Rule-of-three upper bound on a zero-event occurrence rate.
ZERO_EVENT_CONFIDENCE = 3.0


# ---------------------------------------------------------------------------
# Small distribution helpers
# ---------------------------------------------------------------------------

def poisson_pmf(mean, cap):
    if mean <= 0:
        return [1.0] + [0.0] * cap
    probabilities = [math.exp(-mean)]
    for k in range(1, cap + 1):
        probabilities.append(probabilities[-1] * mean / k)
    total = math.fsum(probabilities)
    if total <= 0:
        return [1.0] + [0.0] * cap
    return [p / total for p in probabilities]


def empirical_size_pmf(values):
    """Distribution of positive event sizes on whole units (minimum 1)."""
    if not values:
        return [0.0, 1.0]
    counts = {}
    for value in values:
        units = min(SIZE_SUPPORT, max(1, int(round(value))))
        counts[units] = counts.get(units, 0) + 1
    total = sum(counts.values())
    support = max(counts)
    probabilities = [0.0] * (support + 1)
    for units, count in counts.items():
        probabilities[units] = count / total
    return probabilities


def compound_pmf(occurrence_probability, size_pmf, cap):
    """Zero-inflated daily distribution: no event with 1-q, else ``size_pmf``."""
    q = min(1.0, max(0.0, occurrence_probability))
    length = max(2, min(cap + 1, len(size_pmf)))
    probabilities = [0.0] * length
    probabilities[0] = 1.0 - q
    for units in range(1, length):
        if units < len(size_pmf):
            probabilities[units] += q * size_pmf[units]
    total = math.fsum(probabilities)
    if total <= 0:
        return [1.0] + [0.0] * (length - 1)
    return [p / total for p in probabilities]


def convolve(first, second, cap):
    length = min(len(first) + len(second) - 1, cap + 1)
    probabilities = [0.0] * length
    for i, a in enumerate(first):
        if a == 0.0:
            continue
        for j, b in enumerate(second):
            if b == 0.0 or i + j >= length:
                continue
            probabilities[i + j] += a * b
    total = math.fsum(probabilities)
    if total <= 0:
        probabilities[0] = 1.0
        return probabilities
    return [p / total for p in probabilities]


def window_pmf(day_pmf, lead_days):
    lead = max(1, lead_days)
    cap = max(30, int(math.ceil(len(day_pmf) * lead)))
    result = [1.0]
    for _ in range(lead):
        result = convolve(result, day_pmf, cap)
    return result


def pmf_mean(probabilities):
    return math.fsum(index * probability for index, probability in enumerate(probabilities))


def pmf_quantile(probabilities, quantile):
    """Smallest whole-unit quantity q with P(D <= q) >= quantile.

    Returns ``(quantity, truncated)`` where ``truncated`` marks that the finite
    support does not reach the requested quantile.
    """
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if cumulative >= quantile - 1e-15:
            return index, False
    return len(probabilities) - 1, True


def pmf_tail_probability(probabilities, quantity):
    if quantity < 0:
        return 1.0
    return math.fsum(probabilities[quantity + 1:])


# ---------------------------------------------------------------------------
# Exposure profile
# ---------------------------------------------------------------------------

def exposure_profile(history):
    """Summarise availability from known days only.

    ``mean_hours_per_day`` includes closed days in the denominator, so it is
    the expected open hours on a future calendar day if availability keeps
    resembling the known history.
    """
    if not history:
        return {
            'days_known': 0, 'exposed_days': 0, 'full_days': 0,
            'partial_days': 0, 'closed_days': 0,
            'total_exposed_hours': 0.0, 'exposure_units': 0.0,
            'mean_hours_per_day': 0.0, 'exposure_per_calendar_day': 0.0,
            'mean_hours_per_exposed_day': 0.0, 'has_exposure': False,
        }
    hours = [row['available_hours'] for row in history]
    exposed = [value for value in hours if value > 0]
    full = [value for value in hours if value >= FULL_DAY_HOURS]
    partial = [value for value in hours if 0 < value < FULL_DAY_HOURS]
    closed = [value for value in hours if value <= 0]
    total_exposed_hours = math.fsum(exposed)
    mean_hours_per_day = math.fsum(hours) / len(hours)
    return {
        'days_known': len(hours),
        'exposed_days': len(exposed),
        'full_days': len(full),
        'partial_days': len(partial),
        'closed_days': len(closed),
        'total_exposed_hours': total_exposed_hours,
        'exposure_units': total_exposed_hours / EXPOSURE_UNIT_HOURS,
        'mean_hours_per_day': mean_hours_per_day,
        'exposure_per_calendar_day': mean_hours_per_day / EXPOSURE_UNIT_HOURS,
        'mean_hours_per_exposed_day': (total_exposed_hours / len(exposed)) if exposed else 0.0,
        'has_exposure': bool(exposed),
    }


def training_rows(history, settings):
    rows = list(history)
    if settings['exclude_promotion_from_training']:
        rows = [row for row in rows if row['promotion'] == 0]
    return rows


def _ses(values, alpha):
    if not values:
        return None
    estimate = values[0]
    for value in values[1:]:
        estimate = alpha * value + (1 - alpha) * estimate
    return estimate


def _event_sequence(train):
    """Exposed days in order with cumulative exposure units."""
    cumulative = 0.0
    exposed = []
    for row in train:
        if row['available_hours'] <= 0:
            continue
        cumulative += row['available_hours'] / EXPOSURE_UNIT_HOURS
        exposed.append((cumulative, row))
    return exposed


# ---------------------------------------------------------------------------
# Candidate models
# ---------------------------------------------------------------------------

def fit_croston_sba(train, settings):
    """Croston with the Syntetos-Boylan approximation.

    Intervals are measured in exposure units rather than calendar days, so a
    closed day neither advances the interval nor counts as a zero-demand day.
    """
    alpha = settings['croston_alpha']
    exposed = _event_sequence(train)
    events = [(cumulative, row) for cumulative, row in exposed if row['sold'] > 0]
    if not events:
        total_units = exposed[-1][0] if exposed else 0.0
        return {
            'name': 'croston_sba', 'occurrence_per_exposure_unit': 0.0,
            'sba_factor': 1.0 - alpha / 2.0, 'events': 0,
            'exposed_days': len(exposed), 'size': 0.0,
            'mean_interval_exposure_units': None, 'size_pmf': [0.0, 1.0],
            'zero_events': True,
            'occurrence_upper_bound_per_exposure_unit': (
                ZERO_EVENT_CONFIDENCE / total_units if total_units > 0 else None),
        }
    intervals = []
    previous = None
    for cumulative, _ in events:
        if previous is not None:
            # Only gaps between events are informative.  The exposure before the
            # first event is left-censored: the history may simply start close
            # to that event, so measuring a "gap" from the start of the window
            # would systematically overstate the occurrence rate.
            intervals.append(max(cumulative - previous, 1e-9))
        previous = cumulative
    if intervals:
        interval_hat = _ses(intervals, alpha)
    else:
        # A single event gives no inter-arrival interval.  Falling back to the
        # textbook "interval = 1" would imply an event on almost every day even
        # though the event was rare in the observed exposure; use the observed
        # spacing instead (total exposure / events).
        interval_hat = exposed[-1][0] / len(events)
    size_hat = _ses([row['sold'] for _, row in events], alpha)
    rate = (1.0 / interval_hat) * (1.0 - alpha / 2.0)
    return {
        'name': 'croston_sba', 'occurrence_per_exposure_unit': rate,
        'sba_factor': 1.0 - alpha / 2.0, 'events': len(events),
        'exposed_days': len(exposed), 'size': size_hat,
        'mean_interval_exposure_units': interval_hat,
        'size_pmf': empirical_size_pmf([row['sold'] for _, row in events]),
        'zero_events': False,
    }


def fit_exposure_hurdle(train, settings):
    """Two-part hurdle model with smoothing on exposure units."""
    exposed = _event_sequence(train)
    events = [(cumulative, row) for cumulative, row in exposed if row['sold'] > 0]
    total_units = exposed[-1][0] if exposed else 0.0
    if events:
        rate = ((len(events) + HURDLE_PSEUDO_EXPOSURE)
                / (total_units + HURDLE_PSEUDO_EXPOSURE))
        size_hat = ((math.fsum(row['sold'] for _, row in events) + HURDLE_PSEUDO_EXPOSURE)
                    / (len(events) + 1.0))
        size_pmf = empirical_size_pmf([row['sold'] for _, row in events])
        upper = None
    else:
        # "No event observed" is not "demand is zero".  With only n exposure
        # units the 95% rule-of-three bound on the occurrence rate is 3/n, and
        # we use that as a conservative *upper* estimate rather than a
        # confident zero.  The result is labelled as a bound, not a point fit.
        rate = ZERO_EVENT_CONFIDENCE / total_units if total_units > 0 else 0.0
        size_hat = 1.0
        size_pmf = [0.0, 1.0]
        upper = rate
    return {
        'name': 'exposure_hurdle', 'occurrence_per_exposure_unit': rate,
        'events': len(events), 'exposed_days': len(exposed), 'size': size_hat,
        'size_pmf': size_pmf, 'zero_events': not events,
        'occurrence_is_upper_bound': not events,
        'occurrence_upper_bound_per_exposure_unit': upper,
    }


def fit_exposure_poisson(train, settings):
    """Exposure-aware baseline: Poisson demand per exposed hour."""
    hours = math.fsum(row['available_hours'] for row in train)
    sold = math.fsum(row['sold'] for row in train)
    rate = sold / hours if hours > 0 else 0.0
    return {
        'name': 'exposure_poisson', 'units_per_exposed_hour': rate,
        'observed_sold': sold, 'observed_exposed_hours': hours,
        'zero_events': sold == 0,
    }


FITTERS = {
    'croston_sba': fit_croston_sba,
    'exposure_hurdle': fit_exposure_hurdle,
    'exposure_poisson': fit_exposure_poisson,
}
MODEL_NAMES = tuple(FITTERS)


def fit_all(train, settings):
    return {name: fitter(train, settings) for name, fitter in FITTERS.items()}


def model_params_for_output(params):
    return {key: value for key, value in params.items() if key != 'size_pmf'}


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def _occurrence_probability(rate, exposure_fraction):
    """Probability of at least one sale event on a day with ``exposure_fraction``.

    ``rate`` is events per *exposure unit* (24 open hours).  Intermittent demand
    is modelled as at most one event per exposed day, so the probability is the
    Bernoulli reading ``min(1, rate * exposure)``.  For the sparse rates this
    backtest targets this preserves Croston's expected demand ``size / interval``
    exactly, and a partially open day scales the chance of a sale linearly.

    Limitation: for very dense demand (rate >= 1 per exposure unit) the model
    saturates at one event per day and would understate demand.  That regime is
    outside intermittent-demand scope and is reported through the saturation
    flag rather than silently extrapolated.
    """
    if rate <= 0 or exposure_fraction <= 0:
        return 0.0
    return min(1.0, rate * exposure_fraction)


def _support_cap(mean_window):
    return max(30, int(math.ceil(mean_window * 4 + 8 * math.sqrt(mean_window + 1) + 10)))


def predict_day_pmf(name, params, exposure):
    exposure_fraction = exposure['exposure_per_calendar_day']
    if name in ('croston_sba', 'exposure_hurdle'):
        occurrence = _occurrence_probability(
            params['occurrence_per_exposure_unit'], exposure_fraction)
        mean_day = occurrence * params['size']
        return compound_pmf(occurrence, params['size_pmf'], _support_cap(mean_day * 8))
    if name == 'exposure_poisson':
        mean_day = exposure['mean_hours_per_day'] * params['units_per_exposed_hour']
        return poisson_pmf(mean_day, _support_cap(mean_day * 8))
    raise ValueError(f'未知模型：{name}')


def predict_parameters(name, params, exposure, lead_days):
    """Human-readable fitted inputs to the window forecast.

    The reported means are taken from the *same* distributions that produce the
    predictive pmf, so they reconcile exactly with the quantiles.  Where a
    model also keeps a smoothed scalar (Croston's size / interval), it is
    reported separately for transparency.
    """
    lead = max(1, lead_days)
    if name in ('croston_sba', 'exposure_hurdle'):
        rate = params['occurrence_per_exposure_unit']
        occurrence_day = _occurrence_probability(
            rate, exposure['exposure_per_calendar_day'])
        distribution_size = pmf_mean(params['size_pmf'])
        mean_day = occurrence_day * distribution_size
        return {'occurrence_rate_per_exposure_unit': rate,
                'occurrence_probability_per_day': occurrence_day,
                'mean_event_size': distribution_size,
                'smoothed_event_size': params['size'],
                'mean_units_per_day': mean_day,
                'mean_window_units': mean_day * lead}
    if name == 'exposure_poisson':
        mean_day = exposure['mean_hours_per_day'] * params['units_per_exposed_hour']
        return {'units_per_exposed_hour': params['units_per_exposed_hour'],
                'mean_hours_per_day': exposure['mean_hours_per_day'],
                'mean_units_per_day': mean_day,
                'mean_window_units': mean_day * lead}
    raise ValueError(f'未知模型：{name}')


def predict_window_pmf(name, params, exposure, lead_days):
    lead = max(1, lead_days)
    return window_pmf(predict_day_pmf(name, params, exposure), lead)
