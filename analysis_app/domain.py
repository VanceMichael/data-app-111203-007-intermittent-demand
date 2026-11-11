import math
import statistics
from collections import Counter


def describe(records):
    return {'sold_total': math.fsum(row['sold'] for row in records), 'available_hours': math.fsum(row['available_hours'] for row in records), 'sales_value': math.fsum(row['sold'] * row['price'] for row in records)}
