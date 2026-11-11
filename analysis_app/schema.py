FIELDS = {'observation_id': 'str', 'sku': 'str', 'family': 'str', 'day': 'int', 'sold': 'float', 'available_hours': 'float', 'lead_days': 'int', 'price': 'float', 'promotion': 'int'}
REQUIRED = ['observation_id', 'sku', 'family', 'sold']
ENTITY = 'sku'
GROUP = 'family'
MEASURE = 'sold'
PRODUCT = 'intermittent-demand'
UNIT = '件'
