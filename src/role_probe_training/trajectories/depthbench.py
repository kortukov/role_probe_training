"""Use only the upstream training API implementations, with fresh per-rollout state."""
import json
from pathlib import Path


API_FILES = {
    'BookSearchAPI': ['books.csv'], 'CryptoPriceAPI': ['crypto_price.csv'],
    'CurrencyExchangeAPI': ['exchange_rates.csv'], 'EnglishDictionaryAPI': ['english_dict.csv'],
    'FlightStatusAPI': ['flight_status.csv'], 'MovieSearchAPI': ['movies.csv'],
    'GetLocation': ['location_data.csv'], 'FindDealershipsByLocation': ['car_data.csv'],
    'GetCarListingByDealerships': ['car_data.csv'], 'FilterByCarBuild': ['car_data.csv'],
    'FilterByCarBrand': ['car_data.csv'], 'FilterByCarPrice': ['car_data.csv'],
    'FilterByCarMileage': ['car_data.csv'], 'FilterByCarProductionYear': ['car_data.csv'],
    'FilterByCarTitle': ['car_data.csv'], 'SearchFlight': ['flights.csv'],
    'BookFlight': ['flights.csv'], 'CancelFlight': ['cancel_flight.csv'],
    'SearchSKU': ['products.csv'], 'CreateCart': ['products.csv', 'coupons.csv'],
    'CancelStoreOrder': ['orders.csv'],
}


def normalize_schema(value):
    if isinstance(value, list):
        return [normalize_schema(x) for x in value]
    if isinstance(value, dict):
        return {k: ('object' if k == 'type' and v == 'dict' else normalize_schema(v))
                for k, v in value.items()}
    return value


def tool_schemas(upstream):
    raw = json.loads((Path(upstream)/'src/tier/prompts/function_definitions.json').read_text())
    return [{'type': 'function', 'function': {'name': name, 'description': raw[name]['description'],
             'parameters': normalize_schema(raw[name]['parameters'])}} for name in sorted(API_FILES)]


class DepthBenchEnvironment:
    def __init__(self, upstream):
        from tier.environment import depthbench_apis
        root = Path(upstream)/'src/tier/environment/api_data'
        self.tools = {name: getattr(depthbench_apis, name)(*[str(root/f) for f in files])
                      for name, files in API_FILES.items()}

    def execute(self, name, arguments):
        if name not in self.tools:
            return {'status_code': 400, 'error': f'Unknown or unavailable tool: {name}'}
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
            if not isinstance(args, dict):
                raise ValueError('Tool arguments must be a JSON object')
            return self.tools[name].execute(**args)
        except Exception as error:
            return {'status_code': 400, 'error': f'{type(error).__name__}: {error}'}
