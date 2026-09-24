import importlib
MODULES = [
 'resilience_app.core.shared','resilience_app.services.data_loader','resilience_app.features.home',
 'resilience_app.features.urban_features','resilience_app.features.risk_mapping','resilience_app.features.forecasting',
 'resilience_app.features.urban_heat','resilience_app.features.ai_query','resilience_app.features.chat',
 'resilience_app.features.demographics','resilience_app.features.green_roof','resilience_app.features.rain_garden'
]
def test_modules_import():
    for name in MODULES: importlib.import_module(name)
