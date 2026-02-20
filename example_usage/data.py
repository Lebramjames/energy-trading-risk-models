# %%
from src.data.catalog import DataCatalog
from src.data.interface import DataInterface

catalog = DataCatalog("data/metadata/catalog.yml")
di = DataInterface(catalog)

di.available()              # nice overview of what exists
ds = di.load("ttf_gas")     # returns TimeSeriesDataset
ds.coverage()
ds.missing_report()