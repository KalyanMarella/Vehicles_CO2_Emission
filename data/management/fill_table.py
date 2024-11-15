import pandas as pd
from index import CO2Emission,Session

with Session.begin() as db:
    data = pd.read_csv("data/CO2 Emissions_Canada.csv")
    for index, row in data.iterrows():
        co2_emissions = CO2Emission(
            make=row.iloc[0],
            model=row.iloc[1],
            vehicle_class=row.iloc[2],
            engine_size_l=row.iloc[3],
            cylinders=row.iloc[4],
            transmission=row.iloc[5],
            fuel_type=row.iloc[6],
            fuel_consumption_city_l_per_100_km=row.iloc[7],
            fuel_consumption_hwy_l_per_100_km=row.iloc[8],
            fuel_consumption_comb_l_per_100_km=row.iloc[9],
            fuel_consumption_comb_mpg=row.iloc[10],
            co2_emissions_g_per_km=row.iloc[11],
        )
        db.add(co2_emissions)
