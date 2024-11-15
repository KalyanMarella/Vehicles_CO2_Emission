from index import CO2Emission, Session

with Session.begin() as db:
    result = db.query(CO2Emission).all()
    for row in result:
        print(row.fuel_type)