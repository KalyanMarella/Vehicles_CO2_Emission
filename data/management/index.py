import os

from dotenv import load_dotenv
from sqlalchemy import Column, Date, DateTime, Integer, Numeric, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import QueuePool

load_dotenv()


engine = create_engine(
    os.getenv("DB_URL"),
    echo=True,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=-1,
    pool_recycle=3600,
    pool_pre_ping=True,
    connect_args={
        "connect_timeout": 60,
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5,
    },
)
Session = sessionmaker(bind=engine)
connection = engine.connect()
connection.close()


from sqlalchemy import Column, Sequence, SmallInteger, String

Base = declarative_base()

class CO2Emission(Base):
    __tablename__ = "co2_emission_data"
    __table_args__ = {"schema": "public"} 
    id = Column(SmallInteger, Sequence("co2_emission_id_seq"), primary_key=True)
    make = Column(String)
    model = Column(String)
    vehicle_class = Column(String)
    engine_size_l = Column(Numeric(precision=23, scale=15))
    cylinders = Column(SmallInteger)
    transmission = Column(String)
    fuel_type = Column(String)
    fuel_consumption_city_l_per_100_km = Column(Numeric(precision=23, scale=15))
    fuel_consumption_hwy_l_per_100_km = Column(Numeric(precision=23, scale=15))
    fuel_consumption_comb_l_per_100_km = Column(Numeric(precision=23, scale=15))
    fuel_consumption_comb_mpg = Column(Numeric(precision=23, scale=15))
    co2_emissions_g_per_km = Column(Numeric(precision=23, scale=15))