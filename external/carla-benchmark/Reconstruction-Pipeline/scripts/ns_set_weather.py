import carla
import datetime
import argparse
import os
import sys
import pytz
from pysolar import solar
import time
import pandas as pd
import pvlib
from settings import SUMMARY_DIR

def set_sun_position(year, month, day, hour, minute, second, latitude, longitude, 
    timezone_name = 'UTC',
    carla_host = 'localhost', 
    carla_port= 2000
):
    
    try:
        local_tz = pytz.timezone(timezone_name)
        
        naive_dt = datetime.datetime(year, month, day, hour, minute, second)
        
        t_aware = local_tz.localize(naive_dt)
        
        t_utc = t_aware.astimezone(pytz.utc)
        
        
    except pytz.UnknownTimeZoneError:
        print(f"Error: Unknown timezone name '{timezone_name}'. Using UTC instead.")
        t_utc = datetime.datetime(year, month, day, hour, minute, second, tzinfo=pytz.utc)
    except ValueError as e:
        print(f"Error creating datetime object: {e}")
        return

    try:
        altitude_deg = solar.get_altitude(latitude, longitude, t_utc)
        
        azimuth_deg = solar.get_azimuth(latitude, longitude, t_utc)
        
        
    except Exception as e:
        print(f"Error during solar calculation: {e}")
        return

    try:
        client = carla.Client(carla_host, carla_port)
        client.set_timeout(10.0)
        world = client.get_world()
        
        current_weather = world.get_weather()

        custom_weather = carla.WeatherParameters(
            cloudiness=current_weather.cloudiness,
            precipitation=current_weather.precipitation,
            wind_intensity=current_weather.wind_intensity,
            fog_density=current_weather.fog_density,
            wetness=current_weather.wetness,
            precipitation_deposits=current_weather.precipitation_deposits,
            sun_altitude_angle=altitude_deg,  
            sun_azimuth_angle=azimuth_deg    
        )

        world.set_weather(custom_weather)
        
        print(f"\nSuccessfully set CARLA sun position:")
        
    except carla.World.TimeoutException:
        print(f"\nError: Could not connect to CARLA at {carla_host}:{carla_port}. Is the simulator running?")
    except Exception as e:
        print(f"\nAn error occurred while setting CARLA weather: {e}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Spawn vehicles for a specific crash scenario.")
    
    parser.add_argument("crash_id", type=int, nargs='?', help="The ID of the crash scenario to simulate.")

    args = parser.parse_args()

    if args.crash_id is None:
        print("Error: Crash ID is missing. Please provide a crash ID (e.g., python script.py 510163).")
        sys.exit(1)

    crash_id = args.crash_id
    
    with open(os.path.join(SUMMARY_DIR, f"summary_{crash_id}.txt")) as file:
        lines = file.readlines()
    
    for line in lines:
        if "Date" in line:
            target_year = 2023
            target_day = int(line.strip().split(" ")[1])
            target_month = line.strip().split(" ")[2]
            datetime_object = datetime.datetime.strptime(target_month, "%B")
            target_month = datetime_object.month
        if "Time" in line:
            target_hour = int(line.split(":")[1].strip())
            target_minute = int(line.split(":")[2].strip())
            target_second = 0
        if "Latitude" in line:
            target_lat = float(line.split(":")[1].strip())
        if "Longitude" in line:
            target_lon = float(line.split(":")[1].strip())
        target_tz = 'America/New_York'

    set_sun_position(
        target_year, target_month, target_day, 
        target_hour, target_minute, target_second, 
        target_lat, target_lon, 
        target_tz
    )
    
    time.sleep(2)