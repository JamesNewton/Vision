import math
from datetime import datetime

class DaylightMonitor:
    def __init__(self, lat, lon, tz_offset):
        self.lat = lat
        self.lon = lon
        self.tz_offset = tz_offset
        self.last_calc_date = None
        self.sunrise_mins = 0
        self.sunset_mins = 0

    def get_sunrise_sunset(self, lat, lon, tz_offset, date=None):
        """
        Returns (sunrise, sunset) as "HH:MM" strings.
        Accuracy: Typically within +/- 5-10 minutes.
        """
        if date is None: date = datetime.now()
        n = date.timetuple().tm_yday
        b = 2 * math.pi * (n - 81) / 365 #Earth orbit
        declination = math.radians(23.45) * math.sin(b) #tilt
        # Equation of Time (corrects for elliptical orbit, in minutes)
        eot = 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)
        # 12:00 PM (solar noon) + longitude correction - equation of time
        solar_noon = (12 * 60) + ((15 * tz_offset) - lon) * 4 - eot
        lat_rad = math.radians(lat) #travel from sunrise to noon
        try:
            hour_angle = math.degrees(math.acos(-math.tan(lat_rad) * math.tan(declination)))
        except ValueError: # extreme latitude
            if lat * math.degrees(declination) >= 0: # same sign?
                return 0.0, 1440.0 # always up, 24 hour day
            else:
                return 1440.0, 0.0 # always down, 24 hour night
        sunrise = solar_noon - (hour_angle * 4) #4 min per degree rotation
        sunset = solar_noon + (hour_angle * 4)
        return sunrise, sunset

    def fmt(self, m): # Helper to format minutes into 24-hour HH:MM
        return f"{int((m % 1440) // 60):02d}:{int(m % 60):02d}"

    def update(self):
        now = datetime.now()
        current_date = now.date()
        # Only do the math if we haven't done it today
        if current_date != self.last_calc_date:
            srise, sset = self.get_sunrise_sunset(self.lat, self.lon, self.tz_offset, now)
            self.sunrise = self.fmt(srise)
            self.sunset = self.fmt(sset)
            # Convert "HH:MM" strings to minutes-since-midnight for blazing fast comparisons
            self.sunrise_mins = srise
            self.sunset_mins = sset
            self.last_calc_date = current_date
            #print("update")
        return now

    def is_it_dark(self):
        now = self.update()
        current_mins = now.hour * 60 + now.minute
        return current_mins < self.sunrise_mins or current_mins > self.sunset_mins

    def get_rise_set(self):
        self.update()
        return self.sunrise, self.sunset

if __name__ == "__main__":
    # --- Test for Escondido, CA (using standard Pacific Time, UTC-8) ---
    # Note: Adjust tz_offset to -7 if testing during Daylight Saving Time
    monitor = DaylightMonitor(lat=33.11, lon=-117.08, tz_offset=-8)
    rise, set = monitor.get_rise_set()
    print(f"Sunrise: {rise}, Sunset: {set}") #on 2/28, Sunrise: 06:24, Sunset: 17:38
    if (monitor.is_it_dark()):
        print("night")
    else:
        print("day")

