from pysolar.solar import get_altitude, get_azimuth
import datetime
import pytz

# 设置经纬度和时区
lat = 39.9
lon = 116.4
tz = pytz.timezone('Asia/Shanghai')

# 获取当前时间和太阳高度角、方位角
dt = datetime.datetime.now(tz)
altitude = get_altitude(lat, lon, dt)
azimuth = get_azimuth(lat, lon, dt)

# 输出结果
print("当前时间: ", dt)
print("太阳高度角: ", altitude)
print("太阳方位角: ", azimuth)
