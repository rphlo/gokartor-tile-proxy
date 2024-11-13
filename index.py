from flask import Flask, send_file
from pyproj import Transformer
from io import BytesIO
from PIL import Image
import requests
import numpy as np
import cv2
from flask_caching import Cache
from slippy_tiles import tile_xy_to_north_west_latlon, latlon_to_tile_xy, latlon_to_tile_xy_offset

cache = Cache(config={'CACHE_TYPE': 'FileSystemCache', 'CACHE_DIR': 'cache'})
app = Flask(__name__)
cache.init_app(app)

wgs84_to_crs3006 = Transformer.from_crs(
    "+proj=latlon",
    "+proj=utm +zone=33 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs +type=crs",
)

wgs84_to_crs2056 = Transformer.from_crs(
    "+proj=latlon",
    "+proj=somerc +lat_0=46.9524055555556 +lon_0=7.43958333333333 +k_0=1 +x_0=2600000 +y_0=1200000 +ellps=bessel +towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs +type=crs"
)

wgs84_to_espg27700 = Transformer.from_crs(
    "+proj=latlon",
    "+proj=tmerc +lat_0=49 +lon_0=-2 +k=0.9996012717 +x_0=400000 +y_0=-100000 +ellps=airy +units=m +no_defs +type=crs"
)


def crs3006_to_tile_xy(x, y, zoom):
    scale = (2.0 ** (zoom)) / (2**14) / 256
    xtile = int((x - 265000) * scale)
    ytile = int((7680000 - y) * scale) 
    return (xtile, ytile)


def crs2056_to_tile_xy(x, y, zoom):
    scale = 2 ** zoom / (2 ** 14) / 1000
    xtile = int((x - 248e4) * scale)
    ytile = int((1302e3 - y) * scale) 
    return (xtile, ytile)


def espg27700_to_tile_xy(x, y, zoom):
    scale = 896 / (2 ** zoom) * 256
    xtile = int((x + 238375) / scale)
    ytile = int((1376256 - y) / scale) 
    return (xtile, ytile)


def swedish_tile_xy_to_crs3006_north_west_xy(x_tile, y_tile, zoom):
    scale = (2 ** (zoom)) / (2**14) / 256
    x = x_tile / scale + 265000
    y = 7680000 - y_tile / scale 
    return (x, y)


def swiss_tile_xy_to_crs2056_north_west_xy(x_tile, y_tile, zoom):
    scale = 2 ** zoom / (2 ** 14) / 1000
    x = x_tile / scale + 248e4
    y = 1302e3 - y_tile / scale 
    return (x, y)

def leisure_tile_xy_to_espg27700_north_west_xy(x_tile, y_tile, zoom):
    scale = 896 / (2 ** zoom) * 256
    x = x_tile * scale - 238375
    y = 1376256 - y_tile * scale 
    return (x, y)


def latlon_to_crs3006_tile_coordinates(lat, lon, z):
    zoom = z - 2
    x, y = wgs84_to_crs3006.transform(lon, lat)
    tile_x, tile_y = crs3006_to_tile_xy(x, y, zoom)
    x_min, y_max = swedish_tile_xy_to_crs3006_north_west_xy(tile_x, tile_y, zoom)
    x_max, y_min = swedish_tile_xy_to_crs3006_north_west_xy(tile_x + 1, tile_y + 1, zoom)
    
    tile_height = y_max - y_min
    tile_width = x_max - x_min

    offset_x = (x - x_min) / tile_width * 256
    offset_y = (y_max - y) / tile_height * 256

    return offset_x, offset_y, tile_x, tile_y


def latlon_to_crs2056_tile_coordinates(lat, lon, z):
    zoom = z - 2
    x, y = wgs84_to_crs2056.transform(lon, lat)
    tile_x, tile_y = crs2056_to_tile_xy(x, y, zoom)
    x_min, y_max = swiss_tile_xy_to_crs2056_north_west_xy(tile_x, tile_y, zoom)
    x_max, y_min = swiss_tile_xy_to_crs2056_north_west_xy(tile_x + 1, tile_y + 1, zoom)
    
    tile_height = y_max - y_min
    tile_width = x_max - x_min

    offset_x = (x - x_min) / tile_width * 1000
    offset_y = (y_max - y) / tile_height * 1000

    return offset_x, offset_y, tile_x, tile_y


def latlon_to_espg27700_tile_coordinates(lat, lon, z):
    zoom = z - 7
    x, y = wgs84_to_espg27700.transform(lon, lat)
    tile_x, tile_y = espg27700_to_tile_xy(x, y, zoom)
    x_min, y_max = leisure_tile_xy_to_espg27700_north_west_xy(tile_x, tile_y, zoom)
    x_max, y_min = leisure_tile_xy_to_espg27700_north_west_xy(tile_x + 1, tile_y + 1, zoom)
    
    tile_height = y_max - y_min
    tile_width = x_max - x_min

    offset_x = (x - x_min) / tile_width * 256
    offset_y = (y_max - y) / tile_height * 256

    return offset_x, offset_y, tile_x, tile_y


@cache.memoize(5*60)
def get_gokartor_tile(z, y, x):
    url = f'https://kartor.gokartor.se/Master/{z}/{y}/{x}.png'
    res = requests.get(url, stream=True)
    if res.status_code == 200:
        data = BytesIO(res.raw.read())
        return Image.open(data)
    return None


@cache.memoize(5*60)
def get_mapantch_tile(z, y, x):
    url = f'https://www.mapant.ch/wmts.php?layer=MapAnt%20Switzerland&style=default&tilematrixset=2056&Service=WMTS&Request=GetTile&Version=1.0.0&Format=image%2Fpng&TileMatrix={z}&TileCol={x}&TileRow={y}'
    res = requests.get(url, stream=True)
    if res.status_code == 200:
        data = BytesIO(res.raw.read())
        return Image.open(data)
    return None


@cache.memoize(5*60)
def get_leisure_tile(z, y, x):
    url = f'https://api.os.uk/maps/raster/v1/zxy/Leisure_27700/{z}/{x}/{y}.png?key=5T04qXvNDxLX1gCEAXS0INCgLvczGRYw'
    res = requests.get(url, stream=True)
    if res.status_code == 200:
        data = BytesIO(res.raw.read())
        return Image.open(data)
    return None


@cache.memoize(60*60)
def get_osm_tile(z, y, x):
    url = f"https://a.tile.openstreetmap.org/{z}/{x}/{y}.png"
    session = requests.Session()    
    session.headers.update({'User-Agent': 'Routechoices.com Tiles Proxy App'})
    res = session.get(url, stream=True)
    if res.status_code == 200:
        data = BytesIO(res.raw.read())
        return Image.open(data)
    return None


@app.route('/se/<int:z>/<int:x>/<int:y>.jpg')
@cache.memoize(7*24*3600)
def get_tile_se(z, x, y):
    north, west = tile_xy_to_north_west_latlon(x, y, z)
    south, east = tile_xy_to_north_west_latlon(x + 1, y + 1, z)
    zoom = z - 2
    nw_x, nw_y, nw_tile_x, nw_tile_y = latlon_to_crs3006_tile_coordinates(north, west, z)
    ne_x, ne_y, ne_tile_x, ne_tile_y = latlon_to_crs3006_tile_coordinates(north, east, z)
    se_x, se_y, se_tile_x, se_tile_y = latlon_to_crs3006_tile_coordinates(south, east, z)
    sw_x, sw_y, sw_tile_x, sw_tile_y = latlon_to_crs3006_tile_coordinates(south, west, z)
    
    tile_min_x = min(nw_tile_x, ne_tile_x, se_tile_x, sw_tile_x)
    tile_min_y = min(nw_tile_y, ne_tile_y, se_tile_y, sw_tile_y)
    tile_max_x = max(nw_tile_x, ne_tile_x, se_tile_x, sw_tile_x)
    tile_max_y = max(nw_tile_y, ne_tile_y, se_tile_y, sw_tile_y)

    src_tile_size = 256
    dst_tile_size = 256

    img_width = (tile_max_x - tile_min_x + 1) * src_tile_size
    img_height = (tile_max_y - tile_min_y + 1) * src_tile_size

    p1 = np.float32(
        [
            [0, 0],
            [dst_tile_size, 0],
            [dst_tile_size, dst_tile_size],
            [0, dst_tile_size],
        ]
    )

    p2 = np.float32(
        [
            [nw_x + (nw_tile_x - tile_min_x) * src_tile_size, nw_y + (nw_tile_y - tile_min_y) * src_tile_size],
            [ne_x + (ne_tile_x - tile_min_x) * src_tile_size, ne_y + (ne_tile_y - tile_min_y) * src_tile_size],
            [se_x + (se_tile_x - tile_min_x) * src_tile_size, se_y + (se_tile_y - tile_min_y) * src_tile_size],
            [sw_x + (sw_tile_x - tile_min_x) * src_tile_size, sw_y + (sw_tile_y - tile_min_y) * src_tile_size],
        ]
    )
    im = Image.new(mode="RGB", size=(img_width, img_height), color=(255, 255, 255))
    for yy in range(tile_min_y, tile_max_y+1):
        for xx in range(tile_min_x, tile_max_x+1):
            tile = get_gokartor_tile(zoom, yy, xx)
            if tile:
                Image.Image.paste(im, tile, (int(src_tile_size * (xx - tile_min_x)), int(src_tile_size * (yy - tile_min_y))))
    coeffs, mask = cv2.findHomography( p2, p1,cv2.RANSAC, 5.0)
    img_alpha = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGRA)
    img = cv2.warpPerspective(
        img_alpha,
        coeffs,
        (dst_tile_size, dst_tile_size),
        flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255, 0),
    )
    _, buffer = cv2.imencode(".jpeg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    data_out = BytesIO(buffer)
    return send_file(data_out, mimetype="image/jpeg")


@app.route('/ch/<int:z>/<int:x>/<int:y>.jpg')
@cache.memoize(7*24*3600)
def get_tile_ch(z, x, y):
    north, west = tile_xy_to_north_west_latlon(x, y, z)
    south, east = tile_xy_to_north_west_latlon(x + 1, y + 1, z)
    zoom = z - 7
    nw_x, nw_y, nw_tile_x, nw_tile_y = latlon_to_crs2056_tile_coordinates(north, west, z)
    ne_x, ne_y, ne_tile_x, ne_tile_y = latlon_to_crs2056_tile_coordinates(north, east, z)
    se_x, se_y, se_tile_x, se_tile_y = latlon_to_crs2056_tile_coordinates(south, east, z)
    sw_x, sw_y, sw_tile_x, sw_tile_y = latlon_to_crs2056_tile_coordinates(south, west, z)
    
    tile_min_x = min(nw_tile_x, ne_tile_x, se_tile_x, sw_tile_x)
    tile_min_y = min(nw_tile_y, ne_tile_y, se_tile_y, sw_tile_y)
    tile_max_x = max(nw_tile_x, ne_tile_x, se_tile_x, sw_tile_x)
    tile_max_y = max(nw_tile_y, ne_tile_y, se_tile_y, sw_tile_y)

    src_tile_size = 1000
    dst_tile_size = 256

    img_width = (tile_max_x - tile_min_x + 1) * src_tile_size
    img_height = (tile_max_y - tile_min_y + 1) * src_tile_size

    p1 = np.float32(
        [
            [0, 0],
            [dst_tile_size, 0],
            [dst_tile_size, dst_tile_size],
            [0, dst_tile_size],
        ]
    )

    p2 = np.float32(
        [
            [nw_x + (nw_tile_x - tile_min_x) * src_tile_size, nw_y + (nw_tile_y - tile_min_y) * src_tile_size],
            [ne_x + (ne_tile_x - tile_min_x) * src_tile_size, ne_y + (ne_tile_y - tile_min_y) * src_tile_size],
            [se_x + (se_tile_x - tile_min_x) * src_tile_size, se_y + (se_tile_y - tile_min_y) * src_tile_size],
            [sw_x + (sw_tile_x - tile_min_x) * src_tile_size, sw_y + (sw_tile_y - tile_min_y) * src_tile_size],
        ]
    )
    im = Image.new(mode="RGB", size=(img_width, img_height), color=(255, 255, 255))
    for yy in range(tile_min_y, tile_max_y+1):
        for xx in range(tile_min_x, tile_max_x+1):
            tile = get_mapantch_tile(zoom, yy, xx)
            if tile:
                Image.Image.paste(im, tile, (int(src_tile_size * (xx - tile_min_x)), int(src_tile_size * (yy - tile_min_y))))
    coeffs, mask = cv2.findHomography( p2, p1,cv2.RANSAC, 5.0)
    img_alpha = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGRA)
    img = cv2.warpPerspective(
        img_alpha,
        coeffs,
        (dst_tile_size, dst_tile_size),
        flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255, 0),
    )
    _, buffer = cv2.imencode(".jpeg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    data_out = BytesIO(buffer)
    return send_file(data_out, mimetype="image/jpeg")


@app.route('/uk/<int:z>/<int:x>/<int:y>.webp')
@cache.memoize(7*24*3600)
def get_tile_uk(z, x, y):
    north, west = tile_xy_to_north_west_latlon(x, y, z)
    south, east = tile_xy_to_north_west_latlon(x + 1, y + 1, z)

    nw_x, nw_y, nw_tile_x, nw_tile_y = latlon_to_espg27700_tile_coordinates(north, west, z)
    ne_x, ne_y, ne_tile_x, ne_tile_y = latlon_to_espg27700_tile_coordinates(north, east, z)
    se_x, se_y, se_tile_x, se_tile_y = latlon_to_espg27700_tile_coordinates(south, east, z)
    sw_x, sw_y, sw_tile_x, sw_tile_y = latlon_to_espg27700_tile_coordinates(south, west, z)

    tile_min_x = min(nw_tile_x, ne_tile_x, se_tile_x, sw_tile_x)
    tile_min_y = min(nw_tile_y, ne_tile_y, se_tile_y, sw_tile_y)
    tile_max_x = max(nw_tile_x, ne_tile_x, se_tile_x, sw_tile_x)
    tile_max_y = max(nw_tile_y, ne_tile_y, se_tile_y, sw_tile_y)

    src_tile_size = 256
    dst_tile_size = 256

    img_width = (tile_max_x - tile_min_x + 1) * src_tile_size
    img_height = (tile_max_y - tile_min_y + 1) * src_tile_size

    p1 = np.float32(
        [
            [0, 0],
            [dst_tile_size, 0],
            [dst_tile_size, dst_tile_size],
            [0, dst_tile_size],
        ]
    )

    p2 = np.float32(
        [
            [nw_x + (nw_tile_x - tile_min_x) * src_tile_size, nw_y + (nw_tile_y - tile_min_y) * src_tile_size],
            [ne_x + (ne_tile_x - tile_min_x) * src_tile_size, ne_y + (ne_tile_y - tile_min_y) * src_tile_size],
            [se_x + (se_tile_x - tile_min_x) * src_tile_size, se_y + (se_tile_y - tile_min_y) * src_tile_size],
            [sw_x + (sw_tile_x - tile_min_x) * src_tile_size, sw_y + (sw_tile_y - tile_min_y) * src_tile_size],
        ]
    )
    im = Image.new(mode="RGB", size=(img_width, img_height), color=(255, 255, 255))
    for yy in range(tile_min_y, tile_max_y+1):
        for xx in range(tile_min_x, tile_max_x+1):
            tile = get_leisure_tile(z - 7, yy, xx)

            if tile:
                Image.Image.paste(im, tile, (int(src_tile_size * (xx - tile_min_x)), int(src_tile_size * (yy - tile_min_y))))
    coeffs, mask = cv2.findHomography( p2, p1,cv2.RANSAC, 5.0)
    img_alpha = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGRA)
    img = cv2.warpPerspective(
        img_alpha,
        coeffs,
        (dst_tile_size, dst_tile_size),
        flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255, 0),
    )
    _, buffer = cv2.imencode(".webp", img, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
    data_out = BytesIO(buffer)
    return send_file(data_out, mimetype="image/webp")




@app.route('/garmin/<int:z>/<float:lat>/<float:lng>.jpg')
def get_image_at_location(z, lat, lng):
    x, y = latlon_to_tile_xy(lat, lng, z)
    x_offset, y_offset = latlon_to_tile_xy_offset(lat, lng, z)
    if x_offset < 128:
        x_min = x - 1
        x_max = x
        x_offset += 256
    else:
        x_min = x
        x_max = x + 1
    if y_offset < 128:
        y_min = y - 1
        y_max = y
        y_offset += 256
    else:
        y_min = y
        y_max = y + 1
    final_image = Image.new(mode="RGB", size=(256, 256), color=(255, 255, 255))
    for i in range(2):
        for j in range(2):
            xx = x_min + i
            yy = y_min + j
            paste_xy = (int((256 * i) - x_offset + 128), int((256 * j) - y_offset + 128))
            tile = get_osm_tile(z, yy, xx)
            if tile:    
                Image.Image.paste(final_image, tile, paste_xy)
    data_out = BytesIO()
    final_image.save(data_out, "JPEG")
    data_out.seek(0)
    return send_file(data_out, mimetype="image/jpeg")


@app.route('/')
def index():
    return "Map Tile Proxy"