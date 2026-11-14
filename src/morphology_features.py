"""Building-morphology, sky-view-factor and roof-albedo features (geo-feature
plan, families B "morphology" and C "roof colour"; table rows 4-8).

Primary (pre-registered) outputs of `compute_morphology_features`:
    lp_100, h_mean_100, svf_full, roof_albedo_250, roof_dark_frac_250
Exploration-only outputs (not part of the pre-registered ablation):
    fai_r_dir (8 compass dirs, r=100), z0, zd, h_max_100, h_std_100,
    dist_nearest_bld

`build_synthetic_dsm` is also imported by `shadow_features.py`.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from numba import njit, prange
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject

from city_config import CityConfig

_COMPASS_DIRS = {"N": 90, "NE": 45, "E": 0, "SE": -45, "S": -90, "SW": -135, "W": 180, "NW": 135}
_VON_KARMAN = 0.4
_MACDONALD_A = 4.43
_MACDONALD_CD = 1.2
_MACDONALD_BETA = 1.0
_ROOF_DARK_ALBEDO_THRESHOLD = 0.2  # S2-broadband-albedo cutoff for "dark roof"; matches typical fresh-asphalt/dark-membrane albedo


@contextmanager
def _as_dataset(src):
    """`src` may be a path (we open/close it) or an already-open rasterio dataset (caller owns it)."""
    if isinstance(src, (str, Path)):
        ds = rasterio.open(src)
        try:
            yield ds
        finally:
            ds.close()
    else:
        yield src


def _grid_transform(bounds, resolution_m: float):
    minx, miny, maxx, maxy = bounds
    width = max(1, int(np.ceil((maxx - minx) / resolution_m)))
    height = max(1, int(np.ceil((maxy - miny) / resolution_m)))
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    return transform, width, height


def _reproject_to_grid(src_ds, dst_transform: Affine, width: int, height: int, dst_crs, resampling=Resampling.bilinear) -> np.ndarray:
    dst = np.zeros((height, width), dtype=np.float32)
    reproject(
        source=rasterio.band(src_ds, 1),
        destination=dst,
        src_transform=src_ds.transform, src_crs=src_ds.crs,
        dst_transform=dst_transform, dst_crs=dst_crs,
        resampling=resampling,
    )
    return dst


def build_synthetic_dsm(city: CityConfig, bounds, resolution_m: float, dem_path, buildings: gpd.GeoDataFrame,
                         canopy_path=None, height_col: str = "height"):
    """Returns (dsm_array, transform, crs) covering `bounds` (minx,miny,maxx,maxy
    in city.crs) at `resolution_m`. DSM = DEM elevation + building height
    (rasterized inside polygons) + canopy height (0 where `canopy_path` is None).
    """
    transform, width, height = _grid_transform(bounds, resolution_m)

    with _as_dataset(dem_path) as dem_ds:
        dem_arr = _reproject_to_grid(dem_ds, transform, width, height, city.crs, Resampling.bilinear)

    if len(buildings):
        shapes = list(zip(buildings.geometry, buildings[height_col].astype(float)))
        bld_arr = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0.0, dtype="float32")
    else:
        bld_arr = np.zeros((height, width), dtype=np.float32)

    if canopy_path is not None:
        with _as_dataset(canopy_path) as canopy_ds:
            canopy_arr = _reproject_to_grid(canopy_ds, transform, width, height, city.crs, Resampling.bilinear)
        canopy_arr = np.clip(canopy_arr, 0, None)
    else:
        canopy_arr = np.zeros((height, width), dtype=np.float32)

    dsm = dem_arr + bld_arr + canopy_arr
    return dsm.astype(np.float32), transform, city.crs


@njit(parallel=True, cache=True)
def _svf_ray_trace(dsm: np.ndarray, res_x: float, res_y: float, eye_height: float,
                    n_dir: int, max_radius_m: float) -> np.ndarray:
    """Ray-traced sky view factor at `eye_height` above the DSM surface, per
    pixel. Port of the nearest-neighbour ray-march in
    reference/UrbanSENSE_ML/1_Creating_Input_Data/Calculate_treecanopy_SVFs.ipynb
    (GPU kernel `svfcalculator`), translated to CPU/numba, generalized from
    tree-only DSM to full (terrain+building+canopy) DSM, and from fixed 1m
    steps to a resolution-aware step size.
    """
    height, width = dsm.shape
    out = np.zeros((height, width), dtype=np.float32)
    step_m = min(res_x, res_y)
    n_steps = max(1, int(max_radius_m / step_m))
    two_pi_over_n = 2.0 * np.pi / n_dir

    for iy in prange(height):
        for ix in range(width):
            eye_z = dsm[iy, ix] + eye_height
            svf_sum = 0.0
            for d in range(n_dir):
                theta = d * two_pi_over_n
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)
                beta_max = 0.0
                for s in range(1, n_steps + 1):
                    radius_m = s * step_m
                    jx = ix + int(round(s * step_m / res_x * cos_t))
                    jy = iy - int(round(s * step_m / res_y * sin_t))
                    if jx < 0 or jx >= width or jy < 0 or jy >= height:
                        break
                    rise = dsm[jy, jx] - eye_z
                    beta = np.arctan2(rise, radius_m)
                    if beta > beta_max:
                        beta_max = beta
                svf_sum += np.cos(beta_max) ** 2
            out[iy, ix] = svf_sum / n_dir
    return out


def svf_at_points(dsm: np.ndarray, transform: Affine, points: gpd.GeoSeries, eye_height: float = 1.5,
                   n_dir: int = 36, max_radius_m: float = 50.0) -> np.ndarray:
    """SVF only at the given point locations (cheap: full-grid ray trace restricted
    to the pixels under `points`, since notebook 10 only ever needs per-sensor values)."""
    res_x, res_y = transform.a, -transform.e
    svf_grid = _svf_ray_trace(dsm, float(res_x), float(res_y), float(eye_height), int(n_dir), float(max_radius_m))
    inv = ~transform
    rows, cols = [], []
    for pt in points:
        col, row = inv * (pt.x, pt.y)
        cols.append(int(np.clip(round(col), 0, dsm.shape[1] - 1)))
        rows.append(int(np.clip(round(row), 0, dsm.shape[0] - 1)))
    return svf_grid[rows, cols]


def _frontal_area_index(buildings_in_buffer: gpd.GeoDataFrame, height_col: str, buffer_area: float) -> dict:
    """Approximate FAI per compass direction: for each building, project its
    footprint onto the axis perpendicular to the wind direction (silhouette
    width as seen from that direction) and multiply by height."""
    out = {}
    if not len(buildings_in_buffer):
        return {f"fai_100_{d}": 0.0 for d in _COMPASS_DIRS}
    for d, angle_deg in _COMPASS_DIRS.items():
        theta = np.radians(angle_deg)
        perp = np.array([-np.sin(theta), np.cos(theta)])
        frontal_area = 0.0
        for geom, h in zip(buildings_in_buffer.geometry, buildings_in_buffer[height_col]):
            xs, ys = np.asarray(geom.exterior.coords).T if geom.geom_type == "Polygon" else (np.array([]), np.array([]))
            if xs.size == 0:
                continue
            proj = xs * perp[0] + ys * perp[1]
            width = proj.max() - proj.min()
            frontal_area += width * h
        out[f"fai_100_{d}"] = frontal_area / buffer_area
    return out


def _macdonald_z0_zd(h_mean: float, lp: float, fai_mean: float) -> tuple:
    """Macdonald, Griffiths & Hall (1998) roughness length / displacement height."""
    if h_mean <= 0 or lp <= 0:
        return 0.0, 0.0
    zd = h_mean * (1 + _MACDONALD_A ** (-lp) * (lp - 1))
    exponent = -0.5 * (0.5 * _MACDONALD_BETA * _MACDONALD_CD / _VON_KARMAN ** 2 * (1 - zd / h_mean) * fai_mean) ** -0.5
    z0 = h_mean * (1 - zd / h_mean) * np.exp(exponent) if fai_mean > 0 else 0.0
    return float(z0), float(zd)


def compute_morphology_features(city: CityConfig, sensors: gpd.GeoDataFrame, dem_path, buildings_path,
                                 s2_albedo_path, canopy_path=None, buffer_radii: dict | None = None,
                                 height_col: str = "height") -> pd.DataFrame:
    """sensors: GeoDataFrame indexed by Sensor_ID, geometry = point in city.crs.
    buildings_path: a GeoDataFrame, or a path to a GBA LoD1 (geo)parquet file
    with a height column named `height_col` (GBA's own schema calls this
    `height`; pass a different name if a given extract uses another one).
    Returns a DataFrame indexed by Sensor_ID.
    """
    buffer_radii = buffer_radii or {"roof": 250.0, "morph": 100.0}

    if isinstance(buildings_path, gpd.GeoDataFrame):
        buildings = buildings_path
    else:
        buildings = gpd.read_parquet(buildings_path)
    if height_col not in buildings.columns:
        raise ValueError(f"buildings data has no column {height_col!r}; got {list(buildings.columns)}. Pass height_col=...")
    buildings = buildings.to_crs(city.crs) if buildings.crs is not None else buildings

    r_morph = buffer_radii["morph"]
    r_roof = buffer_radii["roof"]

    # svf_at_points only needs a DSM within ~max_radius_m of each sensor,
    # so build one small local tile per sensor rather than a single grid
    # spanning the whole sensor network - at 1m resolution a city-scale
    # (tens-of-km) network would need a multi-hundred-million-pixel array.
    svf_radius_m = 60.0
    svf_vals = np.empty(len(sensors), dtype=float)
    bld_sindex_svf = buildings.sindex
    for i, (sensor_id, geom) in enumerate(zip(sensors.index, sensors.geometry)):
        local_bounds = (geom.x - svf_radius_m, geom.y - svf_radius_m,
                         geom.x + svf_radius_m, geom.y + svf_radius_m)
        local_cand_idx = list(bld_sindex_svf.intersection(local_bounds))
        local_buildings = buildings.iloc[local_cand_idx]
        local_dsm, local_transform, _ = build_synthetic_dsm(
            city, local_bounds, resolution_m=1.0, dem_path=dem_path, buildings=local_buildings,
            canopy_path=canopy_path, height_col=height_col,
        )
        svf_vals[i] = svf_at_points(local_dsm, local_transform, gpd.GeoSeries([geom], crs=city.crs))[0]

    bld_sindex = buildings.sindex
    rows = []
    for sensor_id, geom in zip(sensors.index, sensors.geometry):
        buf_morph = geom.buffer(r_morph)
        buf_roof = geom.buffer(r_roof)

        cand_idx = list(bld_sindex.intersection(buf_morph.bounds))
        cand = buildings.iloc[cand_idx]
        in_morph = cand[cand.intersects(buf_morph)].copy()
        morph_area = buf_morph.area

        if len(in_morph):
            clipped = in_morph.geometry.intersection(buf_morph)
            clipped_area = clipped.area
            lp_100 = float(clipped_area.sum() / morph_area)
            weights = clipped_area.values
            heights = in_morph[height_col].values
            h_mean_100 = float(np.average(heights, weights=weights)) if weights.sum() > 0 else 0.0
            h_max_100 = float(heights.max())
            h_std_100 = float(heights.std())
            dist_nearest_bld = 0.0 if any(in_morph.contains(geom)) else float(in_morph.distance(geom).min())
        else:
            lp_100 = h_mean_100 = h_max_100 = h_std_100 = 0.0
            dist_nearest_bld = float(buildings.distance(geom).min()) if len(buildings) else np.nan

        fai = _frontal_area_index(in_morph, height_col, morph_area)
        fai_mean = float(np.mean(list(fai.values())))
        z0, zd = _macdonald_z0_zd(h_mean_100, lp_100, fai_mean)

        rows.append({
            "Sensor_ID": sensor_id,
            "lp_100": lp_100, "h_mean_100": h_mean_100,
            "h_max_100": h_max_100, "h_std_100": h_std_100,
            "dist_nearest_bld": dist_nearest_bld,
            "z0": z0, "zd": zd,
            **fai,
        })

    out = pd.DataFrame(rows).set_index("Sensor_ID")
    out["svf_full"] = svf_vals

    roof_albedo_250, roof_dark_frac_250 = _roof_albedo_features(sensors, buildings, s2_albedo_path, r_roof, height_col)
    out["roof_albedo_250"] = pd.Series(roof_albedo_250, name="roof_albedo_250")
    out["roof_dark_frac_250"] = pd.Series(roof_dark_frac_250, name="roof_dark_frac_250")
    return out


def _roof_albedo_features(sensors: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame, s2_albedo_path, radius_m: float,
                           height_col: str) -> tuple[dict, dict]:
    with _as_dataset(s2_albedo_path) as ds:
        albedo = ds.read(1).astype(np.float32)
        transform = ds.transform
        nodata = ds.nodata

        roof_mask = rasterize(
            [(geom, 1) for geom in buildings.geometry], out_shape=albedo.shape,
            transform=transform, fill=0, dtype="uint8",
        )

        inv = ~transform
        rows, pixel_h = albedo.shape[0], transform.e
        pixel_w = transform.a
        radius_px = int(np.ceil(radius_m / min(abs(pixel_w), abs(pixel_h))))

        alb_mean, dark_frac = {}, {}
        for sensor_id, pt in zip(sensors.index, sensors.geometry):
            col, row = inv * (pt.x, pt.y)
            col, row = int(round(col)), int(round(row))
            r0, r1 = max(0, row - radius_px), min(albedo.shape[0], row + radius_px + 1)
            c0, c1 = max(0, col - radius_px), min(albedo.shape[1], col + radius_px + 1)
            sub_alb = albedo[r0:r1, c0:c1]
            sub_mask = roof_mask[r0:r1, c0:c1].astype(bool)
            if nodata is not None:
                sub_mask &= (sub_alb != nodata)
            vals = sub_alb[sub_mask]
            if vals.size:
                alb_mean[sensor_id] = float(vals.mean())
                dark_frac[sensor_id] = float((vals < _ROOF_DARK_ALBEDO_THRESHOLD).mean())
            else:
                alb_mean[sensor_id] = np.nan
                dark_frac[sensor_id] = np.nan
    return alb_mean, dark_frac


if __name__ == "__main__":
    import shapely.geometry as sg

    city = CityConfig(name="test", crs="EPSG:32617", tz="US/Eastern", lat=36.0, lon=-78.9,
                       boundary_path="x", sensors_path=None, raw_geo_dir="x", processed_geo_path="x")

    # Synthetic DEM: flat ground at 100m, 400x400m domain, 2m resolution.
    res = 2.0
    n = 200
    dem_transform = from_bounds(0, 0, n * res, n * res, n, n)
    dem_arr = np.full((n, n), 100.0, dtype=np.float32)
    dem_profile = dict(driver="GTiff", height=n, width=n, count=1, dtype="float32",
                        crs=city.crs, transform=dem_transform)
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**dem_profile) as dataset:
            dataset.write(dem_arr, 1)
        dem_ds = memfile.open()

        # Dense tall buildings around (100,100); open ground around (300,300).
        buildings = gpd.GeoDataFrame({
            "height": [30.0, 25.0, 35.0, 28.0],
            "geometry": [
                sg.box(80, 80, 95, 95), sg.box(100, 80, 115, 95),
                sg.box(80, 100, 95, 115), sg.box(100, 100, 115, 115),
            ],
        }, crs=city.crs)

        sensors = gpd.GeoDataFrame(
            {"geometry": [sg.Point(98, 98), sg.Point(300, 300)]},
            index=["dense_block", "open_ground"], crs=city.crs,
        )
        sensors.index.name = "Sensor_ID"

        dsm, dsm_transform, dsm_crs = build_synthetic_dsm(city, (0, 0, n * res, n * res), resolution_m=2.0,
                                                           dem_path=dem_ds, buildings=buildings)
        assert dsm.shape == (n, n)
        assert dsm.max() > 125  # 100m ground + up to 35m building
        print("build_synthetic_dsm: OK, dsm range", dsm.min(), dsm.max())

        svf_vals = svf_at_points(dsm, dsm_transform, sensors.geometry, n_dir=36, max_radius_m=50.0)
        print("svf at sensors:", dict(zip(sensors.index, svf_vals)))
        assert svf_vals[0] < svf_vals[1], "sensor inside the dense block should see less sky than the open one"

        # Synthetic S2 albedo raster, same grid; roofs darker (0.1) than ground (0.4).
        albedo_arr = np.full((n, n), 0.4, dtype=np.float32)
        roof_mask_arr = rasterize([(g, 1) for g in buildings.geometry], out_shape=(n, n), transform=dem_transform)
        albedo_arr[roof_mask_arr.astype(bool)] = 0.1
        albedo_profile = dict(driver="GTiff", height=n, width=n, count=1, dtype="float32",
                               crs=city.crs, transform=dem_transform, nodata=-9999.0)
        with rasterio.io.MemoryFile() as albedo_memfile:
            with albedo_memfile.open(**albedo_profile) as ds:
                ds.write(albedo_arr, 1)
            albedo_ds = albedo_memfile.open()

            feats = compute_morphology_features(
                city, sensors, dem_path=dem_ds, buildings_path=buildings,
                s2_albedo_path=albedo_ds, buffer_radii={"roof": 150.0, "morph": 100.0},
            )
            albedo_ds.close()
        print(feats)

        assert feats.loc["dense_block", "lp_100"] > feats.loc["open_ground", "lp_100"]
        assert feats.loc["dense_block", "h_mean_100"] > feats.loc["open_ground", "h_mean_100"]
        assert feats.loc["dense_block", "svf_full"] < feats.loc["open_ground", "svf_full"]
        # open_ground has no building within its 150m roof-buffer, so its roof_albedo is legitimately NaN.
        assert feats.loc["dense_block", "roof_albedo_250"] < 0.4
        assert np.isnan(feats.loc["open_ground", "roof_albedo_250"])
        assert feats.loc["dense_block", "z0"] > feats.loc["open_ground", "z0"]
        dem_ds.close()

    print("All self-tests passed.")
