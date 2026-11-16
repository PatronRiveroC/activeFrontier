import os
import sys
import time
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import mapping
from scipy.ndimage import maximum_filter
from numba import jit, objmode
from numba.typed import List
from PIL import Image, ImageDraw, ImageFont
import subprocess


_PROGRESS = {"total": 0, "rep": 1, "num": 1, "multi": False, "t": 0.0}


def _render_progress(step, pixels_a, pixels_b):
    total = _PROGRESS["total"]
    pct = min(100, int(((pixels_a + pixels_b) / total) * 100)) if total > 0 else 100
    bar_width = 30
    blocks = int(bar_width * pct / 100)
    bar = "=" * blocks + " " * (bar_width - blocks)
    if _PROGRESS["multi"]:
        sys.stdout.write(f"\r[{bar}] {pct}% (Rep {_PROGRESS['rep']}/{_PROGRESS['num']})")
    else:
        sys.stdout.write(f"\r[{bar}] {pct}% (Step: {step} | Pixels A: {pixels_a} | Pixels B: {pixels_b})")
    sys.stdout.flush()


def _update_progress(step, pixels_a, pixels_b):
    now = time.time()
    if now - _PROGRESS["t"] < 0.1:
        return
    _PROGRESS["t"] = now
    _render_progress(step, pixels_a, pixels_b)


@jit(nopython=True, cache=True)
def _build_offsets(d):
    int_d = int(np.ceil(d))
    offsets_dy = []
    offsets_dx = []
    for dy in range(-int_d, int_d + 1):
        for dx in range(-int_d, int_d + 1):
            if dx == 0 and dy == 0:
                continue
            if np.sqrt(dx * dx + dy * dy) <= d:
                offsets_dy.append(dy)
                offsets_dx.append(dx)
    return np.array(offsets_dy, dtype=np.int32), np.array(offsets_dx, dtype=np.int32)


@jit(nopython=True, cache=True)
def _run_competition_simulation_numba(mask_A, mask_B, sA_y, sA_x, sB_y, sB_x, max_steps, prob_A, prob_B, d_A, d_B):
    rows, cols = mask_A.shape
    col_A = np.full((rows, cols), -9999, dtype=np.int32)
    col_B = np.full((rows, cols), -9999, dtype=np.int32)
    owner = np.zeros((rows, cols), dtype=np.int8)
    n_A = 0
    n_B = 0

    fA_y = List()
    fA_x = List()
    fB_y = List()
    fB_x = List()

    for i in range(len(sA_y)):
        y = sA_y[i]
        x = sA_x[i]
        if 0 <= y < rows and 0 <= x < cols and mask_A[y, x] and owner[y, x] == 0:
            owner[y, x] = 1
            col_A[y, x] = 1
            n_A += 1
            fA_y.append(y)
            fA_x.append(x)

    for i in range(len(sB_y)):
        y = sB_y[i]
        x = sB_x[i]
        if 0 <= y < rows and 0 <= x < cols and mask_B[y, x]:
            if owner[y, x] == 0:
                owner[y, x] = 2
                col_B[y, x] = 1
                n_B += 1
                fB_y.append(y)
                fB_x.append(x)
            elif owner[y, x] == 1:
                if np.random.rand() < 0.5:
                    owner[y, x] = 2
                    col_A[y, x] = -9999
                    col_B[y, x] = 1
                    n_A -= 1
                    n_B += 1
                    fB_y.append(y)
                    fB_x.append(x)

    offA_dy, offA_dx = _build_offsets(d_A)
    offB_dy, offB_dx = _build_offsets(d_B)
    num_offA = len(offA_dy)
    num_offB = len(offB_dy)

    progress_steps = []
    progress_A = []
    progress_B = []

    for step in range(2, max_steps + 2):
        if len(fA_y) == 0 and len(fB_y) == 0:
            break

        candA = np.zeros((rows, cols), dtype=np.bool_)
        candB = np.zeros((rows, cols), dtype=np.bool_)
        cA_y = List()
        cA_x = List()
        cB_y = List()
        cB_x = List()

        nfA_y = List()
        nfA_x = List()
        nfB_y = List()
        nfB_x = List()
        in_nextA = np.zeros((rows, cols), dtype=np.bool_)
        in_nextB = np.zeros((rows, cols), dtype=np.bool_)

        for idx in range(len(fA_y)):
            r = fA_y[idx]
            c = fA_x[idx]
            if owner[r, c] != 1:
                continue
            all_ok = True
            for k in range(num_offA):
                nr = r + offA_dy[k]
                nc = c + offA_dx[k]
                if 0 <= nr < rows and 0 <= nc < cols:
                    if mask_A[nr, nc] and owner[nr, nc] == 0 and not candA[nr, nc]:
                        if np.random.rand() <= prob_A:
                            candA[nr, nc] = True
                            cA_y.append(nr)
                            cA_x.append(nc)
                        else:
                            all_ok = False
            if not all_ok:
                if not in_nextA[r, c]:
                    in_nextA[r, c] = True
                    nfA_y.append(r)
                    nfA_x.append(c)

        for idx in range(len(fB_y)):
            r = fB_y[idx]
            c = fB_x[idx]
            if owner[r, c] != 2:
                continue
            all_ok = True
            for k in range(num_offB):
                nr = r + offB_dy[k]
                nc = c + offB_dx[k]
                if 0 <= nr < rows and 0 <= nc < cols:
                    if mask_B[nr, nc] and owner[nr, nc] == 0 and not candB[nr, nc]:
                        if np.random.rand() <= prob_B:
                            candB[nr, nc] = True
                            cB_y.append(nr)
                            cB_x.append(nc)
                        else:
                            all_ok = False
            if not all_ok:
                if not in_nextB[r, c]:
                    in_nextB[r, c] = True
                    nfB_y.append(r)
                    nfB_x.append(c)

        for i in range(len(cA_y)):
            y = cA_y[i]
            x = cA_x[i]
            a_wins = True
            if candB[y, x]:
                a_wins = np.random.rand() < 0.5
            if a_wins:
                owner[y, x] = 1
                col_A[y, x] = step
                n_A += 1
                if not in_nextA[y, x]:
                    in_nextA[y, x] = True
                    nfA_y.append(y)
                    nfA_x.append(x)
            else:
                owner[y, x] = 2
                col_B[y, x] = step
                n_B += 1
                if not in_nextB[y, x]:
                    in_nextB[y, x] = True
                    nfB_y.append(y)
                    nfB_x.append(x)

        for i in range(len(cB_y)):
            y = cB_y[i]
            x = cB_x[i]
            if candA[y, x]:
                continue
            owner[y, x] = 2
            col_B[y, x] = step
            n_B += 1
            if not in_nextB[y, x]:
                in_nextB[y, x] = True
                nfB_y.append(y)
                nfB_x.append(x)

        fA_y = nfA_y
        fA_x = nfA_x
        fB_y = nfB_y
        fB_x = nfB_x

        progress_steps.append(step)
        progress_A.append(n_A)
        progress_B.append(n_B)

        with objmode():
            _update_progress(step, n_A, n_B)

    return (col_A, col_B,
            np.array(progress_steps, dtype=np.int32),
            np.array(progress_A, dtype=np.int32),
            np.array(progress_B, dtype=np.int32))


def competition_sim(
    raster_path_A,
    csv_path_A,
    raster_path_B,
    csv_path_B,
    output_tif_path_A=None,
    output_tif_path_B=None,
    longitude_col="Longitude",
    latitude_col="Latitude",
    longitude_col_B=None,
    latitude_col_B=None,
    steps=None,
    prob=0.5,
    probability_A=None,
    probability_B=None,
    d=1,
    d_A=None,
    d_B=None,
    barrier_path=None,
    replicates=None,
    replicates_save=None,
    verbose=False,
    output_dir=None,
    generate_gif=True,
    output_gif_path=None,
    fps=15,
    width=1080,
    height=920,
    skip_steps=1,
    legend=True,
    colors={"occupied": (34, 139, 34), "available": (220, 220, 220), "nodata": (255, 255, 255), "barrier": (0, 0, 0)},
    **kwargs
):
    if replicates == 1:
        replicates = None

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    if output_tif_path_A is None:
        base_a = os.path.splitext(os.path.basename(raster_path_A))[0]
        out_d = output_dir if output_dir is not None else os.path.dirname(raster_path_A)
        output_tif_path_A = os.path.join(out_d, f"{base_a}_competition.tif")

    if output_tif_path_B is None:
        base_b = os.path.splitext(os.path.basename(raster_path_B))[0]
        out_d = output_dir if output_dir is not None else os.path.dirname(raster_path_B)
        output_tif_path_B = os.path.join(out_d, f"{base_b}_competition.tif")

    if output_gif_path is None:
        base_name_tif, _ = os.path.splitext(output_tif_path_A)
        output_gif_path = f"{base_name_tif}.gif"

    if not generate_gif:
        output_gif_path = None

    if probability_A is not None:
        prob_A = float(probability_A)
    else:
        if isinstance(prob, (list, tuple, np.ndarray)) and len(prob) >= 2:
            prob_A = float(prob[0])
        else:
            prob_A = float(prob)

    if probability_B is not None:
        prob_B = float(probability_B)
    else:
        if isinstance(prob, (list, tuple, np.ndarray)) and len(prob) >= 2:
            prob_B = float(prob[1])
        else:
            prob_B = float(prob_A)

    if d_A is not None:
        disp_A = int(d_A)
    else:
        if isinstance(d, (list, tuple, np.ndarray)) and len(d) >= 1:
            disp_A = int(d[0])
        else:
            disp_A = int(d)

    if d_B is not None:
        disp_B = int(d_B)
    else:
        if isinstance(d, (list, tuple, np.ndarray)) and len(d) >= 2:
            disp_B = int(d[1])
        else:
            disp_B = disp_A

    if longitude_col_B is None: longitude_col_B = longitude_col
    if latitude_col_B is None: latitude_col_B = latitude_col

    steps_max = 10**12 if steps is None else int(steps)
    num_reps = replicates if replicates is not None else 1
    is_multi_reps = (replicates is not None and replicates > 1)

    with rasterio.open(raster_path_A) as srcA_ref:
        boundsA = srcA_ref.bounds
        crsA = srcA_ref.crs
        resA = srcA_ref.res
        meta_ref = srcA_ref.meta.copy()

    with rasterio.open(raster_path_B) as srcB_ref:
        boundsB = srcB_ref.bounds
    left = min(boundsA.left, boundsB.left)
    bottom = min(boundsA.bottom, boundsB.bottom)
    right = max(boundsA.right, boundsB.right)
    top = max(boundsA.top, boundsB.top)

    res_x, res_y = resA[0], resA[1]
    unified_width = int(np.ceil((right - left) / res_x))
    unified_height = int(np.ceil((top - bottom) / res_y))
    common_transform = rasterio.transform.from_bounds(left, bottom, right, top, unified_width, unified_height)

    def warp_to_common(path):
        with rasterio.open(path) as src:
            raw = src.read(1)
            nd = src.nodata if src.nodata is not None else -9999
            out_arr = np.full((unified_height, unified_width), nd, dtype=np.float32)
            reproject(
                source=raw.astype(np.float32),
                destination=out_arr,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=common_transform,
                dst_crs=crsA,
                resampling=Resampling.nearest,
                src_nodata=nd,
                dst_nodata=nd
            )
            return out_arr

    rA_data = warp_to_common(raster_path_A)
    rB_data = warp_to_common(raster_path_B)

    meta_out = meta_ref.copy()
    meta_out.update({
        'height': unified_height,
        'width': unified_width,
        'transform': common_transform,
        'dtype': rasterio.int32,
        'nodata': -9999,
        'count': 1
    })

    barA = np.zeros((unified_height, unified_width), dtype=bool)
    if barrier_path is not None:
        if str(barrier_path).endswith(('.shp', '.geojson', '.gpkg')):
            gdf = gpd.read_file(barrier_path)
            if gdf.crs != crsA:
                gdf = gdf.to_crs(crsA)
            geoms = [mapping(geom) for geom in gdf.geometry]
            bar_warp = rasterize(
                [(geom, 1) for geom in geoms],
                out_shape=(unified_height, unified_width),
                transform=common_transform,
                fill=0,
                all_touched=True,
                dtype=np.uint8
            )
            raw_barrier = (bar_warp > 0)
            barA = maximum_filter(raw_barrier, size=3)
        elif str(barrier_path).endswith(('.tif', '.tiff', '.asc')):
            with rasterio.open(barrier_path) as b_src:
                bar_data = b_src.read(1)
                bar_warp = np.zeros((unified_height, unified_width), dtype=np.uint8)
                reproject(
                    source=(bar_data > 0).astype(np.uint8),
                    destination=bar_warp,
                    src_transform=b_src.transform,
                    src_crs=b_src.crs,
                    dst_transform=common_transform,
                    dst_crs=crsA,
                    resampling=Resampling.nearest,
                    src_nodata=b_src.nodata if b_src.nodata is not None else 0,
                    dst_nodata=0
                )
                raw_barrier = (bar_warp > 0) & (bar_warp != 0)
                barA = maximum_filter(raw_barrier, size=3)

    validA = (rA_data != -9999) & (~np.isnan(rA_data)) & (rA_data > 0)
    validB = (rB_data != -9999) & (~np.isnan(rB_data)) & (rB_data > 0)

    target_mask_A = validA & (~barA)
    target_mask_B = validB & (~barA)

    total_target_cells_A = int(np.sum(target_mask_A))
    total_target_cells_B = int(np.sum(target_mask_B))
    total_union_cells = int(np.sum(target_mask_A | target_mask_B))

    def read_seeds(csv_path, lon_col, lat_col, mask):
        occ = pd.read_csv(csv_path)
        points = set()
        for _, row in occ.iterrows():
            try:
                lon = row[lon_col]; lat = row[lat_col]
                if pd.isna(lon) or pd.isna(lat): continue
                r, c = rasterio.transform.rowcol(common_transform, lon, lat)
                if 0 <= r < unified_height and 0 <= c < unified_width:
                    if mask[r, c]: points.add((c, r))
            except Exception:
                pass
        sy = np.array([pt[1] for pt in points], dtype=np.int64)
        sx = np.array([pt[0] for pt in points], dtype=np.int64)
        return sy, sx

    def build_stats(arr, mask, total):
        valid = (arr != -9999) & mask
        colonized = int(np.sum(valid))
        max_st = int(np.max(arr[valid])) if np.any(valid) else 0
        not_reached = max(0, total - colonized)
        pct_not_reached = 100 - int((colonized / total) * 100) if total > 0 else 0
        return {
            "Steps": max_st,
            "Pixels": colonized,
            "Percentage": round((colonized / total) * 100, 2) if total > 0 else 100.0,
            "Not_reached": not_reached,
            "Percentage_not_reached": pct_not_reached
        }

    def save_species(out_path, reps_list, stats_list):
        base_dir = os.path.dirname(out_path)
        base_name = os.path.splitext(os.path.basename(out_path))[0]
        ext = os.path.splitext(out_path)[1]
        if base_dir: os.makedirs(base_dir, exist_ok=True)

        if is_multi_reps:
            stack = np.array(reps_list, dtype=np.float32)
            stack[stack == -9999] = np.nan
            with np.errstate(all='ignore'):
                arr_mean = np.nanmean(stack, axis=0)
                arr_sd = np.nanstd(stack, axis=0)
            arr_mean = np.nan_to_num(arr_mean, nan=-9999).astype(np.int32)
            arr_sd = np.nan_to_num(arr_sd, nan=-9999).astype(np.float32)

            with rasterio.open(os.path.join(base_dir, f"{base_name}_mean{ext}"), 'w', **meta_out) as dst:
                dst.write(arr_mean, 1)

            meta_sd = meta_out.copy()
            meta_sd.update({'dtype': rasterio.float32, 'nodata': -9999})
            with rasterio.open(os.path.join(base_dir, f"{base_name}_sd{ext}"), 'w', **meta_sd) as dst:
                dst.write(arr_sd, 1)

            df = pd.DataFrame(stats_list)
            mean_row = df.mean(numeric_only=True).to_dict()
            sd_row = df.std(numeric_only=True).to_dict()
            mean_row["Steps"] = "Mean"
            sd_row["Steps"] = "SD"
            df_out = pd.concat([df, pd.DataFrame([mean_row, sd_row])], ignore_index=True)
            df_out.to_csv(os.path.join(base_dir, f"{base_name}_stats.csv"), index=False)
            return arr_mean
        else:
            with rasterio.open(out_path, 'w', **meta_out) as dst:
                dst.write(reps_list[0], 1)
            return reps_list[0]

    seedA_y, seedA_x = read_seeds(csv_path_A, longitude_col, latitude_col, target_mask_A)
    seedB_y, seedB_x = read_seeds(csv_path_B, longitude_col_B, latitude_col_B, target_mask_B)

    tmA_reps_list = []
    tmB_reps_list = []
    stats_a = []
    stats_b = []
    print("Simulating Species A and B (competition)")
    for rep in range(num_reps):
        if is_multi_reps:
            sys.stdout.write(f"\r--- Running replicate {rep + 1} / {num_reps} ---")
            sys.stdout.flush()

        _PROGRESS.update({"total": total_union_cells, "rep": rep + 1, "num": num_reps, "multi": is_multi_reps, "t": 0.0})

        out_A, out_B, prog_steps, prog_A, prog_B = _run_competition_simulation_numba(
            target_mask_A.copy(), target_mask_B.copy(), seedA_y, seedA_x, seedB_y, seedB_x,
            steps_max, prob_A, prob_B, disp_A, disp_B
        )

        if len(prog_steps) > 0:
            _render_progress(int(prog_steps[-1]), int(prog_A[-1]), int(prog_B[-1]))

        if not is_multi_reps:
            print()

        tmA_reps_list.append(out_A)
        tmB_reps_list.append(out_B)
        stats_a.append(build_stats(out_A, target_mask_A, total_target_cells_A))
        stats_b.append(build_stats(out_B, target_mask_B, total_target_cells_B))

        if replicates_save is not None and is_multi_reps:
            for out_path, arr in ((output_tif_path_A, out_A), (output_tif_path_B, out_B)):
                base_dir = os.path.dirname(out_path)
                base_name = os.path.splitext(os.path.basename(out_path))[0]
                ext = os.path.splitext(out_path)[1]
                with rasterio.open(os.path.join(base_dir, f"{base_name}_{rep + 1}{ext}"), 'w', **meta_out) as dst:
                    dst.write(arr, 1)

    if is_multi_reps:
        print()

    tmA_last = tmA_reps_list[-1]
    tmB_last = tmB_reps_list[-1]

    tmA_arr = save_species(output_tif_path_A, tmA_reps_list, stats_a)
    tmB_arr = save_species(output_tif_path_B, tmB_reps_list, stats_b)

    if not is_multi_reps:
        unreached_A = max(0, total_target_cells_A - int(np.sum((tmA_arr != -9999) & target_mask_A)))
        unreached_B = max(0, total_target_cells_B - int(np.sum((tmB_arr != -9999) & target_mask_B)))
        pct_A = 100 - int(((total_target_cells_A - unreached_A) / total_target_cells_A) * 100) if total_target_cells_A > 0 else 0
        pct_B = 100 - int(((total_target_cells_B - unreached_B) / total_target_cells_B) * 100) if total_target_cells_B > 0 else 0
        print(f"Pixels do not reached A: {unreached_A} | {pct_A}%")
        print(f"Pixels do not reached B: {unreached_B} | {pct_B}%")

    out_dir = output_dir if output_dir is not None else '.'
    os.makedirs(out_dir, exist_ok=True)

    if output_gif_path is not None:
        print("Generating GIF simulation")

        valid_habitat_a = (rA_data > 0) & (rA_data != -9999) & (~np.isnan(rA_data))
        valid_habitat_b = (rB_data > 0) & (rB_data != -9999) & (~np.isnan(rB_data))

        if barrier_path is not None:
            barA_gif = barA & (valid_habitat_a | valid_habitat_b)
        else:
            barA_gif = np.zeros((unified_height, unified_width), dtype=bool)

        base_frame = np.full((unified_height, unified_width, 3), colors["nodata"], dtype=np.uint8)

        only_a = valid_habitat_a & ~valid_habitat_b
        only_b = ~valid_habitat_a & valid_habitat_b
        both = valid_habitat_a & valid_habitat_b

        gif_colors = {
            "Suitable for A": (230, 245, 230),
            "Suitable for B": (230, 240, 255),
            "Suitable A ∩ B": (240, 230, 250),
            "Occupied A": (34, 139, 34),
            "Occupied B": (30, 144, 255),
            "barrier": colors["barrier"]
        }

        base_frame[only_a] = gif_colors["Suitable for A"]
        base_frame[only_b] = gif_colors["Suitable for B"]
        base_frame[both] = gif_colors["Suitable A ∩ B"]

        if barrier_path is not None:
            base_frame[barA_gif] = gif_colors["barrier"]

        step_arrays = []
        if np.any(tmA_last > 0):
            step_arrays.append(tmA_last[tmA_last > 0])
        if np.any(tmB_last > 0):
            step_arrays.append(tmB_last[tmB_last > 0])

        if len(step_arrays) > 0:
            max_gif_step = int(np.max(np.concatenate(step_arrays)))
        else:
            max_gif_step = 1

        steps_range = list(range(1, max_gif_step + 1))

        title_font_size = max(14, int(width * 0.025))
        legend_font_size = max(10, int(width * 0.013))
        font_title = font_legend = None
        for fname in ("arial.ttf", "DejaVuSans.ttf"):
            try:
                font_title = ImageFont.truetype(fname, title_font_size)
                font_legend = ImageFont.truetype(fname, legend_font_size)
                break
            except Exception:
                continue
        if font_title is None:
            font_title = ImageFont.load_default()
            font_legend = ImageFont.load_default()

        title_area_h = int(height * 0.12)
        map_max_w = int(width * 0.92)
        map_max_h = height - title_area_h - int(height * 0.04)

        scale = min(map_max_w / unified_width, map_max_h / unified_height)
        new_w, new_h = max(1, int(unified_width * scale)), max(1, int(unified_height * scale))

        paste_x = (width - new_w) // 2
        paste_y = title_area_h + (map_max_h - new_h) // 2

        base_canvas = Image.new("RGB", (width, height), (255, 255, 255))

        items = []
        box_x = box_y = box_w = box_h = 0
        swatch_size = row_h = box_pad = 0
        if legend:
            items = [
                (gif_colors["Suitable for A"], "Suitable for A"),
                (gif_colors["Suitable for B"], "Suitable for B"),
                (gif_colors["Suitable A ∩ B"], "Suitable A ∩ B"),
                (gif_colors["Occupied A"], "Occupied A"),
                (gif_colors["Occupied B"], "Occupied B"),
            ]
            if barrier_path is not None:
                items.append((gif_colors["barrier"], "Barrier"))

            swatch_size = max(10, int(width * 0.015))
            box_pad = max(8, int(width * 0.01))
            row_h = swatch_size + max(8, int(width * 0.008))

            dummy_draw = ImageDraw.Draw(base_canvas)
            max_text_w = 0
            for _, label in items:
                try:
                    t_w = dummy_draw.textlength(label, font=font_legend)
                except AttributeError:
                    t_w = font_legend.getsize(label)[0]
                if t_w > max_text_w:
                    max_text_w = t_w

            text_spacing = int(width * 0.012)
            box_w = int(box_pad * 2 + swatch_size + text_spacing + max_text_w)
            box_h = box_pad * 2 + len(items) * row_h

            box_x = 30
            box_y = height - box_h - 30

        total_gif_frames = len(steps_range)

        ffmpeg_cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}', '-pix_fmt', 'rgb24', '-r', str(fps),
            '-i', '-', '-filter_complex', '[0:v] split [a][b];[a] palettegen=stats_mode=full [p];[b][p] paletteuse=dither=sierra2_4a',
            output_gif_path
        ]

        pipe = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

        for idx_g, step in enumerate(steps_range):
            pct_gif = int(((idx_g + 1) / total_gif_frames) * 100)
            bar_width = 30
            blocks = int(bar_width * pct_gif / 100)
            bar = "=" * blocks + " " * (bar_width - blocks)
            sys.stdout.write(f"\r[{bar}] {pct_gif}% (GIF Frame {idx_g + 1}/{total_gif_frames})")
            sys.stdout.flush()

            frame = base_frame.copy()

            frame[(tmA_last != -9999) & (tmA_last <= step)] = gif_colors["Occupied A"]
            frame[(tmB_last != -9999) & (tmB_last <= step)] = gif_colors["Occupied B"]

            if barrier_path is not None:
                frame[barA_gif] = gif_colors["barrier"]

            img = Image.fromarray(frame)
            img_resized = img.resize((new_w, new_h), Image.Resampling.NEAREST)

            canvas = base_canvas.copy()
            canvas.paste(img_resized, (paste_x, paste_y))

            draw = ImageDraw.Draw(canvas)
            draw.text((width // 2, title_area_h // 2), f"Simulation Step: {int(step)}", fill=(0, 0, 0), anchor="mm", font=font_title)

            if legend:
                draw.rectangle([box_x, box_y, box_x + box_w, box_y + box_h], fill=(255, 255, 255), outline=(0, 0, 0), width=max(1, int(width * 0.001)))
                for i, (col, label) in enumerate(items):
                    current_y = box_y + box_pad + i * row_h
                    draw.rectangle([box_x + box_pad, current_y + 2, box_x + box_pad + swatch_size, current_y + 2 + swatch_size], fill=col, outline=(0, 0, 0))
                    draw.text((box_x + box_pad + swatch_size + int(width * 0.012), current_y + swatch_size // 2 + 2), label, fill=(0, 0, 0), anchor="lm", font=font_legend)

            pipe.stdin.write(canvas.tobytes())

        pipe.stdin.close()
        pipe.wait()

        print()
        print("GIF generated")