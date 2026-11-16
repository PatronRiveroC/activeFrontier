import os
import sys
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import mapping
from scipy.ndimage import maximum_filter
from numba import jit
from numba.typed import List
from PIL import Image, ImageDraw, ImageFont
import subprocess
import time
from numba import objmode

try:
    from single_sim import single_sim
except ImportError:
    pass

_PROGRESS = {"total": 0, "rep": 1, "num": 1, "multi": False, "t": 0.0}


def _render_progress(step, pixels):
    total = _PROGRESS["total"]
    pct = min(100, int((pixels / total) * 100)) if total > 0 else 100
    bar_width = 30
    blocks = int(bar_width * pct / 100)
    bar = "=" * blocks + " " * (bar_width - blocks)
    if _PROGRESS["multi"]:
        sys.stdout.write(f"\r[{bar}] {pct}% (Rep {_PROGRESS['rep']}/{_PROGRESS['num']})")
    else:
        sys.stdout.write(f"\r[{bar}] {pct}% (Step: {step} | Pixels A: {pixels})")
    sys.stdout.flush()


def _update_progress(step, pixels):
    now = time.time()
    if now - _PROGRESS["t"] < 0.1:
        return
    _PROGRESS["t"] = now
    _render_progress(step, pixels)


@jit(nopython=True, cache=True)
def _push_event(ev, n, step, y, x, t):
    if n >= ev.shape[0]:
        new_ev = np.empty((ev.shape[0] * 2, 4), dtype=np.int32)
        for i in range(n):
            for j in range(4):
                new_ev[i, j] = ev[i, j]
        ev = new_ev
    ev[n, 0] = step
    ev[n, 1] = y
    ev[n, 2] = x
    ev[n, 3] = t
    return ev, n + 1


@jit(nopython=True, cache=True)
def _run_amensalism_simulation_numba(habitat_mask, zone_type, tmB, start_y, start_x, max_steps, prob, d, extirpated_time):
    rows, cols = habitat_mask.shape
    colonization_time = np.full((rows, cols), -9999, dtype=np.int32)
    coexist_time = np.full((rows, cols), -9999, dtype=np.int32)
    extirpation_time = np.full((rows, cols), -9999, dtype=np.int32)

    visited = np.zeros((rows, cols), dtype=np.bool_)
    extirpated = np.zeros((rows, cols), dtype=np.bool_)

    ev = np.empty((1024, 4), dtype=np.int32)
    n_ev = 0
    n_visited = 0

    frontier_y = List()
    frontier_x = List()
    watch_y = List()
    watch_x = List()

    for i in range(len(start_y)):
        y = start_y[i]
        x = start_x[i]
        if 0 <= y < rows and 0 <= x < cols and habitat_mask[y, x] and not visited[y, x]:
            visited[y, x] = True
            n_visited += 1
            colonization_time[y, x] = 1
            if zone_type[y, x] == 2:
                tbv = tmB[y, x]
                if tbv != -9999 and tbv <= 1:
                    coexist_time[y, x] = 1
                    ev, n_ev = _push_event(ev, n_ev, 1, y, x, 2)
                else:
                    ev, n_ev = _push_event(ev, n_ev, 1, y, x, 1)
                watch_y.append(y)
                watch_x.append(x)
            else:
                ev, n_ev = _push_event(ev, n_ev, 1, y, x, 1)
            frontier_y.append(y)
            frontier_x.append(x)

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

    offsets_dy_arr = np.array(offsets_dy, dtype=np.int32)
    offsets_dx_arr = np.array(offsets_dx, dtype=np.int32)
    num_offsets = len(offsets_dy_arr)

    progress_steps = []
    progress_colonized = []

    for step in range(2, max_steps + 2):
        if len(frontier_y) == 0 and len(watch_y) == 0:
            break

        next_frontier_y = List()
        next_frontier_x = List()
        in_next = np.zeros((rows, cols), dtype=np.bool_)
        next_watch_y = List()
        next_watch_x = List()

        for idx in range(len(watch_y)):
            r = watch_y[idx]
            c = watch_x[idx]
            if coexist_time[r, c] == -9999:
                tbv = tmB[r, c]
                if tbv != -9999 and tbv <= step:
                    coexist_time[r, c] = step
                    ev, n_ev = _push_event(ev, n_ev, step, r, c, 2)
                next_watch_y.append(r)
                next_watch_x.append(c)
            elif step - coexist_time[r, c] >= extirpated_time:
                extirpated[r, c] = True
                extirpation_time[r, c] = step
                ev, n_ev = _push_event(ev, n_ev, step, r, c, 3)
            else:
                next_watch_y.append(r)
                next_watch_x.append(c)

        for idx in range(len(frontier_y)):
            r = frontier_y[idx]
            c = frontier_x[idx]

            if extirpated[r, c]:
                continue

            all_ok = True
            for k in range(num_offsets):
                nr = r + offsets_dy_arr[k]
                nc = c + offsets_dx_arr[k]
                if 0 <= nr < rows and 0 <= nc < cols:
                    if habitat_mask[nr, nc] and not visited[nr, nc]:
                        if np.random.rand() <= prob:
                            visited[nr, nc] = True
                            n_visited += 1
                            colonization_time[nr, nc] = step
                            if zone_type[nr, nc] == 2:
                                tbv = tmB[nr, nc]
                                if tbv != -9999 and tbv <= step:
                                    coexist_time[nr, nc] = step
                                    ev, n_ev = _push_event(ev, n_ev, step, nr, nc, 2)
                                else:
                                    ev, n_ev = _push_event(ev, n_ev, step, nr, nc, 1)
                                next_watch_y.append(nr)
                                next_watch_x.append(nc)
                            else:
                                ev, n_ev = _push_event(ev, n_ev, step, nr, nc, 1)
                            if not in_next[nr, nc]:
                                in_next[nr, nc] = True
                                next_frontier_y.append(nr)
                                next_frontier_x.append(nc)
                        else:
                            all_ok = False
            if not all_ok:
                if not in_next[r, c]:
                    in_next[r, c] = True
                    next_frontier_y.append(r)
                    next_frontier_x.append(c)

        frontier_y = next_frontier_y
        frontier_x = next_frontier_x
        watch_y = next_watch_y
        watch_x = next_watch_x

        progress_steps.append(step)
        progress_colonized.append(n_visited)

        with objmode():
            _update_progress(step, n_visited)

    return (colonization_time, extirpation_time,
            np.array(progress_steps, dtype=np.int32),
            np.array(progress_colonized, dtype=np.int32),
            ev[:n_ev].copy())


def amensalism_sim(
    raster_path_A,
    csv_path_A,
    raster_path_B=None,
    csv_path_B=None,
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
    extirpated_time=1,
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

    if "extirped_time" in kwargs and kwargs["extirped_time"] is not None:
        extirpated_time = kwargs["extirped_time"]
    extirpated_time = max(1, int(extirpated_time))

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    if output_tif_path_A is None:
        base_a = os.path.splitext(os.path.basename(raster_path_A))[0]
        out_d = output_dir if output_dir is not None else os.path.dirname(raster_path_A)
        output_tif_path_A = os.path.join(out_d, f"{base_a}_amensalism.tif")

    if raster_path_B is not None and output_tif_path_B is None:
        base_b = os.path.splitext(os.path.basename(raster_path_B))[0]
        out_d = output_dir if output_dir is not None else os.path.dirname(raster_path_B)
        output_tif_path_B = os.path.join(out_d, f"{base_b}_amensalism.tif")

    if output_gif_path is None:
        if output_tif_path_A is not None:
            base_name_tif, _ = os.path.splitext(output_tif_path_A)
            output_gif_path = f"{base_name_tif}.gif"
        else:
            output_gif_path = "amensalism.gif"

    if not generate_gif:
        output_gif_path = None

    if probability_A is not None:
        prob_A = float(probability_A)
    else:
        if isinstance(prob, (list, tuple, np.ndarray)) and len(prob) >= 2:
            prob_A = float(prob[0])
        else:
            prob_A = float(prob)

    if raster_path_B is not None and csv_path_B is not None:
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

    if raster_path_B is not None:
        with rasterio.open(raster_path_B) as srcB_ref:
            boundsB = srcB_ref.bounds
        left = min(boundsA.left, boundsB.left)
        bottom = min(boundsA.bottom, boundsB.bottom)
        right = max(boundsA.right, boundsB.right)
        top = max(boundsA.top, boundsB.top)
    else:
        boundsB = boundsA
        left, bottom, right, top = boundsA.left, boundsA.bottom, boundsA.right, boundsA.top

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
    rB_data = warp_to_common(raster_path_B) if raster_path_B is not None else rA_data.copy()

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

    if raster_path_B is not None and csv_path_B is not None:
        print("Simulating species B")
        temp_tif_B = output_tif_path_B

        single_sim(
            raster_path=raster_path_B,
            csv_path=csv_path_B,
            output_tif_path=temp_tif_B,
            longitude_col=longitude_col_B,
            latitude_col=latitude_col_B,
            steps=steps,
            prob=prob_B,
            d=disp_B,
            barrier_path=barrier_path,
            replicates=replicates,
            replicates_save=replicates_save,
            generate_gif=False,
            output_gif_path=None
        )

        base_b, ext_b = os.path.splitext(temp_tif_B)
        actual_tif_B = f"{base_b}_mean{ext_b}" if is_multi_reps else temp_tif_B

        tmB = warp_to_common(actual_tif_B)

        if output_tif_path_B is not None:
            with rasterio.open(actual_tif_B, 'w', **meta_out) as dst:
                dst.write(np.where(np.isnan(tmB), -9999, tmB).astype(np.int32), 1)

        tmB = np.where(np.isnan(tmB), -9999, tmB).astype(np.int32)
    else:
        tmB = np.where(validB & (~barA), 0, -9999).astype(np.int32)

    zone_type = np.zeros((unified_height, unified_width), dtype=np.int32)
    in_b_mask = (tmB != -9999)
    zone_type[target_mask_A & ~in_b_mask] = 1
    zone_type[target_mask_A & in_b_mask] = 2

    tmA_reps_list = []
    exA_reps_list = []
    stats_a = []
    events_last = np.zeros((0, 4), dtype=np.int32)
    total_target_cells_A = int(np.sum(target_mask_A))
    print("Simulating Species A (amensalism)")
    for rep in range(num_reps):
        if is_multi_reps:
            sys.stdout.write(f"\r--- Running replicate {rep + 1} / {num_reps} ---")
            sys.stdout.flush()

        occA = pd.read_csv(csv_path_A)
        initial_points = set()
        for _, row in occA.iterrows():
            try:
                lon = row[longitude_col]; lat = row[latitude_col]
                if pd.isna(lon) or pd.isna(lat): continue
                r, c = rasterio.transform.rowcol(common_transform, lon, lat)
                if 0 <= r < unified_height and 0 <= c < unified_width:
                    if target_mask_A[r, c]: initial_points.add((c, r))
            except Exception:
                pass

        start_y = np.array([pt[1] for pt in initial_points], dtype=np.int64)
        start_x = np.array([pt[0] for pt in initial_points], dtype=np.int64)

        habitat_mask_A = target_mask_A.copy()

        _PROGRESS.update({"total": total_target_cells_A, "rep": rep + 1, "num": num_reps, "multi": is_multi_reps, "t": 0.0})

        out_data, out_extirp, prog_steps, prog_colonized, events_last = _run_amensalism_simulation_numba(
            habitat_mask_A, zone_type, tmB, start_y, start_x, steps_max, prob_A, disp_A, extirpated_time
        )

        if len(prog_steps) > 0:
            _render_progress(int(prog_steps[-1]), int(prog_colonized[-1]))

        if not is_multi_reps:
            print()

        tmA_arr = out_data
        tmA_reps_list.append(tmA_arr)
        exA_reps_list.append(out_extirp)

        valid_colonized = (tmA_arr != -9999) & target_mask_A
        valid_extirpated = (out_extirp != -9999) & target_mask_A
        colonized_a = int(np.sum(valid_colonized))
        extirpated_a = int(np.sum(valid_extirpated))
        max_st_a = int(np.max(tmA_arr[valid_colonized])) if np.any(valid_colonized) else 0
        not_reached_a = max(0, total_target_cells_A - colonized_a)
        pct_not_reached_a = 100 - int((colonized_a / total_target_cells_A) * 100) if total_target_cells_A > 0 else 0

        stats_a.append({
            "Steps": max_st_a,
            "Pixels": colonized_a,
            "Percentage": round((colonized_a / total_target_cells_A) * 100, 2) if total_target_cells_A > 0 else 100.0,
            "Not_reached": not_reached_a,
            "Percentage_not_reached": pct_not_reached_a,
            "Extirpated": extirpated_a,
            "Percentage_extirpated": round((extirpated_a / total_target_cells_A) * 100, 2) if total_target_cells_A > 0 else 0.0
        })

        if replicates_save is not None and output_tif_path_A is not None and is_multi_reps:
            base_dir_a = os.path.dirname(output_tif_path_A)
            base_name_a = os.path.splitext(os.path.basename(output_tif_path_A))[0]
            ext_a = os.path.splitext(output_tif_path_A)[1]
            rep_path_a = os.path.join(base_dir_a, f"{base_name_a}_{rep + 1}{ext_a}")
            rep_ex_path_a = os.path.join(base_dir_a, f"{base_name_a}_extirpated_{rep + 1}{ext_a}")
            with rasterio.open(rep_path_a, 'w', **meta_out) as dst:
                dst.write(tmA_arr, 1)
            with rasterio.open(rep_ex_path_a, 'w', **meta_out) as dst:
                dst.write(out_extirp, 1)

    if is_multi_reps:
        print()
        stackA = np.array(tmA_reps_list, dtype=np.float32)
        stackA[stackA == -9999] = np.nan
        stackE = np.array(exA_reps_list, dtype=np.float32)
        stackE[stackE == -9999] = np.nan
        with np.errstate(all='ignore'):
            tmA_mean = np.nanmean(stackA, axis=0) if stackA.size > 0 else np.full((unified_height, unified_width), -9999, dtype=np.float32)
            tmA_sd = np.nanstd(stackA, axis=0) if stackA.size > 0 else np.zeros((unified_height, unified_width), dtype=np.float32)
            exA_mean = np.nanmean(stackE, axis=0) if stackE.size > 0 else np.full((unified_height, unified_width), -9999, dtype=np.float32)
        tmA_mean = np.nan_to_num(tmA_mean, nan=-9999).astype(np.int32)
        tmA_sd = np.nan_to_num(tmA_sd, nan=-9999).astype(np.float32)
        exA_arr = np.nan_to_num(np.rint(exA_mean), nan=-9999).astype(np.int32)
        tmA_arr = tmA_mean

        if output_tif_path_A is not None:
            base_dir_a = os.path.dirname(output_tif_path_A)
            base_name_a = os.path.splitext(os.path.basename(output_tif_path_A))[0]
            ext_a = os.path.splitext(output_tif_path_A)[1]
            if base_dir_a: os.makedirs(base_dir_a, exist_ok=True)

            mean_path_a = os.path.join(base_dir_a, f"{base_name_a}_mean{ext_a}")
            sd_path_a = os.path.join(base_dir_a, f"{base_name_a}_sd{ext_a}")
            ex_mean_path_a = os.path.join(base_dir_a, f"{base_name_a}_extirpated_mean{ext_a}")
            csv_path_a = os.path.join(base_dir_a, f"{base_name_a}_stats.csv")

            meta_mean_a = meta_out.copy()
            with rasterio.open(mean_path_a, 'w', **meta_mean_a) as dst:
                dst.write(tmA_mean, 1)

            with rasterio.open(ex_mean_path_a, 'w', **meta_mean_a) as dst:
                dst.write(exA_arr, 1)

            meta_sd_a = meta_out.copy()
            meta_sd_a.update({'dtype': rasterio.float32, 'nodata': -9999})
            with rasterio.open(sd_path_a, 'w', **meta_sd_a) as dst:
                dst.write(tmA_sd, 1)

            df_a = pd.DataFrame(stats_a)
            mean_row_a = df_a.mean(numeric_only=True).to_dict()
            sd_row_a = df_a.std(numeric_only=True).to_dict()
            mean_row_a["Steps"] = "Mean"
            sd_row_a["Steps"] = "SD"

            df_a_out = pd.concat([df_a, pd.DataFrame([mean_row_a, sd_row_a])], ignore_index=True)
            df_a_out.to_csv(csv_path_a, index=False)
    else:
        tmA_arr = tmA_reps_list[0]
        exA_arr = exA_reps_list[0]
        if output_tif_path_A is not None:
            base_name_a, ext_a = os.path.splitext(output_tif_path_A)
            with rasterio.open(output_tif_path_A, 'w', **meta_out) as dstA:
                dstA.write(tmA_arr, 1)
            with rasterio.open(f"{base_name_a}_extirpated{ext_a}", 'w', **meta_out) as dstE:
                dstE.write(exA_arr, 1)

    if not is_multi_reps:
        total_colonized_A = np.sum((tmA_arr != -9999) & target_mask_A)
        total_unreached_A = max(0, total_target_cells_A - total_colonized_A)
        unreached_pct_A = 100 - int((total_colonized_A / total_target_cells_A) * 100) if total_target_cells_A > 0 else 0
        print(f"Pixels do not reached: {total_unreached_A} | {unreached_pct_A}%")

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
            "Occupied B": (30, 144, 255),
            "Occupied A": (34, 139, 34),
            "Established A + B": (128, 0, 128),
            "Occupied B (by extirpated A)": (158, 202, 225),
            "barrier": colors["barrier"]
        }

        base_frame[only_a] = gif_colors["Suitable for A"]
        base_frame[only_b] = gif_colors["Suitable for B"]
        base_frame[both] = gif_colors["Suitable A ∩ B"]

        if barrier_path is not None:
            base_frame[barA_gif] = gif_colors["barrier"]

        step_arrays = []
        if len(events_last) > 0:
            step_arrays.append(events_last[:, 0])

        if raster_path_B is not None and csv_path_B is not None:
            valid_b = (tmB != -9999)
            if np.any(valid_b & (tmB > 0)):
                step_arrays.append(tmB[valid_b & (tmB > 0)])

        if len(step_arrays) > 0:
            unique_colonization_steps = np.unique(np.concatenate(step_arrays))
        else:
            unique_colonization_steps = np.array([1])

        steps_range = list(range(1, int(np.max(unique_colonization_steps)) + 1))

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
            ]
            if raster_path_B is not None and csv_path_B is not None:
                items.append((gif_colors["Occupied B"], "Occupied B"))
            items += [
                (gif_colors["Occupied A"], "Occupied A"),
                (gif_colors["Established A + B"], "Established A + B"),
                (gif_colors["Occupied B (by extirpated A)"], "Occupied B (by extirpated A)"),
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

        ev_steps = events_last[:, 0]
        ev_y = events_last[:, 1]
        ev_x = events_last[:, 2]
        lut = np.array([0, 1, 2, 3], dtype=np.uint8)
        ev_state = lut[events_last[:, 3]]
        state = np.zeros((unified_height, unified_width), dtype=np.uint8)
        ptr = 0

        for idx_g, step in enumerate(steps_range):
            pct_gif = int(((idx_g + 1) / total_gif_frames) * 100)
            bar_width = 30
            blocks = int(bar_width * pct_gif / 100)
            bar = "=" * blocks + " " * (bar_width - blocks)
            sys.stdout.write(f"\r[{bar}] {pct_gif}% (GIF Frame {idx_g + 1}/{total_gif_frames})")
            sys.stdout.flush()

            new_ptr = int(np.searchsorted(ev_steps, step, side='right'))
            if new_ptr > ptr:
                state[ev_y[ptr:new_ptr], ev_x[ptr:new_ptr]] = ev_state[ptr:new_ptr]
                ptr = new_ptr

            frame = base_frame.copy()

            if raster_path_B is not None and csv_path_B is not None:
                colonized_b_mask = (tmB <= step) & (tmB != -9999)
                frame[colonized_b_mask] = gif_colors["Occupied B"]

            frame[state == 1] = gif_colors["Occupied A"]
            frame[state == 2] = gif_colors["Established A + B"]
            frame[state == 3] = gif_colors["Occupied B (by extirpated A)"]

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