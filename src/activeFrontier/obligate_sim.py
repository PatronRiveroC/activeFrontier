import os
import sys
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import mapping
from scipy.ndimage import maximum_filter
from numba import jit
from PIL import Image, ImageDraw, ImageFont
import subprocess

try:
    from single_sim import single_sim
except ImportError:
    pass

@jit(nopython=True, cache=True)
def _run_obligate_simulation_numba(habitat_mask, tmB, start_y, start_x, max_steps, prob, d):
    rows, cols = habitat_mask.shape
    colonization_time = np.full((rows, cols), -9999, dtype=np.int32)
    
    frontier = np.zeros((rows, cols), dtype=np.bool_)
    colonized = np.zeros((rows, cols), dtype=np.bool_)
    seed_mask = np.zeros((rows, cols), dtype=np.bool_)
    
    for i in range(len(start_y)):
        y = start_y[i]
        x = start_x[i]
        if 0 <= y < rows and 0 <= x < cols and habitat_mask[y, x]:
            colonization_time[y, x] = 1
            colonized[y, x] = True
            frontier[y, x] = True
            seed_mask[y, x] = True
            
    int_d = int(np.ceil(d))
    offsets_dy = []
    offsets_dx = []
    for dy in range(-int_d, int_d + 1):
        for dx in range(-int_d, int_d + 1):
            if dx == 0 and dy == 0:
                continue
            if np.sqrt(dx*dx + dy*dy) <= d:
                offsets_dy.append(dy)
                offsets_dx.append(dx)
                
    offsets_dy_arr = np.array(offsets_dy, dtype=np.int32)
    offsets_dx_arr = np.array(offsets_dx, dtype=np.int32)
    num_offsets = len(offsets_dy_arr)
    
    progress_steps = []
    progress_colonized = []

    for step in range(2, max_steps + 2):
        has_frontier = False
        for r in range(rows):
            for c in range(cols):
                if frontier[r, c]:
                    has_frontier = True
                    break
            if has_frontier:
                break
        if not has_frontier:
            break
            
        next_frontier = np.zeros((rows, cols), dtype=np.bool_)
        
        for r in range(rows):
            for c in range(cols):
                if frontier[r, c]:
                    if not colonized[r, c]:
                        established_neighbors = 0
                        for k in range(num_offsets):
                            nr = r + offsets_dy_arr[k]
                            nc = c + offsets_dx_arr[k]
                            if 0 <= nr < rows and 0 <= nc < cols:
                                if colonized[nr, nc]:
                                    established_neighbors += 1
                        
                        if established_neighbors >= (num_offsets // 2):
                            colonized[r, c] = True
                            colonization_time[r, c] = step
                            next_frontier[r, c] = True
                        continue

                    if not seed_mask[r, c]:
                        established_neighbors = 0
                        for k in range(num_offsets):
                            nr = r + offsets_dy_arr[k]
                            nc = c + offsets_dx_arr[k]
                            if 0 <= nr < rows and 0 <= nc < cols:
                                if colonized[nr, nc]:
                                    established_neighbors += 1

                        if established_neighbors < num_offsets:
                            extinction_probability = 0.3 * (1.0 - (established_neighbors / num_offsets)) if num_offsets > 0 else 0.0
                            if np.random.rand() <= extinction_probability:
                                colonized[r, c] = False
                                colonization_time[r, c] = -9999
                                next_frontier[r, c] = True
                                continue

                    has_any_valid = False
                    has_any_future_b = False
                    ready_indices_dy = np.empty(num_offsets, dtype=np.int32)
                    ready_indices_dx = np.empty(num_offsets, dtype=np.int32)
                    ready_num = 0
                    
                    for k in range(num_offsets):
                        nr = r + offsets_dy_arr[k]
                        nc = c + offsets_dx_arr[k]
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if habitat_mask[nr, nc] and not colonized[nr, nc]:
                                has_any_valid = True
                                tbv = tmB[nr, nc]
                                if tbv != -9999:
                                    has_any_future_b = True
                                    if tbv <= step:
                                        ready_indices_dy[ready_num] = offsets_dy_arr[k]
                                        ready_indices_dx[ready_num] = offsets_dx_arr[k]
                                        ready_num += 1
                    
                    if not has_any_valid or not has_any_future_b:
                        continue
                        
                    if ready_num == 0:
                        next_frontier[r, c] = True
                        continue
                        
                    retire_a = True
                    for i in range(ready_num):
                        nr = r + ready_indices_dy[i]
                        nc = c + ready_indices_dx[i]
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if habitat_mask[nr, nc] and not colonized[nr, nc]:
                                if np.random.rand() <= prob:
                                    colonized[nr, nc] = True
                                    colonization_time[nr, nc] = step
                                    next_frontier[nr, nc] = True
                                else:
                                    retire_a = False
                                    
                    if not retire_a:
                        next_frontier[r, c] = True

        frontier = next_frontier
        
        current_colonized = np.sum(colonized & habitat_mask)
        progress_steps.append(step)
        progress_colonized.append(current_colonized)
        
    return colonization_time, np.array(progress_steps, dtype=np.int32), np.array(progress_colonized, dtype=np.int32)

def obligate_sim(
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
        output_tif_path_A = os.path.join(out_d, f"{base_a}_obligate.tif" if not out_d else f"{output_dir}/{base_a}_obligate.tif")

    if raster_path_B is not None and output_tif_path_B is None:
        base_b = os.path.splitext(os.path.basename(raster_path_B))[0]
        out_d = output_dir if output_dir is not None else os.path.dirname(raster_path_B)
        output_tif_path_B = os.path.join(out_d, f"{base_b}_obligate.tif" if not out_d else f"{output_dir}/{base_b}_obligate.tif")

    if output_gif_path is None:
        if output_tif_path_A is not None:
            base_name_tif, _ = os.path.splitext(output_tif_path_A)
            output_gif_path = f"{base_name_tif}.gif"
        else:
            output_gif_path = "obligate.gif"

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

    meta_out = srcA_ref.meta.copy()
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
                dst.write(tmB.astype(np.int32), 1)

        B_ever = (tmB != -9999) & (~np.isnan(tmB))
        target_mask_A = validA & B_ever & (~barA)
    else:
        target_mask_A = validA & validB & (~barA)
        B_ever = np.full((unified_height, unified_width), True, dtype=bool)
        tmB = np.where(target_mask_A, 0, -9999).astype(np.int32)

    tmA_reps_list = []
    stats_a = []
    print("Simulating species A (A ∩ B)")
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
            except: pass

        start_y = np.array([pt[1] for pt in initial_points], dtype=np.int64)
        start_x = np.array([pt[0] for pt in initial_points], dtype=np.int64)

        total_target_cells_A = int(np.sum(target_mask_A))
        habitat_mask_A = target_mask_A.copy()

        out_data, prog_steps, prog_colonized = _run_obligate_simulation_numba(
            habitat_mask_A, tmB, start_y, start_x, steps_max, prob_A, disp_A
        )

        total_p_steps = len(prog_steps)
        for idx_p in range(total_p_steps):
            c_step = prog_steps[idx_p]
            c_pixels = prog_colonized[idx_p]
            percentage = int((c_pixels / total_target_cells_A) * 100) if total_target_cells_A > 0 else 100
            is_last = (idx_p == total_p_steps - 1)
            
            if c_step % 250 == 0 or is_last or percentage >= 100:
                bar_width = 30
                blocks = int(bar_width * percentage / 100)
                bar = "=" * blocks + " " * (bar_width - blocks)
                
                if is_multi_reps:
                    sys.stdout.write(f"\r[{bar}] {percentage}% (Rep {rep + 1}/{num_reps})")
                else:
                    end_char = "\n" if is_last else "\r"
                    print(f"[{bar}] {percentage}% (Step: {c_step} | Pixels A: {c_pixels})", end=end_char)
                sys.stdout.flush()

        if not is_multi_reps:
            print()

        tmA_arr = out_data
        tmA_reps_list.append(tmA_arr)

        colonized_a = int(np.sum((tmA_arr != -9999) & target_mask_A))
        max_st_a = int(np.max(tmA_arr[tmA_arr != -9999])) if colonized_a > 0 else 0
        not_reached_a = max(0, total_target_cells_A - colonized_a)
        pct_not_reached_a = 100 - int((colonized_a / total_target_cells_A) * 100) if total_target_cells_A > 0 else 0

        stats_a.append({
            "Steps": max_st_a,
            "Pixels": colonized_a,
            "Percentage": round((colonized_a / total_target_cells_A) * 100, 2) if total_target_cells_A > 0 else 100.0,
            "Not_reached": not_reached_a,
            "Percentage_not_reached": pct_not_reached_a
        })

        if replicates_save is not None and output_tif_path_A is not None and is_multi_reps:
            base_dir_a = os.path.dirname(output_tif_path_A)
            base_name_a = os.path.splitext(os.path.basename(output_tif_path_A))[0]
            ext_a = os.path.splitext(output_tif_path_A)[1]
            rep_path_a = os.path.join(base_dir_a, f"{base_name_a}_{rep + 1}{ext_a}")
            with rasterio.open(rep_path_a, 'w', **meta_out) as dst:
                dst.write(tmA_arr, 1)

    if is_multi_reps:
        print()
        stackA = np.array(tmA_reps_list, dtype=np.float32)
        stackA[stackA == -9999] = np.nan
        with np.errstate(all='ignore'):
            tmA_mean = np.nanmean(stackA, axis=0) if stackA.size > 0 else np.full((unified_height, unified_width), -9999, dtype=np.float32)
            tmA_sd = np.nanstd(stackA, axis=0) if stackA.size > 0 else np.zeros((unified_height, unified_width), dtype=np.float32)
        tmA_mean = np.nan_to_num(tmA_mean, nan=-9999).astype(np.int32)
        tmA_sd = np.nan_to_num(tmA_sd, nan=-9999).astype(np.float32)
        tmA_arr = tmA_mean

        if output_tif_path_A is not None:
            base_dir_a = os.path.dirname(output_tif_path_A)
            base_name_a = os.path.splitext(os.path.basename(output_tif_path_A))[0]
            ext_a = os.path.splitext(output_tif_path_A)[1]
            if base_dir_a: os.makedirs(base_dir_a, exist_ok=True)
            
            mean_path_a = os.path.join(base_dir_a, f"{base_name_a}_mean{ext_a}")
            sd_path_a = os.path.join(base_dir_a, f"{base_name_a}_sd{ext_a}")
            csv_path_a = os.path.join(base_dir_a, f"{base_name_a}_stats.csv")

            meta_mean_a = meta_out.copy()
            with rasterio.open(mean_path_a, 'w', **meta_mean_a) as dst:
                dst.write(tmA_mean, 1)

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
        if output_tif_path_A is not None:
            with rasterio.open(output_tif_path_A, 'w', **meta_out) as dstA:
                dstA.write(tmA_arr, 1)

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
        valid_a = (tmA_arr != -9999) & (~np.isnan(tmA_arr))

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
            "Occupied A + B": (128, 0, 128),
            "barrier": colors["barrier"]
        }

        base_frame[only_a] = gif_colors["Suitable for A"]
        base_frame[only_b] = gif_colors["Suitable for B"]
        base_frame[both] = gif_colors["Suitable A ∩ B"]
        
        if barrier_path is not None:
            base_frame[barA_gif] = gif_colors["barrier"]

        max_step_a = int(np.nanmax(tmA_arr[valid_a & (tmA_arr > 0)])) if np.any(valid_a & (tmA_arr > 0)) else 1

        if raster_path_B is not None and csv_path_B is not None:
            valid_b = (tmB != -9999) & (~np.isnan(tmB))
            max_step_b = int(np.nanmax(tmB[valid_b & (tmB > 0)])) if np.any(valid_b & (tmB > 0)) else 1
            max_global_step = max(max_step_a, max_step_b)
            unique_steps_a = tmA_arr[valid_a & (tmA_arr > 0)]
            unique_steps_b = tmB[valid_b & (valid_b > 0)]
            unique_colonization_steps = np.unique(np.concatenate([unique_steps_a, unique_steps_b]))
        else:
            max_global_step = max_step_a
            unique_colonization_steps = np.unique(tmA_arr[valid_a & (tmA_arr > 0)])

        if len(unique_colonization_steps) == 0:
            unique_colonization_steps = np.array([1])
        steps_range = sorted(unique_colonization_steps)
        if skip_steps > 1:
            steps_range = steps_range[::skip_steps]

        title_font_size = max(14, int(width * 0.025))
        legend_font_size = max(10, int(width * 0.013))
        try:
            font_title = ImageFont.truetype("arial.ttf", title_font_size)
            font_legend = ImageFont.truetype("arial.ttf", legend_font_size)
        except:
            font_title = ImageFont.load_default()
            font_legend = ImageFont.load_default()

        title_area_h = int(height * 0.12)
        map_max_w = int(width * 0.92)
        map_max_h = height - title_area_h - int(height * 0.04)

        scale = min(map_max_w / unified_width, map_max_h / unified_height)
        new_w, new_h = int(unified_width * scale), int(unified_height * scale)

        paste_x = (width - new_w) // 2
        paste_y = title_area_h + (map_max_h - new_h) // 2

        base_canvas = Image.new("RGB", (width, height), (255, 255, 255))
        draw_base = ImageDraw.Draw(base_canvas)

        if legend:
            if raster_path_B is not None and csv_path_B is not None:
                items = [
                    (gif_colors["Suitable for A"], "Suitable for A"),
                    (gif_colors["Suitable for B"], "Suitable for B"),
                    (gif_colors["Suitable A ∩ B"], "Suitable A ∩ B"),
                    (gif_colors["Occupied B"], "Occupied B"),
                    (gif_colors["Occupied A + B"], "Occupied A + B")
                ]
            else:
                items = [
                    (gif_colors["Suitable for A"], "Suitable for A"),
                    (gif_colors["Suitable for B"], "Suitable for B"),
                    (gif_colors["Suitable A ∩ B"], "Suitable A ∩ B"),
                    (gif_colors["Occupied A + B"], "Occupied A + B")
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

            draw_base.rectangle([box_x, box_y, box_x + box_w, box_y + box_h], fill=(255, 255, 255), outline=(0, 0, 0), width=max(1, int(width * 0.001)))

            for i, (col, label) in enumerate(items):
                current_y = box_y + box_pad + i * row_h
                draw_base.rectangle([box_x + box_pad, current_y + 2, box_x + box_pad + swatch_size, current_y + 2 + swatch_size], fill=col, outline=(0, 0, 0))
                draw_base.text((box_x + box_pad + swatch_size + int(width * 0.012), current_y + swatch_size // 2 + 2), label, fill=(0, 0, 0), anchor="lm", font=font_legend)

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

            if raster_path_B is not None and csv_path_B is not None:
                colonized_b_mask = (tmB <= step) & (tmB != -9999)
                frame[colonized_b_mask] = gif_colors["Occupied B"]

            colonized_a_mask = (tmA_arr <= step) & valid_a
            frame[colonized_a_mask] = gif_colors["Occupied A + B"]

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