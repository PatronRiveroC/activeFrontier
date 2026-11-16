import os
import sys
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
import geopandas as gpd
from scipy.ndimage import maximum_filter
from numba import jit
from PIL import Image, ImageDraw, ImageFont
import subprocess

try:
    from single_sim import single_sim
except ImportError:
    pass

@jit(nopython=True, cache=True)
def _run_exclusion_simulation_numba(target_mask_B, tmA, start_y, start_x, max_steps, prob_B, disp_B, rows, cols):
    colonization_time = np.full((rows, cols), -9999, dtype=np.int32)
    colonized = np.zeros((rows, cols), dtype=np.bool_)
    frontier = np.zeros((rows, cols), dtype=np.bool_)
    
    for i in range(len(start_y)):
        y = start_y[i]
        x = start_x[i]
        if 0 <= y < rows and 0 <= x < cols:
            if target_mask_B[y, x] and not (tmA[y, x] != -9999 and tmA[y, x] <= 0):
                colonization_time[y, x] = 0
                colonized[y, x] = True
                frontier[y, x] = True

    int_d = int(np.ceil(disp_B))
    offsets_dy = []
    offsets_dx = []
    for dy in range(-int_d, int_d + 1):
        for dx in range(-int_d, int_d + 1):
            if dx == 0 and dy == 0:
                continue
            if np.sqrt(dx*dx + dy*dy) <= disp_B:
                offsets_dy.append(dy)
                offsets_dx.append(dx)
                
    offsets_dy_arr = np.array(offsets_dy, dtype=np.int32)
    offsets_dx_arr = np.array(offsets_dx, dtype=np.int32)
    num_offsets = len(offsets_dy_arr)

    progress_steps = []
    progress_colonized = []

    for step in range(max_steps):
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
        next_t = step + 1

        for r in range(rows):
            for c in range(cols):
                if frontier[r, c]:
                    if tmA[r, c] != -9999 and tmA[r, c] <= next_t:
                        continue

                    has_any_valid = False
                    has_any_future = False
                    ready_indices_dy = np.empty(num_offsets, dtype=np.int32)
                    ready_indices_dx = np.empty(num_offsets, dtype=np.int32)
                    ready_num = 0

                    for k in range(num_offsets):
                        nr = r + offsets_dy_arr[k]
                        nc = c + offsets_dx_arr[k]
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if target_mask_B[nr, nc] and not colonized[nr, nc]:
                                if not (tmA[nr, nc] != -9999 and tmA[nr, nc] <= next_t):
                                    has_any_valid = True
                                    has_any_future = True
                                    ready_indices_dy[ready_num] = offsets_dy_arr[k]
                                    ready_indices_dx[ready_num] = offsets_dx_arr[k]
                                    ready_num += 1

                    if not has_any_valid or not has_any_future:
                        continue

                    if ready_num == 0:
                        next_frontier[r, c] = True
                        continue

                    retire_b = True
                    for i in range(ready_num):
                        nr = r + ready_indices_dy[i]
                        nc = c + ready_indices_dx[i]
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if target_mask_B[nr, nc] and not colonized[nr, nc]:
                                if not (tmA[nr, nc] != -9999 and tmA[nr, nc] <= next_t):
                                    if np.random.rand() <= prob_B:
                                        colonized[nr, nc] = True
                                        colonization_time[nr, nc] = next_t
                                        next_frontier[nr, nc] = True
                                    else:
                                        retire_b = False

                    if not retire_b:
                        next_frontier[r, c] = True

        frontier = next_frontier
        current_colonized = np.sum(colonized & target_mask_B)
        progress_steps.append(next_t)
        progress_colonized.append(current_colonized)

    return colonization_time, np.array(progress_steps, dtype=np.int32), np.array(progress_colonized, dtype=np.int32)

def exclusion_sim(
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
        output_tif_path_A = os.path.join(out_d, f"{base_a}_exclusion.tif" if out_d else f"{base_a}_exclusion.tif")

    if raster_path_B is not None and output_tif_path_B is None:
        base_b = os.path.splitext(os.path.basename(raster_path_B))[0]
        out_d = output_dir if output_dir is not None else os.path.dirname(raster_path_B)
        output_tif_path_B = os.path.join(out_d, f"{base_b}_exclusion.tif" if out_d else f"{base_b}_exclusion.tif")

    if output_gif_path is None:
        if output_dir is not None:
            output_gif_path = os.path.join(output_dir, "exclusion.gif")
        else:
            output_gif_path = "exclusion.gif"

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
        bl = str(barrier_path).lower()
        if bl.endswith(('.shp', '.geojson', '.gpkg')):
            gdf = gpd.read_file(barrier_path)
            if gdf.crs != crsA: gdf = gdf.to_crs(crsA)
            shapes = [(geom, 1) for geom in gdf.geometry]
            barA_raw = rasterize(
                shapes,
                out_shape=(unified_height, unified_width),
                transform=common_transform,
                fill=0,
                default_value=1,
                all_touched=True
            ).astype(bool)
            barA = maximum_filter(barA_raw, size=3)
        elif bl.endswith(('.tif', '.tiff', '.asc')):
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
                barA = maximum_filter(bar_warp > 0, size=3)

    print("Simulating species A")
    temp_tif_A = output_tif_path_A

    single_sim(
        raster_path=raster_path_A,
        csv_path=csv_path_A,
        output_tif_path=temp_tif_A,
        longitude_col=longitude_col,
        latitude_col=latitude_col,
        steps=steps,
        prob=prob_A,
        d=disp_A,
        barrier_path=barrier_path,
        replicates=replicates,
        replicates_save=replicates_save,
        generate_gif=False,
        output_gif_path=None
    )

    base_a, ext_a = os.path.splitext(temp_tif_A)
    actual_tif_A = f"{base_a}_mean{ext_a}" if is_multi_reps else temp_tif_A

    tmA = warp_to_common(actual_tif_A)

    if output_tif_path_A is not None:
        with rasterio.open(actual_tif_A, 'w', **meta_out) as dst:
            dst.write(tmA.astype(np.int32), 1)

    tmB_reps_list = []
    stats_b = []
    validB = (rB_data != -9999) & (~np.isnan(rB_data)) & (rB_data > 0)

    if raster_path_B is not None and csv_path_B is not None:
        print("Simulating species B (Exclusion by A)")
        for rep in range(num_reps):
            occB = pd.read_csv(csv_path_B)
            initB_list = []
            target_mask_B = validB & (~barA)
            for _, row in occB.iterrows():
                try:
                    lon = row[longitude_col_B]; lat = row[latitude_col_B]
                    if pd.isna(lon) or pd.isna(lat): continue
                    r, c = rasterio.transform.rowcol(common_transform, lon, lat)
                    if 0 <= r < unified_height and 0 <= c < unified_width:
                        if target_mask_B[r, c] and not (tmA[r, c] != -9999 and tmA[r, c] <= 0):
                            initB_list.append((r, c))
                except: pass

            start_y = np.array([pt[0] for pt in initB_list], dtype=np.int64)
            start_x = np.array([pt[1] for pt in initB_list], dtype=np.int64)
            total_target_cells_B = int(np.sum(target_mask_B))

            out_data_b, prog_steps_b, prog_colonized_b = _run_exclusion_simulation_numba(
                target_mask_B, tmA, start_y, start_x, steps_max, prob_B, disp_B, unified_height, unified_width
            )

            total_p_steps = len(prog_steps_b)
            for idx_p in range(total_p_steps):
                c_step = prog_steps_b[idx_p]
                c_pixels = prog_colonized_b[idx_p]
                percentage = int((c_pixels / total_target_cells_B) * 100) if total_target_cells_B > 0 else 100
                is_last = (idx_p == total_p_steps - 1)

                if c_step % 10 == 0 or is_last or percentage >= 100:
                    bar_width = 30
                    blocks = int(bar_width * percentage / 100)
                    bar = "=" * blocks + " " * (bar_width - blocks)
                    if is_multi_reps:
                        print(f"[{bar}] {percentage}% (Rep {rep + 1}/{num_reps})", end='\r')
                    else:
                        print(f"[{bar}] {percentage}% (Step: {c_step} | Pixels B: {int(c_pixels)})", end='\r')

            if not is_multi_reps:
                print()

            tmB_arr = out_data_b
            tmB_reps_list.append(tmB_arr)
            colonized_b = int(np.sum((tmB_arr != -9999) & target_mask_B))
            max_st_b = int(np.max(tmB_arr[tmB_arr != -9999])) if np.any(tmB_arr != -9999) else 0
            not_reached_b = max(0, total_target_cells_B - colonized_b)
            pct_not_reached_b = 100 - int((colonized_b / total_target_cells_B) * 100) if total_target_cells_B > 0 else 0

            stats_b.append({
                "Steps": max_st_b,
                "Pixels": colonized_b,
                "Percentage": round((colonized_b / total_target_cells_B) * 100, 2) if total_target_cells_B > 0 else 100.0,
                "Not_reached": not_reached_b,
                "Percentage_not_reached": pct_not_reached_b
            })

            if replicates_save is not None and output_tif_path_B is not None and is_multi_reps:
                base_dir = os.path.dirname(output_tif_path_B)
                base_name = os.path.splitext(os.path.basename(output_tif_path_B))[0]
                ext = os.path.splitext(output_tif_path_B)[1]
                rep_path = os.path.join(base_dir, f"{base_name}_{rep + 1}{ext}")
                with rasterio.open(rep_path, 'w', **meta_out) as dst:
                    dst.write(tmB_arr, 1)

        if is_multi_reps:
            print()
            stackB = np.array(tmB_reps_list, dtype=np.float32)
            stackB[stackB == -9999] = np.nan
            with np.errstate(all='ignore'):
                tmB_mean = np.nanmean(stackB, axis=0) if stackB.size > 0 else np.full((unified_height, unified_width), -9999, dtype=np.float32)
                tmB_sd = np.nanstd(stackB, axis=0) if stackB.size > 0 else np.zeros((unified_height, unified_width), dtype=np.float32)
            tmB_mean = np.nan_to_num(tmB_mean, nan=-9999).astype(np.int32)
            tmB_sd = np.nan_to_num(tmB_sd, nan=-9999).astype(np.float32)
            tmB = tmB_mean

            if output_tif_path_B is not None:
                base_dir = os.path.dirname(output_tif_path_B)
                base_name = os.path.splitext(os.path.basename(output_tif_path_B))[0]
                ext = os.path.splitext(output_tif_path_B)[1]
                if base_dir: os.makedirs(base_dir, exist_ok=True)
                
                mean_path = os.path.join(base_dir, f"{base_name}_mean{ext}")
                sd_path = os.path.join(base_dir, f"{base_name}_sd{ext}")
                csv_path = os.path.join(base_dir, f"{base_name}_stats.csv")

                meta_mean = meta_out.copy()
                with rasterio.open(mean_path, 'w', **meta_mean) as dst:
                    dst.write(tmB_mean, 1)

                meta_sd = meta_out.copy()
                meta_sd.update({'dtype': rasterio.float32, 'nodata': -9999})
                with rasterio.open(sd_path, 'w', **meta_sd) as dst:
                    dst.write(tmB_sd, 1)

                df_b = pd.DataFrame(stats_b)
                mean_row_b = df_b.mean(numeric_only=True).to_dict()
                sd_row_b = df_b.std(numeric_only=True).to_dict()
                mean_row_b["Steps"] = "Mean"
                sd_row_b["Steps"] = "SD"
                
                df_b_out = pd.concat([df_b, pd.DataFrame([mean_row_b, sd_row_b])], ignore_index=True)
                df_b_out.to_csv(csv_path, index=False)
        else:
            tmB = tmB_reps_list[0]
            if output_tif_path_B is not None:
                with rasterio.open(output_tif_path_B, 'w', **meta_out) as dstB:
                    dstB.write(tmB, 1)
    else:
        tmB = np.full((unified_height, unified_width), -9999, dtype=np.int32)

    if output_gif_path is not None:
        print("Generating GIF simulation")

        valid_habitat_a = (rA_data > 0) & (rA_data != -9999) & (~np.isnan(rA_data))
        valid_habitat_b = (rB_data > 0) & (rB_data != -9999) & (~np.isnan(rB_data))
        valid_a = (tmA != -9999) & (~np.isnan(tmA))
        valid_b = (tmB != -9999) & (~np.isnan(tmB))

        if barrier_path is not None:
            barA_gif = barA & (valid_habitat_a | valid_habitat_b)
        else:
            barA_gif = np.zeros((unified_height, unified_width), dtype=bool)

        base_frame = np.full((unified_height, unified_width, 3), colors["nodata"], dtype=np.uint8)

        only_a = valid_habitat_a & ~valid_habitat_b
        only_b = ~valid_habitat_a & valid_habitat_b
        both = valid_habitat_a & valid_habitat_b

        gif_colors = {
            "available_A": (230, 245, 230),
            "available_B": (230, 240, 255),
            "available_both": (240, 230, 250),
            "species_A": colors["occupied"],
            "species_B": (30, 144, 255),
            "excluded": (128, 0, 128),
            "barrier": colors["barrier"]
        }

        base_frame[only_a] = gif_colors["available_A"]
        base_frame[only_b] = gif_colors["available_B"]
        base_frame[both] = gif_colors["available_both"]

        if barrier_path is not None:
            base_frame[barA_gif] = gif_colors["barrier"]

        max_step_a = int(np.nanmax(tmA[valid_a & (tmA > 0)])) if np.any(valid_a & (tmA > 0)) else 1
        max_step_b = int(np.nanmax(tmB[valid_b & (tmB > 0)])) if np.any(valid_b & (tmB > 0)) else 1
        max_global_step = max(max_step_a, max_step_b)

        title_font_size = max(14, int(width * 0.025))
        legend_font_size = max(10, int(width * 0.013))
        try:
            font_title = ImageFont.truetype("arial.ttf", title_font_size)
            font_legend = ImageFont.truetype("arial.ttf", legend_font_size)
        except:
            font_title = ImageFont.load_default()
            font_legend = ImageFont.load_default()

        map_max_w = int(width * 0.92)
        map_max_h = int(height * 0.80)
        scale = min(map_max_w / unified_width, map_max_h / unified_height)
        new_w, new_h = int(unified_width * scale), int(unified_height * scale)

        paste_x = (width - new_w) // 2
        paste_y = 65

        base_canvas = Image.new("RGB", (width, height), (255, 255, 255))

        if legend:
            items = [
                (gif_colors["available_A"], "Available A"),
                (gif_colors["available_B"], "Available B"),
                (gif_colors["available_both"], "Suitable A ∩ B"),
                (gif_colors["species_A"], "Colonized A"),
                (gif_colors["species_B"], "Colonized B"),
                (gif_colors["excluded"], "Colonized A (by excluding B)")
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

            box_x = paste_x + 10
            box_y = max(paste_y + 10, paste_y + new_h - box_h - 10)

        steps_range = list(range(1, max_global_step + 1, skip_steps))
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

            colonized_b_mask = (tmB <= step) & valid_b
            colonized_a_mask = (tmA <= step) & valid_a

            excluded_mask = colonized_a_mask & colonized_b_mask
            only_b_mask = colonized_b_mask & ~colonized_a_mask
            only_a_mask = colonized_a_mask & ~colonized_b_mask

            frame[only_b_mask] = gif_colors["species_B"]
            frame[only_a_mask] = gif_colors["species_A"]
            frame[excluded_mask] = gif_colors["excluded"]

            if barrier_path is not None:
                frame[barA_gif] = gif_colors["barrier"]

            img = Image.fromarray(frame)
            img_resized = img.resize((new_w, new_h), Image.Resampling.NEAREST)

            canvas = base_canvas.copy()
            canvas.paste(img_resized, (paste_x, paste_y))

            draw = ImageDraw.Draw(canvas)
            draw.text((width // 2, 30), f"Simulation Step: {step}", fill=(0, 0, 0), anchor="mm", font=font_title)

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