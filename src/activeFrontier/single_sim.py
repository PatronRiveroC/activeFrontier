import os
import sys
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import mapping
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import maximum_filter
from numba import jit
import subprocess
import time
from numba import objmode

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
        sys.stdout.write(f"\r[{bar}] {pct}% (Step: {step} | Pixels: {pixels})")
    sys.stdout.flush()


def _update_progress(step, pixels):
    now = time.time()
    if now - _PROGRESS["t"] < 0.1:
        return
    _PROGRESS["t"] = now
    _render_progress(step, pixels)


@jit(nopython=True, cache=True)
def _run_single_simulation_numba(habitat_mask, start_y, start_x, max_steps, prob, d):
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

                    has_available_neighbor = False
                    for k in range(num_offsets):
                        nr = r + offsets_dy_arr[k]
                        nc = c + offsets_dx_arr[k]
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if habitat_mask[nr, nc] and not colonized[nr, nc]:
                                has_available_neighbor = True
                                if np.random.rand() <= prob:
                                    colonized[nr, nc] = True
                                    colonization_time[nr, nc] = step
                                    next_frontier[nr, nc] = True

                    if has_available_neighbor:
                        next_frontier[r, c] = True

        frontier = next_frontier
        
        current_colonized = np.sum(colonized)
        progress_steps.append(step)
        progress_colonized.append(current_colonized)

        with objmode():
            _update_progress(step, current_colonized)
        
    return colonization_time, np.array(progress_steps, dtype=np.int32), np.array(progress_colonized, dtype=np.int32)

def single_sim(
    raster_path,
    csv_path,
    output_tif_path,
    longitude_col="Longitude",
    latitude_col="Latitude",
    steps=None,
    prob=0.5,
    d=1,
    barrier_path=None,
    replicates=None,
    replicates_save=None,
    generate_gif=True,
    output_gif_path=None,
    fps=15,
    width=1080,
    height=920,
    skip_steps=1,
    legend=True,
    colors={"occupied": (34, 139, 34), "available": (220, 220, 220), "nodata": (255, 255, 255), "barrier": (0, 0, 0)}
):
    if replicates == 1:
        replicates = None

    print("Simulating single species dispersal")

    if output_gif_path is None:
        base_name_tif, _ = os.path.splitext(output_tif_path)
        output_gif_path = f"{base_name_tif}.gif"

    if not generate_gif:
        output_gif_path = None

    with rasterio.open(raster_path) as src:
        ext_data = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata_val = src.nodata
        rows, cols = src.shape
        meta = src.meta.copy()

    barrier_mask = np.zeros((rows, cols), dtype=bool)
    if barrier_path is not None:
        if str(barrier_path).endswith(('.shp', '.geojson', '.gpkg')):
            gdf = gpd.read_file(barrier_path)
            if gdf.crs != crs:
                gdf = gdf.to_crs(crs)
            geoms = [mapping(geom) for geom in gdf.geometry]
            bar_raster, _ = mask(rasterio.open(raster_path), geoms, crop=False, nodata=0, filled=True, all_touched=True)
            raw_barrier = (bar_raster[0] > 0)
            barrier_mask = maximum_filter(raw_barrier, size=3)
        elif str(barrier_path).endswith(('.tif', '.tiff', '.asc')):
            with rasterio.open(barrier_path) as b_src:
                bar_data = b_src.read(1)
                raw_barrier = (bar_data > 0) & (bar_data != b_src.nodata)
                barrier_mask = maximum_filter(raw_barrier, size=3)

    if nodata_val is not None:
        habitat_mask = (ext_data != nodata_val) & (~np.isnan(ext_data)) & (~barrier_mask) & (ext_data > 0)
    else:
        habitat_mask = (~np.isnan(ext_data)) & (~barrier_mask) & (ext_data > 0)

    total_valid_cells = int(np.sum(habitat_mask))

    occ = pd.read_csv(csv_path)
    initial_points = set()
    for _, row in occ.iterrows():
        row_idx, col_idx = rasterio.transform.rowcol(transform, row[longitude_col], row[latitude_col])
        if 0 <= row_idx < rows and 0 <= col_idx < cols:
            if habitat_mask[row_idx, col_idx]:
                initial_points.add((col_idx, row_idx))

    max_steps = steps if steps is not None else 100000
    num_reps = replicates if replicates is not None else 1
    is_multi_reps = (replicates is not None and replicates > 1)
    
    all_reps_data = []
    summary_records = []
    base_name, ext = os.path.splitext(output_tif_path)

    for rep in range(int(num_reps)):
        if is_multi_reps:
            sys.stdout.write(f"\r--- Running replicate {rep + 1} / {int(num_reps)} ---")
            sys.stdout.flush()

        start_y = np.array([pt[1] for pt in initial_points], dtype=np.int64)
        start_x = np.array([pt[0] for pt in initial_points], dtype=np.int64)
        
        _PROGRESS.update({"total": total_valid_cells, "rep": rep + 1, "num": int(num_reps), "multi": is_multi_reps, "t": 0.0})

        out_data, prog_steps, prog_colonized = _run_single_simulation_numba(habitat_mask, start_y, start_x, max_steps, prob, d)
        
        if len(prog_steps) > 0:
            _render_progress(int(prog_steps[-1]), int(prog_colonized[-1]))

        if not is_multi_reps:
            print()

        colonized_cells = int(np.sum(out_data != -9999))
        final_step = int(np.max(out_data[out_data != -9999])) if colonized_cells > 0 else 1

        total_colonized = colonized_cells
        total_unreached = max(0, total_valid_cells - total_colonized)
        unreached_pct = 100 - int((total_colonized / total_valid_cells) * 100) if total_valid_cells > 0 else 0
        final_percentage = round((total_colonized / total_valid_cells) * 100, 2) if total_valid_cells > 0 else 100.0

        if not is_multi_reps:
            print(f"Pixels do not reached: {total_unreached} | {unreached_pct}%")

        if is_multi_reps:
            summary_records.append({
                "Replicates": f"Sim_{rep + 1}",
                "Steps": final_step,
                "Pixels": total_colonized,
                "Percentage": final_percentage,
                "Not_reached": total_unreached,
                "Percentage_not_reached": unreached_pct
            })

        all_reps_data.append(out_data.astype(np.float32))

        if replicates_save is not None and is_multi_reps:
            rep_tif_path = f"{base_name}_{rep + 1}{ext}"
            meta.update(dtype=rasterio.float32, nodata=-9999, count=1)
            with rasterio.open(rep_tif_path, 'w', **meta) as dst:
                dst.write(out_data.astype(np.float32), 1)

    if is_multi_reps:
        print()

    mean_tif_path = output_tif_path
    sd_tif_path = f"{base_name}_sd{ext}"

    if is_multi_reps:
        stack = np.array(all_reps_data)
        mean_arr = np.full((rows, cols), -9999, dtype=np.float32)
        sd_arr = np.full((rows, cols), -9999, dtype=np.float32)

        for r in range(rows):
            for c in range(cols):
                vals = stack[:, r, c]
                valid_vals = vals[vals != -9999]
                if len(valid_vals) > 0:
                    raw_mean = np.mean(valid_vals)
                    frac, whole = np.modf(raw_mean)
                    mean_arr[r, c] = whole + 1 if frac >= 0.5 else whole
                    sd_arr[r, c] = np.std(valid_vals) if len(valid_vals) > 1 else 0.0

        mean_tif_path = f"{base_name}_mean{ext}"
        meta.update(dtype=rasterio.float32, nodata=-9999, count=1)
        with rasterio.open(mean_tif_path, 'w', **meta) as dst:
            dst.write(mean_arr, 1)

        with rasterio.open(sd_tif_path, 'w', **meta) as dst:
            dst.write(sd_arr, 1)

        df_summary = pd.DataFrame(summary_records)
        mean_row = {"Replicates": "Mean"}
        sd_row = {"Replicates": "SD"}
        
        numeric_cols = ["Steps", "Pixels", "Percentage", "Not_reached", "Percentage_not_reached"]
        for col in numeric_cols:
            mean_row[col] = df_summary[col].mean()
            sd_row[col] = df_summary[col].std()

        df_summary = pd.concat([df_summary, pd.DataFrame([mean_row, sd_row])], ignore_index=True)
        csv_output_path = f"{base_name}.csv"
        df_summary.to_csv(csv_output_path, index=False)
    else:
        meta.update(dtype=rasterio.float32, nodata=-9999, count=1)
        with rasterio.open(output_tif_path, 'w', **meta) as dst:
            dst.write(all_reps_data[0].astype(np.float32), 1)

    if output_gif_path is not None:
        print("Generating GIF simulation")
        with rasterio.open(mean_tif_path) as src_res:
            tm_arr = src_res.read(1)

        valid_habitat = (ext_data > 0) & (ext_data != nodata_val) & (~np.isnan(ext_data)) & (~barrier_mask)
        valid_sim = (tm_arr != -9999) & (~np.isnan(tm_arr))

        base_frame = np.full((rows, cols, 3), colors["nodata"], dtype=np.uint8)
        base_frame[valid_habitat] = colors["available"]
        
        if barrier_path is not None:
            base_frame[barrier_mask] = colors["barrier"]

        unique_colonization_steps = np.unique(tm_arr[valid_sim & (tm_arr > 0)])
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

        scale = min(map_max_w / cols, map_max_h / rows)
        new_w, new_h = int(cols * scale), int(rows * scale)

        paste_x = (width - new_w) // 2
        paste_y = title_area_h + (map_max_h - new_h) // 2

        base_canvas = Image.new("RGB", (width, height), (255, 255, 255))
        draw_base = ImageDraw.Draw(base_canvas)

        if legend:
            items = [
                (colors["available"], "Available"),
                (colors["occupied"], "Occupied")
            ]
            if barrier_path is not None:
                items.append((colors["barrier"], "Barrier"))

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
            colonized_mask = (tm_arr <= step) & valid_sim
            frame[colonized_mask] = colors["occupied"]

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