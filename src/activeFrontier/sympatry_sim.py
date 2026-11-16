import os
import sys
import glob
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
    HAS_SINGLE_SIM = True
except ImportError:
    HAS_SINGLE_SIM = False

@jit(nopython=True, cache=True)
def _run_sympatry_simulation_numba(habitat_mask, start_y, start_x, max_steps, prob, d):
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
        current_colonized = np.sum(colonized & habitat_mask)
        progress_steps.append(step)
        progress_colonized.append(current_colonized)
        
    return colonization_time, np.array(progress_steps, dtype=np.int32), np.array(progress_colonized, dtype=np.int32)

@jit(nopython=True, cache=True)
def _compute_sympatry_levels_numba(tm_stack, valid_stack, any_valid, combined_barrier, max_global_step, num_species):
    rows, cols = any_valid.shape
    sympatry_steps = np.full((num_species, rows, cols), -9999, dtype=np.int32)
    
    for s in range(1, max_global_step + 1):
        for r in range(rows):
            for c in range(cols):
                if not any_valid[r, c] or combined_barrier[r, c]:
                    continue
                
                richness = 0
                for i in range(num_species):
                    val = tm_stack[i, r, c]
                    if val != -9999 and val <= s and valid_stack[i, r, c]:
                        richness += 1
                
                for r_lvl in range(1, num_species + 1):
                    if richness >= r_lvl and sympatry_steps[r_lvl - 1, r, c] == -9999:
                        sympatry_steps[r_lvl - 1, r, c] = s
                        
    return sympatry_steps

def sympatry_sim(
    raster_path,
    csv_path,
    output_tif_path=None,
    longitude_col="Longitude",
    latitude_col="Latitude",
    steps=None,
    prob=0.5,
    d=1,
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
    simpatry_mode="richness",
    colors={"nodata": (255, 255, 255), "colonizable": (200, 200, 200), "barrier": (0, 0, 0)},
    **kwargs
):
    if isinstance(raster_path, str):
        if os.path.isdir(raster_path):
            raster_path = sorted(glob.glob(os.path.join(raster_path, "*.tif")) + glob.glob(os.path.join(raster_path, "*.tiff")))
        else:
            raster_path = [raster_path]
    elif isinstance(raster_path, (list, tuple, np.ndarray)):
        expanded_rasters = []
        for rp in raster_path:
            if os.path.isdir(rp):
                expanded_rasters.extend(sorted(glob.glob(os.path.join(rp, "*.tif")) + glob.glob(os.path.join(rp, "*.tiff"))))
            else:
                expanded_rasters.append(rp)
        raster_path = expanded_rasters

    num_species = len(raster_path)

    if isinstance(csv_path, str) and os.path.isdir(csv_path):
        csv_files = sorted(glob.glob(os.path.join(csv_path, "*.csv")))
        csv_path = csv_files if len(csv_files) > 0 else [None] * num_species
    elif isinstance(csv_path, (list, tuple, np.ndarray)):
        expanded_csvs = []
        for cp in csv_path:
            if cp is not None and isinstance(cp, str) and os.path.isdir(cp):
                expanded_csvs.extend(sorted(glob.glob(os.path.join(cp, "*.csv"))))
            else:
                expanded_csvs.append(cp)
        csv_path = expanded_csvs
    elif csv_path is None or (isinstance(csv_path, str) and csv_path.strip().upper() == "NA"):
        csv_path = [csv_path] * num_species

    if len(csv_path) < num_species:
        csv_path = list(csv_path) + [csv_path[-1] if len(csv_path) > 0 else None] * (num_species - len(csv_path))

    if output_tif_path is None:
        output_tif_path = []
        for r_p in raster_path:
            base_r = os.path.splitext(os.path.basename(r_p))[0]
            out_d = output_dir if output_dir is not None else os.path.dirname(r_p)
            output_tif_path.append(os.path.join(out_d, f"{base_r}_sympatry.tiff"))
    elif isinstance(output_tif_path, str):
        base_out, ext_out = os.path.splitext(output_tif_path)
        output_tif_path = [f"{base_out}_spp{i+1}{ext_out}" for i in range(num_species)]

    if not isinstance(d, (list, tuple, np.ndarray)):
        d_list = [d] * num_species
    else:
        d_list = list(d)
        if len(d_list) < num_species:
            d_list = d_list + [d_list[-1]] * (num_species - len(d_list))

    if barrier_path is None or isinstance(barrier_path, (str, bytes)):
        barrier_list = [barrier_path] * num_species
    elif isinstance(barrier_path, (list, tuple, np.ndarray)):
        barrier_list = list(barrier_path)
        if len(barrier_list) < num_species:
            barrier_list = barrier_list + [barrier_list[-1]] * (num_species - len(barrier_list))
    else:
        barrier_list = [barrier_path] * num_species

    if replicates == 1:
        replicates = None

    if output_gif_path is None:
        out_d = output_dir if output_dir is not None else "."
        os.makedirs(out_d, exist_ok=True)
        output_gif_path = os.path.join(out_d, "sympatry.gif")

    if not generate_gif:
        output_gif_path = None

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(raster_path[0]) as src_ref:
        bounds_ref = src_ref.bounds
        crs_ref = src_ref.crs
        res_ref = src_ref.res

    left, bottom, right, top = bounds_ref.left, bounds_ref.bottom, bounds_ref.right, bounds_ref.top
    for p in raster_path[1:]:
        with rasterio.open(p) as src_p:
            b = src_p.bounds
            left = min(left, b.left)
            bottom = min(bottom, b.bottom)
            right = max(right, b.right)
            top = max(top, b.top)

    res_x, res_y = res_ref[0], res_ref[1]
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
                dst_crs=crs_ref,
                resampling=Resampling.nearest,
                src_nodata=nd,
                dst_nodata=nd
            )
            return out_arr

    meta_out = src_ref.meta.copy()
    meta_out.update({
        'height': unified_height,
        'width': unified_width,
        'transform': common_transform,
        'dtype': rasterio.int32,
        'nodata': -9999,
        'count': 1
    })

    tm_list = []
    valid_list = []
    study_area_list = []
    barrier_masks = []

    for i in range(num_species):
        r_path = raster_path[i]
        c_path = csv_path[i] if i < len(csv_path) else None
        b_path = barrier_list[i]
        curr_d = int(d_list[i])

        r_data = warp_to_common(r_path)
        nd_val = -9999

        s_mask = (r_data != nd_val) & (~np.isnan(r_data)) & (r_data > 0)
        study_area_list.append(s_mask)

        bar_species = np.zeros((unified_height, unified_width), dtype=bool)
        if b_path is not None:
            bl = str(b_path).lower()
            if bl.endswith(('.shp', '.geojson', '.gpkg')):
                gdf = gpd.read_file(b_path)
                if gdf.crs != crs_ref: gdf = gdf.to_crs(crs_ref)
                shapes = [(geom, 1) for geom in gdf.geometry]
                bar_raw = rasterize(
                    shapes,
                    out_shape=(unified_height, unified_width),
                    transform=common_transform,
                    fill=0,
                    default_value=1,
                    all_touched=True
                ).astype(bool)
                bar_species = maximum_filter(bar_raw, size=2)
            elif bl.endswith(('.tif', '.tiff', '.asc')):
                with rasterio.open(b_path) as b_src:
                    bar_data = b_src.read(1)
                    bar_warp = np.zeros((unified_height, unified_width), dtype=np.uint8)
                    reproject(
                        source=(bar_data > 0).astype(np.uint8),
                        destination=bar_warp,
                        src_transform=b_src.transform,
                        src_crs=b_src.crs,
                        dst_transform=common_transform,
                        dst_crs=crs_ref,
                        resampling=Resampling.nearest,
                        src_nodata=b_src.nodata if b_src.nodata is not None else 0,
                        dst_nodata=0
                    )
                    bar_species = maximum_filter(bar_warp > 0, size=2)

        barrier_masks.append(bar_species)
        v_mask = s_mask & (~bar_species)
        valid_list.append(v_mask)

        is_na = (c_path is None) or (str(c_path).strip().upper() == "NA") or (pd.isna(c_path))

        if is_na:
            max_steps_val = steps if steps is not None else 50
            np.random.seed(42 + i)
            tm_arr = np.where(v_mask, np.random.randint(1, max_steps_val + 1, size=v_mask.shape), -9999).astype(np.int32)
            tm_arr[~v_mask] = -9999
        else:
            if HAS_SINGLE_SIM:
                print(f"Simulating Species {i+1}...")
                # Nota: Adaptar argumentos según la firma de tu función single_sim si difiere
                tm_arr = single_sim(
                    raster_path=r_path,
                    csv_path=c_path,
                    longitude_col=longitude_col,
                    latitude_col=latitude_col,
                    steps=steps,
                    prob=prob,
                    d=curr_d,
                    barrier_path=b_path,
                    verbose=verbose,
                    **kwargs
                )
                if isinstance(tm_arr, str):
                    with rasterio.open(tm_arr) as src_tmp:
                        tm_arr = src_tmp.read(1)
                tm_arr = tm_arr.astype(np.int32)
                tm_arr[bar_species] = -9999
            else:
                occ = pd.read_csv(c_path)
                initial_points = set()
                for _, row in occ.iterrows():
                    try:
                        lon = row[longitude_col]; lat = row[latitude_col]
                        if pd.isna(lon) or pd.isna(lat): continue
                        r, c = rasterio.transform.rowcol(common_transform, lon, lat)
                        if 0 <= r < unified_height and 0 <= c < unified_width:
                            if v_mask[r, c]: initial_points.add((c, r))
                    except: pass

                start_y = np.array([pt[1] for pt in initial_points], dtype=np.int64)
                start_x = np.array([pt[0] for pt in initial_points], dtype=np.int64)
                max_steps_val = steps if steps is not None else 100000

                total_target_cells = int(np.sum(v_mask))
                habitat_mask_sp = v_mask.copy()

                out_data, prog_steps, prog_colonized = _run_sympatry_simulation_numba(
                    habitat_mask_sp, start_y, start_x, max_steps_val, prob, curr_d
                )

                total_p_steps = len(prog_steps)
                for idx_p in range(total_p_steps):
                    c_step = prog_steps[idx_p]
                    c_pixels = prog_colonized[idx_p]
                    percentage = int((c_pixels / total_target_cells) * 100) if total_target_cells > 0 else 100
                    is_last = (idx_p == total_p_steps - 1)

                    if c_step % 250 == 0 or is_last or percentage >= 100:
                        bar_width = 30
                        blocks = int(bar_width * percentage / 100)
                        bar = "=" * blocks + " " * (bar_width - blocks)
                        end_char = "\n" if is_last else "\r"
                        sys.stdout.write(f"Species {i+1} [{bar}] {percentage}% (Step: {c_step} | Pixels: {c_pixels}){end_char}")
                        sys.stdout.flush()

                tm_arr = out_data
                tm_arr[bar_species] = -9999

        tm_list.append(tm_arr)

        if output_tif_path is not None and i < len(output_tif_path):
            out_p = output_tif_path[i]
            with rasterio.open(out_p, 'w', **meta_out) as dst:
                dst.write(tm_arr.astype(np.int32), 1)

    any_study_area = np.any(np.stack(study_area_list, axis=0), axis=0)
    any_valid = np.any(np.stack(valid_list, axis=0), axis=0)

    raw_combined_barrier = np.any(np.stack(barrier_masks, axis=0), axis=0)
    combined_barrier = raw_combined_barrier & any_study_area

    max_global_step = 1
    for tm in tm_list:
        v_m = (tm != -9999) & (~np.isnan(tm))
        if np.any(v_m & (tm > 0)):
            mx = int(np.nanmax(tm[v_m & (tm > 0)]))
            if mx > max_global_step:
                max_global_step = mx

    tm_stack = np.stack(tm_list, axis=0).astype(np.int32)
    valid_stack = np.stack(valid_list, axis=0).astype(np.bool_)

    sympatry_steps_arr = _compute_sympatry_levels_numba(
        tm_stack, valid_stack, any_valid, combined_barrier, max_global_step, num_species
    )

    sympatry_step_tiffs = {}
    for r_lvl in range(1, num_species + 1):
        arr = sympatry_steps_arr[r_lvl - 1]
        if np.any(arr != -9999):
            sympatry_step_tiffs[r_lvl] = arr

    out_dir_path = output_dir if output_dir is not None else "."
    os.makedirs(out_dir_path, exist_ok=True)
    meta_sym = meta_out.copy()
    meta_sym.update(dtype=rasterio.int32, nodata=-9999)

    for r_lvl, arr in sympatry_step_tiffs.items():
        with rasterio.open(os.path.join(out_dir_path, f"sympatry_{r_lvl}.tiff"), "w", **meta_sym) as dst:
            dst.write(arr, 1)

    if output_gif_path is not None:
        print("Generating GIF simulation")

        barrier_draw = raw_combined_barrier & any_study_area

        max_richness_observed = max(list(sympatry_step_tiffs.keys()) + [1])
        max_pot = max(max_richness_observed, 1)

        pastel_palette = [
            (254, 224, 139),
            (253, 174, 97),
            (244, 109, 67),
            (215, 48, 39)
        ]

        qualitative_palette = [
            (27, 158, 119),
            (217, 95, 2),
            (117, 112, 179),
            (231, 41, 138),
            (102, 166, 30),
            (230, 171, 2),
            (166, 86, 40),
            (164, 194, 215)
        ]

        def get_species_color(idx):
            return qualitative_palette[idx % len(qualitative_palette)]

        def get_pastel_palette_color(val, max_val):
            if max_val <= 1 or len(pastel_palette) == 1:
                return pastel_palette[0]
            ratio = (val - 1) / (max_val - 1)
            ratio = max(0.0, min(1.0, ratio))
            pos = ratio * (len(pastel_palette) - 1)
            i0 = int(np.floor(pos))
            i1 = min(i0 + 1, len(pastel_palette) - 1)
            t = pos - i0
            c0 = pastel_palette[i0]
            c1 = pastel_palette[i1]
            return tuple(int(round(c0[k] + (c1[k] - c0[k]) * t)) for k in range(3))

        base_frame = np.full((unified_height, unified_width, 3), colors.get("nodata", (255, 255, 255)), dtype=np.uint8)
        colonizable_color = colors.get("colonizable", (200, 200, 200))
        base_frame[any_study_area & (~barrier_draw)] = colonizable_color

        if np.any(barrier_draw):
            base_frame[barrier_draw] = colors.get("barrier", (0, 0, 0))

        try:
            font_title = ImageFont.truetype("arial.ttf", 34)
            font_legend = ImageFont.truetype("arial.ttf", 14)
            font_legend_title = ImageFont.truetype("arial.ttf", 16)
        except:
            font_title = ImageFont.load_default()
            font_legend = ImageFont.load_default()
            font_legend_title = ImageFont.load_default()

        steps_range = list(range(1, max_global_step + 1, skip_steps))
        total_gif_steps = len(steps_range)

        base_canvas = Image.new("RGB", (width, height), (255, 255, 255))
        draw_base = ImageDraw.Draw(base_canvas)

        left_margin = int(width * 0.16)
        right_margin = int(width * 0.18)

        map_max_w = width - left_margin - right_margin
        map_max_h = int(height * 0.85)
        scale = min(map_max_w / unified_width, map_max_h / unified_height)
        new_w, new_h = int(unified_width * scale), int(unified_height * scale)

        paste_x = left_margin + (map_max_w - new_w) // 2
        paste_y = 65 + (map_max_h - new_h) // 2

        if legend:
            items = [
                (colonizable_color, "Available")
            ]
            if barrier_path is not None:
                items.append((colors.get("barrier", (0, 0, 0)), "Barrier"))

            swatch_size = max(10, int(width * 0.015))
            box_pad = max(8, int(width * 0.01))
            row_h = swatch_size + max(8, int(width * 0.008))

            max_text_w = 0
            for _, label in items:
                try:
                    t_w = draw_base.textlength(label, font=font_legend)
                except AttributeError:
                    t_w = font_legend.getsize(label)[0]
                if t_w > max_text_w:
                    max_text_w = t_w

            text_spacing = int(width * 0.012)
            box_w = int(box_pad * 2 + swatch_size + text_spacing + max_text_w)
            box_h = box_pad * 2 + len(items) * row_h

            leg_x = paste_x + new_w + 20
            leg_y = paste_y
            leg_w = width - leg_x - 20
            leg_h = new_h

            if leg_w > 60:
                draw_base.text((leg_x + leg_w // 2, leg_y + 10), "Richness", fill=(0, 0, 0), anchor="mm", font=font_legend_title)

                bar_x = leg_x + 20
                bar_y = leg_y + 40
                bar_w_box = 30
                bar_h_box = leg_h - 60

                for y_px in range(bar_h_box):
                    ratio_y = 1.0 - (y_px / max(1, bar_h_box - 1))
                    val_ratio = 1.0 + ratio_y * (max_pot - 1.0) if max_pot > 1 else 1.0
                    col_box = get_pastel_palette_color(int(round(val_ratio)), max_pot)
                    draw_base.line([(bar_x, bar_y + y_px), (bar_x + bar_w_box, bar_y + y_px)], fill=col_box)

                draw_base.rectangle([bar_x, bar_y, bar_x + bar_w_box, bar_y + bar_h_box], outline=(0, 0, 0), width=1)

                for v in range(1, max_pot + 1):
                    if max_pot > 1:
                        frac_v = (v - 1) / (max_pot - 1)
                    else:
                        frac_v = 0.0
                    py = bar_y + bar_h_box - int(frac_v * bar_h_box)
                    draw_base.line([(bar_x + bar_w_box, py), (bar_x + bar_w_box + 5, py)], fill=(0, 0, 0))
                    draw_base.text((bar_x + bar_w_box + 10, py), str(v), fill=(0, 0, 0), anchor="lm", font=font_legend)

            box_x = 30
            box_y = paste_y + new_h - box_h

            draw_base.rectangle([box_x, box_y, box_x + box_w, box_y + box_h], fill=(255, 255, 255), outline=(0, 0, 0), width=max(1, int(width * 0.001)))

            for i, (col, label) in enumerate(items):
                current_y = box_y + box_pad + i * row_h
                draw_base.rectangle([box_x + box_pad, current_y + 2, box_x + box_pad + swatch_size, current_y + 2 + swatch_size], fill=col, outline=(0, 0, 0))
                draw_base.text((box_x + box_pad + swatch_size + int(width * 0.012), current_y + swatch_size // 2 + 2), label, fill=(0, 0, 0), anchor="lm", font=font_legend)

        ffmpeg_cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}', '-pix_fmt', 'rgb24', '-r', str(fps),
            '-i', '-', '-filter_complex', '[0:v] split [a][b];[a] palettegen=stats_mode=full [p];[b][p] paletteuse=dither=sierra2_4a',
            output_gif_path
        ]

        pipe = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

        for idx_g, step in enumerate(steps_range):
            pct_gif = int(((idx_g + 1) / total_gif_steps) * 100)
            bar_width = 30
            blocks = int(bar_width * pct_gif / 100)
            bar = "=" * blocks + " " * (bar_width - blocks)
            sys.stdout.write(f"\r[{bar}] {pct_gif}% (GIF Frame {idx_g + 1}/{total_gif_steps})")
            sys.stdout.flush()

            frame = base_frame.copy()

            masks = []
            for i in range(num_species):
                m = (tm_list[i] <= step) & (tm_list[i] != -9999) & valid_list[i] & (~barrier_masks[i])
                masks.append(m)
            stack_masks = np.stack(masks, axis=0)

            if simpatry_mode == "qualitative":
                sum_active = np.sum(stack_masks.astype(int), axis=0)
                single_active_mask = (sum_active == 1) & any_study_area & (~barrier_draw)
                multi_active_mask = (sum_active > 1) & any_study_area & (~barrier_draw)

                for i in range(num_species):
                    sp_single = single_active_mask & stack_masks[i]
                    if np.any(sp_single):
                        frame[sp_single] = get_species_color(i)

                if np.any(multi_active_mask):
                    frame[multi_active_mask] = (117, 112, 179)
            else:
                richness = np.sum(stack_masks.astype(int), axis=0)
                for val in range(1, max_pot + 1):
                    mask_r = (richness == val) & any_study_area & (~barrier_draw)
                    if np.any(mask_r):
                        c_rgb = get_pastel_palette_color(val, max_pot)
                        frame[mask_r] = c_rgb

            if np.any(barrier_draw):
                frame[barrier_draw] = colors.get("barrier", (0, 0, 0))

            img = Image.fromarray(frame)
            img_resized = img.resize((new_w, new_h), Image.Resampling.NEAREST)

            canvas = base_canvas.copy()
            canvas.paste(img_resized, (paste_x, paste_y))

            draw = ImageDraw.Draw(canvas)
            draw.text((width // 2, 30), f"Simulation Step: {step}", fill=(0, 0, 0), anchor="mm", font=font_title)

            pipe.stdin.write(canvas.tobytes())

        pipe.stdin.close()
        pipe.wait()

        print()
        print("GIF generated")