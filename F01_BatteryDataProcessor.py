# Battery Data Processor

# Import necessary dependencies
import os
import gc
import re
import numpy as np
import pandas as pd
import time
import json
import shutil
from tqdm.auto import tqdm
import h5py

# Configuration settings
DATA_DIR        = ".\dataset"
BATCH_MAT       = "BatteryDataset.mat"
CACHE_ROOT      = ".\dataset_cache"
MAKE_LOCAL_COPY = False

# Real channel IDs for the batteries, provided by user
REAL_CHANNEL_IDS = ["03_1", "04", "08", "11_1", "11_2", "12", "13", "14", "15", "16"]

def tstamp(msg=""):
    """Print timestamp with optional message"""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def load_batch_as_dicts(mat_path):
    """Read MAT file using h5py and convert to dictionary list, properly handling MAT-File v7.3 format"""
    cells = []
    
    # Open MAT file with h5py
    with h5py.File(mat_path, 'r') as f:
        if 'batch' not in f:
            raise RuntimeError("'batch' group not found in MAT file")
        
        batch = f['batch']
        
        # Check for necessary fields
        required_fields = ['channel_id', 'cycle_life', 'summary']
        for field in required_fields:
            if field not in batch:
                raise RuntimeError(f"Required field missing in batch group: {field}")
        
        # Get number of battery cells (from channel_id field)
        channel_id_data = batch['channel_id']
        num_cells = channel_id_data.shape[0] if channel_id_data.size > 0 else 0
        
        # Process all cells
        max_cells = num_cells
        print(f"Found {num_cells} battery cells, processing all {max_cells}")
        
        # Iterate through each battery cell
        for i in range(max_cells):
            try:
                # Create battery cell dictionary
                cell_dict = {
                    'cell_id': i + 1,
                    'channel_id': np.nan,
                    'policy_readable': 'Unknown',
                    'policy': 'Unknown',
                    'cycle_life': np.nan,
                    'summary': {
                        'cycle': np.array([], dtype=float),
                        'QDischarge': np.array([], dtype=float),
                        'cycle_life': np.nan
                    },
                    'cycles': []
                }
                
                # Map battery index to real channel_id from configuration
                if i < len(REAL_CHANNEL_IDS):
                    channel_id = REAL_CHANNEL_IDS[i]
                    print(f"  Battery {i+1} using real channel_id: {channel_id}")
                else:
                    channel_id = f"cell{i+1}"
                    print(f"  Battery {i+1} using default channel_id: {channel_id}")
                
                cell_dict['channel_id'] = channel_id
                
                # Extract cycle_life
                if 'cycle_life' in batch:
                    try:
                        cycle_ref = batch['cycle_life'][i, 0]
                        if isinstance(cycle_ref, h5py.Reference):
                            cycle_data = f[cycle_ref]
                            cycle_life = cycle_data[()]
                            if np.isfinite(cycle_life):
                                cell_dict['cycle_life'] = float(cycle_life)
                                cell_dict['summary']['cycle_life'] = float(cycle_life)
                    except:
                        pass
                
                # Extract policy_readable - Fixed to properly read string instead of ASCII codes
                if 'policy_readable' in batch:
                    try:
                        policy_ref = batch['policy_readable'][i, 0]
                        if isinstance(policy_ref, h5py.Reference):
                            policy_data = f[policy_ref]
                            # Properly handle string conversion
                            policy_array = policy_data[()]
                            if isinstance(policy_array, np.ndarray):
                                # Check if it's an array of characters (ASCII codes)
                                if policy_array.dtype.kind in 'iu':
                                    # Convert ASCII codes to string
                                    policy_str = ''.join(chr(int(c)) for c in policy_array.flatten() if int(c) > 0)
                                else:
                                    policy_str = str(policy_array)
                            elif isinstance(policy_array, bytes):
                                policy_str = policy_array.decode('utf-8')
                            else:
                                policy_str = str(policy_array)
                            cell_dict['policy_readable'] = policy_str
                            cell_dict['policy'] = policy_str
                    except:
                        pass
                
                # Extract summary data
                if 'summary' in batch:
                    try:
                        summary_ref = batch['summary'][i, 0]
                        if isinstance(summary_ref, h5py.Reference):
                            summary_group = f[summary_ref]
                            
                            # Extract key fields from summary - only cycle and QDischarge as requested
                            for summary_field in ['cycle', 'QDischarge']:
                                if summary_field in summary_group:
                                    try:
                                        field_data = summary_group[summary_field]
                                        if isinstance(field_data, h5py.Dataset):
                                            data_array = field_data[()]
                                            if isinstance(data_array, np.ndarray):
                                                cell_dict['summary'][summary_field] = data_array.flatten()
                                    except:
                                        pass
                    except:
                        pass
                
                # Extract cycles data
                cell_dict['cycles'] = []
                try:
                    if 'cycles' in batch and len(batch['cycles'].shape) >= 2 and i < batch['cycles'].shape[0]:
                        cycles_ref = batch['cycles'][i, 0]
                        
                        if isinstance(cycles_ref, h5py.Reference):
                            cell_structure = f[cycles_ref]
                            
                            if hasattr(cell_structure, '__len__'):
                                required_fields = ['t', 'V', 'I', 'Q', 'CO', 'CO2', 'C2H4']
                                
                                # Always check for cycle field regardless of required_fields
                                if 'cycle' in cell_structure:
                                    # Process all cycles with cycle data
                                    max_cycles = len(cell_structure['cycle'])
                                    
                                    for cycle_idx in range(max_cycles):
                                        current_cycle_dict = {'cycle_number': cycle_idx + 1}
                                        
                                        # Extract cycle number from cycle field
                                        cycle_ref = cell_structure['cycle'][cycle_idx]
                                        if isinstance(cycle_ref, np.ndarray) and cycle_ref.size > 0:
                                            ref_element = cycle_ref[0]
                                            if isinstance(ref_element, h5py.Reference):
                                                try:
                                                    cycle_obj = f[ref_element]
                                                    if isinstance(cycle_obj, h5py.Dataset):
                                                        cycle_data = cycle_obj[:]
                                                        if isinstance(cycle_data, np.ndarray):
                                                            cycle_num = float(cycle_data.flatten()[0])
                                                            current_cycle_dict['cycle'] = int(cycle_num)
                                                except Exception:
                                                    pass
                                        
                                        # Extract other required fields
                                        for field in required_fields:
                                            if field in cell_structure and cycle_idx < len(cell_structure[field]):
                                                field_ref = cell_structure[field][cycle_idx]
                                                if isinstance(field_ref, np.ndarray) and field_ref.size > 0:
                                                    ref_element = field_ref[0]
                                                    if isinstance(ref_element, h5py.Reference):
                                                        try:
                                                            field_obj = f[ref_element]
                                                            if isinstance(field_obj, h5py.Dataset):
                                                                data = field_obj[:]
                                                                if isinstance(data, np.ndarray):
                                                                    field_data = np.array(data, dtype=np.float64).flatten()
                                                                    current_cycle_dict[field] = field_data
                                                        except Exception:
                                                            pass
                                        
                                        if len(current_cycle_dict) > 1:  # Only containing cycle_number is not valid data
                                            cell_dict['cycles'].append(current_cycle_dict)
                                elif any(field in cell_structure for field in required_fields):
                                    # Fallback to original logic if cycle field not available
                                    if hasattr(cell_structure, 'keys') and 't' in cell_structure:
                                        max_cycles = len(cell_structure['t'])
                                    else:
                                        max_cycles = len(cell_structure) if hasattr(cell_structure, '__len__') else 10
                                    
                                    for cycle_idx in range(max_cycles):
                                        current_cycle_dict = {'cycle_number': cycle_idx + 1}
                                        
                                        for field in required_fields:
                                            if field in cell_structure and cycle_idx < len(cell_structure[field]):
                                                field_data = None
                                                field_ref = cell_structure[field][cycle_idx]
                                                if isinstance(field_ref, np.ndarray) and field_ref.size > 0:
                                                    ref_element = field_ref[0]
                                                    if isinstance(ref_element, h5py.Reference):
                                                        try:
                                                            field_obj = f[ref_element]
                                                            if isinstance(field_obj, h5py.Dataset):
                                                                data = field_obj[:]
                                                                if isinstance(data, np.ndarray):
                                                                    field_data = np.array(data, dtype=np.float64).flatten()
                                                        except Exception:
                                                            pass
                                                if field_data is not None:
                                                    current_cycle_dict[field] = field_data
                                        
                                        if len(current_cycle_dict) > 1:  # Only containing cycle_number is not valid data
                                            cell_dict['cycles'].append(current_cycle_dict)
                except Exception:
                    pass

                # Always add the cell to the list since we're using string channel_id
                cells.append(cell_dict)
            except Exception:
                continue
    
    print(f"Successfully processed {len(cells)} battery cells")
    return cells

def get_field(d, key):
    if isinstance(d, list) and len(d) == 1 and isinstance(d[0], dict): d = d[0]
    if not isinstance(d, dict): return None
    for k in (key, key.lower(), key.capitalize()):
        if k in d: return d[k]
    for k in d.keys():
        if k.lower() == key.lower(): return d[k]
    return None

def to_1d(x):
    if x is None: return np.array([], dtype=float)
    if isinstance(x, list):
        try: return np.array(x, dtype=float).reshape(-1)
        except: return np.array([np.nan if v is None else float(v) for v in x], dtype=float).reshape(-1)
    a = np.array(x)
    if a.dtype == object:
        out = []
        for v in a.reshape(-1):
            if isinstance(v, (list, tuple, np.ndarray)):
                vv = np.array(v).reshape(-1)
                out.append(np.nan if vv.size == 0 else float(vv[0]))
            else:
                try: out.append(float(v))
                except: out.append(np.nan)
        return np.array(out, dtype=float)
    return a.reshape(-1)

def to_scalar(x):
    if isinstance(x, (list, tuple)): return x[0] if x else np.nan
    if isinstance(x, np.ndarray):     return x.reshape(-1)[0] if x.size else np.nan
    return x

def slugify_policy(s):
    if s is None: return ""
    s = str(s)
    s = re.sub(r"\s+", "_", s)
    s = s.replace("/", "_").replace("%", "per").replace("(", "").replace(")", "")
    s = re.sub(r"_+", "_", s).strip("_")
    return s

# locate and optionally copy the .mat to local
def setup_paths():
    """Set up data paths and verify file existence"""
    batch_path_drive = os.path.join(DATA_DIR, BATCH_MAT)
    
    # Check and create data directory
    if not os.path.exists(DATA_DIR):
        # Create severson directory and inform user
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"Warning: Created data directory, please ensure MAT file is placed in: {DATA_DIR}")
    
    # Verify MAT file exists
    assert os.path.exists(batch_path_drive), f"Batch not found: {batch_path_drive}"
    
    # Handle local copy logic
    if MAKE_LOCAL_COPY:
        # Use temporary directory instead of Linux-style /content directory
        import tempfile
        temp_dir = tempfile.gettempdir()
        local_path = os.path.join(temp_dir, os.path.basename(batch_path_drive))
        if not os.path.exists(local_path):
            tstamp("Copying mat to local for faster reads")
            tic = time.perf_counter()
            shutil.copy2(batch_path_drive, local_path)
            tstamp(f"Copy done in {time.perf_counter()-tic:.1f}s, size about {os.path.getsize(local_path)/1e9:.2f} GB")
        return local_path
    else:
        return batch_path_drive

# load and convert to dicts
def load_batch_data(batch_path):
    """Load batch battery data and set up output directory"""
    tstamp(f"Loading batch {os.path.basename(batch_path)}")
    tic = time.perf_counter()
    batch = load_batch_as_dicts(batch_path)
    tstamp(f"Parsed {len(batch)} cells in {time.perf_counter()-tic:.1f}s")
    
    # Set up output directory - directly use CACHE_ROOT as requested
    OUT_DIR = CACHE_ROOT
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Simplified batch_tag since all batteries are in one batch
    batch_tag = "battery"
    
    return batch, batch_tag, OUT_DIR

def process_cell_cycles(cell, ch, OUT_DIR):
    # Create cycles subdirectory with cellxx_cycles format
    cycles_dir = os.path.join(OUT_DIR, f"cell{ch}_cycles")
    os.makedirs(cycles_dir, exist_ok=True)
    
    cycles_info = []
    # Check availability of cycles data
    if 'cycles' in cell and cell['cycles']:
        total_cycles_in_data = len(cell['cycles'])
        # Get total cycle count from summary
        summ = get_field(cell, "summary")
        if isinstance(summ, list) and len(summ) == 1 and isinstance(summ[0], dict):
            summ = summ[0]
        cycle_array = to_1d(get_field(summ, "cycle"))
        total_cycles_in_summary = len(cycle_array) if cycle_array is not None else 0
        
        print(f"  Starting to process cycle data for battery {ch}: {total_cycles_in_data} cycles with detailed curve data, {total_cycles_in_summary} cycles recorded in summary")
        
        # Process all cycle data, no longer limiting quantity
        for cycle_idx, cycle_data in enumerate(cell['cycles']):
            # Get cycle number, prioritize 'cycle' field if available, otherwise use cycle_number or index
            cycle_num = cycle_data.get('cycle', cycle_data.get('cycle_number', cycle_idx + 1))
            try:
                cycle_num = int(cycle_num)
            except (ValueError, TypeError):
                cycle_num = cycle_idx + 1
            
            # Save cycle data using three-digit format
            cycle_npz_path = os.path.join(cycles_dir, f"cycle_{int(cycle_num):03d}.npz")
            
            # Save current cycle curve data
            cycle_data_to_save = {}
            data_available = []
            
            # Key fields to process
            fields = ['t', 'V', 'I', 'Q', 'CO', 'CO2', 'C2H4']
            
            try:
                for field in fields:
                    field_data = get_field(cycle_data, field)
                    if field_data is not None:
                        # Use existing to_1d function to ensure data is correct
                        final_data = to_1d(field_data)
                        
                        # Ensure data is float64 type
                        final_data = final_data.astype(np.float64)
                        # Filter out invalid data
                        final_data = final_data[np.isfinite(final_data)]
                        
                        if len(final_data) > 0:
                            cycle_data_to_save[field] = final_data
                            data_available.append(True)
                        else:
                            data_available.append(False)
                            if cycle_idx < 5 or (total_cycles_in_data > 20 and cycle_idx % 10 == 0) or cycle_idx == total_cycles_in_data - 1:
                                print(f"    Skipping {field}: no valid data")
                    else:
                        data_available.append(False)
                        if cycle_idx < 5 or (total_cycles_in_data > 20 and cycle_idx % 10 == 0) or cycle_idx == total_cycles_in_data - 1:
                            print(f"    Skipping {field}: field not found")
            except Exception as e:
                print(f"    Error processing cycle {cycle_num}: {str(e)}")
                data_available = [False] * len(fields)
            
            # Only save if there's enough valid data
            if cycle_data_to_save and len(cycle_data_to_save) >= 2:  # Ensure at least two fields
                np.savez_compressed(cycle_npz_path, **cycle_data_to_save)
                cycles_info.append({
                    'cycle_number': cycle_num,
                    'file_path': os.path.relpath(cycle_npz_path, OUT_DIR),
                    'available_fields': list(cycle_data_to_save.keys())
                })
                if cycle_idx < 5 or (total_cycles_in_data > 20 and cycle_idx % 100 == 0) or cycle_idx == total_cycles_in_data - 1:
                    print(f"  Saved cycle {cycle_num} ({cycle_idx+1}/{total_cycles_in_data}) to {os.path.basename(cycle_npz_path)}")
            else:
                if cycle_idx < 5 or (total_cycles_in_data > 20 and cycle_idx % 100 == 0) or cycle_idx == total_cycles_in_data - 1:
                    print(f"  Skipping cycle {cycle_num} ({cycle_idx+1}/{total_cycles_in_data}): insufficient valid data")
        
        print(f"  Battery {ch} processing completed, successfully saved detailed curve data for {len(cycles_info)} cycles")
    else:
        print(f"  Battery {ch} has no available cycle data")
    
    return cycles_info

def save_cell_metadata(ch, batch_tag, policy, cycles_info, OUT_DIR, npz_path):
    """Save battery cell metadata"""
    meta_path = npz_path.replace(".npz", ".json")
    with open(meta_path, "w") as f:
        json.dump({
            "name": f"{batch_tag}_{slugify_policy(policy)}_CH{ch}",
            "policy": policy, 
            "channel_id": ch,
            "cycles_available": len(cycles_info) > 0,
            "number_of_cycles": len(cycles_info),
            "cycles_info": cycles_info
        }, f)
    return meta_path

def process_cell(cell, i, batch_tag, OUT_DIR):
    """Process individual battery cell data"""
    summ = get_field(cell, "summary")
    if isinstance(summ, list) and len(summ) == 1 and isinstance(summ[0], dict):
        summ = summ[0]

    # Extract basic data - only cycle and QDischarge as requested
    cycle = to_1d(get_field(summ, "cycle"))
    qd = to_1d(get_field(summ, "QDischarge"))

    # Get battery information
    policy = get_field(cell, "policy_readable") or get_field(cell, "policy") or ""
    ch = get_field(cell, "channel_id")
    if ch is None:
        ch = f"cell{i}"

    cycle_life_gt = get_field(cell, "cycle_life")
    if cycle_life_gt is None and isinstance(summ, dict):
        cycle_life_gt = get_field(summ, "cycle_life")
    cycle_life_gt = float(to_scalar(cycle_life_gt)) if cycle_life_gt is not None else np.nan

    # Save summary data, including metadata information
    npz_path = os.path.join(OUT_DIR, f"cell{ch}.npz")
    np.savez_compressed(npz_path, 
                      # Metadata information
                      batch_tag=batch_tag,
                      channel_id=ch,
                      policy=policy,
                      cycle_life_gt=cycle_life_gt,
                      # Original data fields - only cycle and QDischarge as requested
                      cycle=cycle,
                      QDischarge=qd)

    # Process cycle data
    cycles_info = process_cell_cycles(cell, ch, OUT_DIR)

    # Save metadata
    meta_path = save_cell_metadata(ch, batch_tag, policy, cycles_info, OUT_DIR, npz_path)

    # Prepare index and result data
    index_entry = {
        "batch_tag": batch_tag, "batch_file": BATCH_MAT, "cell_index": i, "channel_id": ch,
        "policy": policy, "cache_npz": npz_path, "cache_meta": meta_path,
        "cycle_life_gt": cycle_life_gt, "cycles_recorded": int(len(cycle))
    }

    result_entry = {
        "batch_tag": batch_tag, 
        "channel_id": ch, 
        "policy": policy,
        "cycle_life_gt": cycle_life_gt
    }

    return index_entry, result_entry

# iterate first N cells
def process_all_cells(batch, batch_tag, OUT_DIR):
    """Process all battery cells"""
    rows_index, rows_results = [], []
    n_total = len(batch)
    tstamp(f"Processing all {n_total} cell(s)")

    for i, cell in enumerate(tqdm(batch, desc="Cells"), start=1):
        index_entry, result_entry = process_cell(cell, i, batch_tag, OUT_DIR)
        rows_index.append(index_entry)
        rows_results.append(result_entry)

    # Clean up memory
    del batch
    gc.collect()

    return rows_index, rows_results

def main_process():
    """Main processing flow function"""
    # Set up paths
    batch_path = setup_paths()
    
    # Load data
    batch, batch_tag, OUT_DIR = load_batch_data(batch_path)
    
    # Process all battery cells
    rows_index, rows_results = process_all_cells(batch, batch_tag, OUT_DIR)
    
    return rows_index, rows_results, OUT_DIR

# Main entry point
if __name__ == "__main__":
    # Directly execute main processing, process all cycle data for all batteries
    print("Starting to process all cycle data for all batteries...")
    rows_index, rows_results, OUT_DIR = main_process()
    print(f"\nProcessing completed!")
    print(f"Successfully processed {len(rows_index)} battery cells")
    print(f"Data has been saved to directory: {OUT_DIR}")
    
    # Display all battery information
    for i in range(len(rows_index)):  # Display information for all batteries
        cell_info = rows_index[i]
        meta_path = cell_info['cache_meta']
        try:
            with open(meta_path, 'r') as f:
                import json
                meta_data = json.load(f)
            cycles_count = meta_data.get('number_of_cycles', 0)
            print(f"\nBattery {cell_info['channel_id']}:")
            print(f"  Policy: {cell_info['policy']}")
            print(f"  Recorded cycles: {cell_info['cycles_recorded']}")
            print(f"  Saved cycle data: {cycles_count}")
        except Exception as e:
            print(f"Error reading metadata for battery {cell_info['channel_id']}: {e}")