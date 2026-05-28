import re
from pathlib import Path
import pandas as pd

def parse_time(t):
    """Converts either Unix float timestamps or HH:MM:SS.mmm strings into seconds."""
    
    t = t.strip().strip("'").strip('"')  # 👈 IMPORTANT FIX

    # Unix float seconds
    if re.match(r'^\d+(\.\d+)?$', t):
        return float(t)

    # HH:MM:SS.mmm format
    if re.match(r'^\d+:\d+:\d+(\.\d+)?$', t):
        h, m, s = t.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)

    raise ValueError(f"Unknown time format: {t}")

def process_txt(filepath):
    """
    Parses trial data from a txt file containing trial information.
    Returns a pandas DataFrame with transition data and trial information.
    
    Args:
        filepath: Path to txt file or directory containing txt file
    
    Returns:
        pd.DataFrame with columns: trial_num, Start_node, Next_node, Time_start,
                                   Time_end, Diff (s), Length (m), Velocity (m/s),
                                   Trial_start_time, Trial_end_time
    """
    all_transitions = []
    trial_start_times = {}  # Store the start time for each trial
    
    # Convert to Path object if string
    filepath = Path(filepath)
    
    # If filepath is a directory, find the .txt file in it
    if filepath.is_dir():
        txt_files = list(filepath.glob('*.txt'))
        if not txt_files:
            raise FileNotFoundError(f"No .txt files found in {filepath}")
        filepath = txt_files[0]  # Use the first .txt file found
    
    content = filepath.read_text()
    
    # Extract header info
    header_match = re.search(r'Rat number: (\d+)\s*,\s*Date: (\d+)', content)
    rat_number = int(header_match.group(1)) if header_match else None
    date = header_match.group(2) if header_match else None
    
    # Find all trial sections
    trial_pattern = r'Summary Trial (\d+)\r?\nTrial End \(Sync Seconds\): ([^\n]+)\r?\n(.*?)(?=Summary Trial|\Z)'
    
    for trial_match in re.finditer(trial_pattern, content, re.DOTALL):
        trial_num = int(trial_match.group(1))
        trial_end_time = parse_time(trial_match.group(2))
        trial_content = trial_match.group(3)

        # Split into lines and skip header row
        lines = trial_content.strip().split('\n')[1:]
        trial_start_time = None

        for line in lines:
            line = line.strip()
            # Skip empty lines or header labels
            if not line or 'Start-Next' in line:
                continue

            # Parse a transition row:
            # ('nodeA', 'nodeB') (time_start, time_end) diff length velocity
            match = re.match(
                r"\('(\d+)',\s*'(\d+)'\)\s+\(([^,]+),\s*([^)]+)\)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
                line
            )

            if match:
                # Convert timestamps using unified parser
                time_start = parse_time(match.group(3))
                time_end = parse_time(match.group(4))

                # First transition defines trial start time
                if trial_start_time is None:
                    trial_start_time = time_start
                    trial_start_times[trial_num] = trial_start_time

                # Store parsed transition
                all_transitions.append({
                    'Trial_Num': trial_num,
                    'Start_node': match.group(1),
                    'Next_node': match.group(2),
                    'Time_start': time_start,
                    'Time_end': time_end,
                    'Diff_s': float(match.group(5)),
                    'Length_m': float(match.group(6)),
                    'Velocity_m_s': float(match.group(7)),
                    'Trial_start_time': trial_start_time,
                    'Trial_end_time': trial_end_time,
                })

    return  pd.DataFrame(all_transitions)


if __name__ == "__main__":
    filepath = Path(r"s:\data\Rat1\20201010\20201010_Rat1.txt")
    trial_df = process_txt(filepath)
    print(trial_df)