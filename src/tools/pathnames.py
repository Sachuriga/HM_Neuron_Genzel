from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import sys

# convertes a Path to a string and simplifies it
# removes data + rat part before key part (key is a string of the notable part of the filename, indicating its use)
# removes the filename extension .ext (e.g. .csv, .txt)
def normalize_key(path: Path) -> str:
    """
    For example:
      20200914_Rat1_Coordinates_Full.csv -> coordinates_full
      20200914_Rat1_framewise_ts.csv     -> framewise_ts
      stitched_framewise_seconds.csv     -> stitched_framewise_seconds
    """
    # returns final file name (excl. .ext)
    stem = path.stem

    # splits the string into a list of strings delimited by the first two "_" occurences
    # changing the filename structure might break this! -> in that case will return full stem
    parts = stem.split("_", 2)
    if len(parts) == 3 and parts[0].isdigit() and parts[1].lower().startswith("rat"):
        return parts[2].lower()

    return stem.lower()

# ensures no duplicate keys are added
# files can be different but the keys can turn out to be the same, this prevents not noticing that
def add_unique(d: dict[str, Path], path: Path) -> None:
    key = normalize_key(path)
    if key in d:
        raise ValueError(f"Duplicate normalized key '{key}': {d[key].name} and {path.name}")
    d[key] = path

# create class for the different keys and paths of the files, one dictonary per .ext type
# folders is just a list of paths because there keys dont really matter
@dataclass
class SessionPaths:
    log_paths: dict[str, Path] = field(default_factory=dict)
    txt_paths: dict[str, Path] = field(default_factory=dict)
    csv_paths: dict[str, Path] = field(default_factory=dict)
    numpy_paths: dict[str, Path] = field(default_factory=dict)
    folder_paths: list[Path] = field(default_factory=list)
    maze_merged_rec: Optional[Path] = None

    # used to show the recognized files and folder structure
    def __str__(self):
        # name: str(name_Of_Dictonary)
        # d: dict[str(key), Path(path_to_file)]
        def fmt_dict(name, d):
            if not d:
                return f"{name}: None"
            lines = [f"{name}:"]
            for _, path in d.items():
                lines.append(f"      - {path.name}")
            return "\n".join(lines)

        # since numpy files are in a different folder this function shows the folder and numpy files in it
        def fmt_folders_with_numpy(folders, numpy_dict):
            if not folders:
                return "      Folders: None"

            lines = ["      Folders:"]
            for folder in folders:
                lines.append(f"      {folder.name}/")

                np_files = [p for p in numpy_dict.values() if p.parent == folder]
                if np_files:
                    for p in np_files:
                        lines.append(f"      - {p.name}")
                else:
                    lines.append("      (no .npy files)")

            return "\n".join(lines)

        return "\n\n".join([
            fmt_dict("      LOG files", self.log_paths),
            fmt_dict("      TXT files", self.txt_paths),
            fmt_dict("      CSV files", self.csv_paths),
            fmt_folders_with_numpy(self.folder_paths, self.numpy_paths),
            f"      Maze REC: {self.maze_merged_rec.name if self.maze_merged_rec else 'None'}",
            f" --------------------------------- "
        ])

# skips dot-files: hidden files and macOS AppleDouble sidecars (._*), which
# appear alongside real data on network/exFAT shares. Real data files never
# start with a dot, so these are always junk and must not be globbed in.
def _is_hidden(p: Path) -> bool:
    return p.name.startswith(".")

# finds the paths to files in the working directory for each .ext type
# sorts the loaded files alphabetically
# then adds them per .ext type making sure it is recognized as unique
def find_paths(work_dir: Path) -> SessionPaths:
    if not work_dir.exists():
        sys.exit(f"Error: work directory does not exist: {work_dir}")

    log_paths: dict[str, Path] = {}
    for p in sorted(work_dir.glob("*.log")):
        if _is_hidden(p): continue
        add_unique(log_paths, p)

    txt_paths: dict[str, Path] = {}
    for p in sorted(work_dir.glob("*.txt")):
        if _is_hidden(p): continue
        add_unique(txt_paths, p)

    csv_paths: dict[str, Path] = {}
    for p in sorted(work_dir.glob("*.csv")):
        if _is_hidden(p): continue
        add_unique(csv_paths, p)

    folder_paths = sorted([p for p in work_dir.iterdir() if p.is_dir() and not _is_hidden(p)])

    numpy_paths: dict[str, Path] = {}
    for folder in folder_paths:
        for p in sorted(folder.glob("*.npy")):
            if _is_hidden(p): continue
            add_unique(numpy_paths, p)

    maze_merged_rec = None
    for rec_path in work_dir.glob("*.rec"):
        if _is_hidden(rec_path): continue
        if "maze_merged" in rec_path.name.lower():
            maze_merged_rec = rec_path
            break

    return SessionPaths(
        log_paths=log_paths,
        txt_paths=txt_paths,
        csv_paths=csv_paths,
        numpy_paths=numpy_paths,
        folder_paths=folder_paths,
        maze_merged_rec=maze_merged_rec,
    )

# Parse a folder name string to extract session identifier and notes
# Looks for 8 consecutive digits (typically a date like 20210612)
# If found: session_part = everything up to and including the 8 digits, note = everything after
# If not found: session_part = entire folder name, note = empty string
def parse_folder_name(folder_name: str) -> tuple[str, str]:
    """
    Parse a folder name to extract the session identifier and notes.
    
    Looks for 8 consecutive digits (date) in the folder name.
    - If found: session part includes everything up to and including those 8 digits
    - If not found: session part is the entire folder name
    
    Args:
        folder_name: The name of the folder to parse
    
    Returns:
        tuple[str, str]: (session_identifier, notes)
        
    Examples:
        "20210612_Rat5" -> ("20210612", "_Rat5")
        "session_1" -> ("session_1", "")
        "20210612_missing_lfp" -> ("20210612", "_missing_lfp")
        "some_folder" -> ("some_folder", "")
    """
    # Look for 8 consecutive digits (date) anywhere in the folder name
    for i in range(len(folder_name) - 7):
        if all(c.isdigit() for c in folder_name[i:i+8]):
            # Found 8 consecutive digits
            session_part = folder_name[:i+8]
            note = folder_name[i+8:]  # Keep everything after to allow perfect reconstruction
            return session_part, note
    
    # If no 8 digits found, return the whole name as session_part
    return folder_name, ""

# function to read multiple folders containing data for each session
# Returns a tuple of (session_folders, notes) where session_folders contains paths with the session identifier
# and notes contains any text after the session identifier
# If no 8-digit pattern is found, the full folder name becomes the session identifier
def find_session_folders(sessions_dir: Path) -> tuple[list[Path], list[str]]:
    if not sessions_dir.exists():
        sys.exit(f"Error: sessions directory does not exist: {sessions_dir}")

    session_folders_raw = sorted(
        p for p in sessions_dir.iterdir()
        if p.is_dir()
    )

    session_folders = []
    notes = []
    
    for folder in session_folders_raw:
        session_name, note = parse_folder_name(folder.name)
        session_path = folder.parent / session_name
        session_folders.append(session_path)
        notes.append(note)

    return session_folders, notes
