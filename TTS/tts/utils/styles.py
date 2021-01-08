import os
import json


def make_styles_json_path(out_path):
    """Returns conventional styles.json location."""
    return os.path.join(out_path, "styles.json")


def load_style_mapping(out_path):
    """Loads style mapping if already present."""
    try:
        if os.path.splitext(out_path)[1] == '.json':
            json_file = out_path
        else:
            json_file = make_styles_json_path(out_path)
        with open(json_file) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_style_mapping(out_path, style_mapping):
    """Saves style mapping if not yet present."""
    styles_json_path = make_styles_json_path(out_path)
    with open(styles_json_path, "w") as f:
        json.dump(style_mapping, f, indent=4)


def get_styles(items):
    """Returns a sorted, unique list of styles in a given dataset."""
    styles = {e[3] for e in items}
    return sorted(styles)
