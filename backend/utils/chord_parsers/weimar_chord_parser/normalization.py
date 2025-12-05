import regex as re
import math

from .definitions import QUALITY_PATTERNS, QUALITY_CLASSES

# Split a chord label into its components
def _parse_chord(label: str):

    # Check if label is empty or if first character is not a letter A-G
    if label is None or not isinstance(label, str) or (isinstance(label, float) and math.isnan(label)):
        return None, None, None, None, None
    
    label.strip()
    if not label or label[0] not in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
        return None, None, None, None, None
    
    #Extract first character for key root
    root = _read_root(label)
    bass = _read_root(label)

    label = label[len(root):]

    # Determine bass note if present (if / and A-G follows)
    if '/' in label:
        parts = label.rsplit('/', 1)  # split only at the last slash
        after_slash = parts[1].strip() if len(parts) > 1 else ""

        if after_slash and after_slash[0].upper() in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
            label = parts[0]
            bass = _read_root(after_slash)

    # Determine quality, quality_class, and extensions
    quality, quality_class, extensions = _parse_quality(label)

    # Return root, quality, quality_class, extensions, bass
    return root, quality, quality_class, extensions, bass


# Function to parse quality, quality_class, and extensions from the remaining label
def _parse_quality(label: str):

    # Normalize quality
    normalized_pattern = ''
    matched_pattern = ''

    # Find the longest matching quality pattern at the start of the label
    for pattern in sorted(QUALITY_PATTERNS.keys(), key=len, reverse=True):
        if label.startswith(pattern):
            normalized_pattern = QUALITY_PATTERNS[pattern]
            matched_pattern = pattern
            break
    
    quality = normalized_pattern
    remaining_label = label[len(matched_pattern):]

    # Initialize extensions
    extensions = []
    
    # Check for implied extensions based on matched pattern
    implied_ext = re.findall(r'(9|11|13)', matched_pattern)
    if implied_ext:
        for ext in implied_ext:
            if ext not in extensions:
                extensions.append(ext)
    
    # Extract additional extensions from the remaining label
    extensions.extend(_get_extensions(remaining_label))

    # Get quality class
    quality_class = QUALITY_CLASSES.get(quality, "")

    return quality, quality_class, extensions

# Helper function to tokenize and add extensions from a string
def _get_extensions(raw: str):
    if not raw:
        return []

    raw = raw.strip().lower().replace(" ", "")
    valid_numbers = {"5", "7", "9", "11", "13"}
    seen, extensions = set(), []

    # Match digits with optional trailing accidental (suffix style)
    for match in re.finditer(r'(5|7|9|11|13)([b#]?)', raw):
        num, acc = match.groups()

        if num not in valid_numbers:
            continue

        # Normalize to prefix-accidental style for output
        token = f"{acc}{num}" if acc else num

        if token not in seen:
            seen.add(token)
            extensions.append(token)

    return extensions

# Helper function to read root note
def _read_root(label: str) -> str:
    root = label[0]

    if root not in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
        return ""
    
    if len(label) > 1 and label[1] in ['b', '#']:
        root += label[1]
    return root