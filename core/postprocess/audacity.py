from pathlib import Path

from pyannote.core.annotation import Annotation, Segment


def annotation_to_audacity_format(
    annotation: Annotation,
    output_path: Path,
    flag_short: float = 0.5,
    delimiter: str = "\t",
) -> None:
    """Write an Annotation as an Audacity TSV label file.

    Segments shorter than `flag_short` seconds are marked by appending '?'
    to the label, so they sort/highlight separately during human correction.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for segment, _, label in annotation.itertracks(yield_label=True):
            out_label = f"{label}?" if segment.duration < flag_short else str(label)
            f.write(
                f"{segment.start:.3f}{delimiter}{segment.end:.3f}{delimiter}{out_label}\n"
            )


def audacity_to_annotation_format(
    file_path: Path,
    uri: str | None = None,
    delimiter: str = "\t",
) -> Annotation:
    """Extract the timestamps and labels from an Audacity label file.

    By default, Audacity exports its labels as a tab-separated value (TSV) file,
    however this function will work with CSV or other delimiters as well.

    The expected input format is:
        <START_TIME><DELIMITER><END_TIME><DELIMITER><LABEL>

    Args:
        file_path (str): The path to the file, including its extension.
        delimiter (str, optional): The delimiter between values. Defaults to "\t".

    Returns:
        Annotation: The Audacity label file as a pyannote.core.Annotation class.
    """
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in map(str.strip, f):
            if not line:
                continue
            parts = line.split(delimiter)
            # Auto-detect indexed variant: 4 cols + first is integer
            if len(parts) == 4 and parts[0].isdigit():
                _, start, end, label = parts
            elif len(parts) >= 3:
                start, end, label = parts[0], parts[1], delimiter.join(parts[2:])
            else:
                continue
            records.append((Segment(float(start), float(end)), 0, label))
    return Annotation.from_records(records, uri=uri or Path(file_path).stem)
