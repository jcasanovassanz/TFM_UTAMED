from pathlib import Path

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
}

EXCLUDE_FILES = {
    ".DS_Store",
}

MAX_DEPTH = 3  # cambia a 4 o 5 si quieres más detalle


def should_exclude(path: Path) -> bool:
    return path.name in EXCLUDE_DIRS or path.name in EXCLUDE_FILES


def build_tree(path: Path, prefix: str = "", depth: int = 0) -> list[str]:
    if depth > MAX_DEPTH:
        return []

    entries = sorted(
        [p for p in path.iterdir() if not should_exclude(p)],
        key=lambda p: (p.is_file(), p.name.lower())
    )

    lines = []

    for index, entry in enumerate(entries):
        connector = "└── " if index == len(entries) - 1 else "├── "
        lines.append(f"{prefix}{connector}{entry.name}")

        if entry.is_dir():
            extension = "    " if index == len(entries) - 1 else "│   "
            lines.extend(build_tree(entry, prefix + extension, depth + 1))

    return lines


if __name__ == "__main__":
    root = Path.cwd()
    output = [root.name + "/"]
    output.extend(build_tree(root))

    text = "\n".join(output)

    print(text)

    with open("estructura_proyecto.txt", "w", encoding="utf-8") as f:
        f.write(text)

    print("\nEstructura guardada en: estructura_proyecto.txt")