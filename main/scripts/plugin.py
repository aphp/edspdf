import os
import shutil
from pathlib import Path

import mkdocs

# Add the files from the project root

# Generate the code reference pages and navigation.
doc_reference = Path("docs/reference")
shutil.rmtree(doc_reference, ignore_errors=True)
os.makedirs(doc_reference, exist_ok=True)
root = Path("edspdf")
for path in sorted(root.rglob("*.py")):
    if "poppler_src" in str(path):
        continue
    module_path = path.relative_to(root.parent).with_suffix("")
    doc_path = path.relative_to(root.parent).with_suffix(".md")
    full_doc_path = doc_reference / doc_path
    parts = list(module_path.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue
    ident = ".".join(parts)
    os.makedirs(full_doc_path.parent, exist_ok=True)
    with open(full_doc_path, "w") as fd:
        print(f"# `{ident}`\n", file=fd)
        print("::: " + ident, file=fd)
        if root != "edspdf":
            print("    options:", file=fd)
            print("        show_source: false", file=fd)


def on_files(files: mkdocs.structure.files.Files, config: mkdocs.config.Config):
    """
    Recursively the navigation of the mkdocs config
    and recursively content of directories of page that point
    to directories.

    Parameters
    ----------
    config: mkdocs.config.Config
        The configuration object
    kwargs: dict
        Additional arguments
    """

    def get_nested_files(path):
        files = []
        for file in path.iterdir():
            if file.is_dir():
                index = file / "index.md"
                if index.exists():
                    # Get name from h1 heading in index
                    name = index.read_text().split("\n")[0].strip("# ")
                    if name.startswith("`edspdf"):
                        name = name[1:-1].split(".")[-1]
                    files.append({name: get_nested_files(file)})
                else:
                    title = file.name.replace("_", " ").replace("-", " ").title()
                    files.append({title: get_nested_files(file)})
            else:
                name = file.read_text().split("\n")[0].strip("# ")
                if name.startswith("`edspdf"):
                    name = name[1:-1].split(".")[-1]
                    files.append({name: str(file.relative_to(config["docs_dir"]))})
                else:
                    files.append(str(file.relative_to(config["docs_dir"])))
        return files

    def rec(tree):
        if isinstance(tree, list):
            return [rec(item) for item in tree]
        elif isinstance(tree, dict):
            return {k: rec(item) for k, item in tree.items()}
        elif isinstance(tree, str):
            if tree.endswith("/"):
                # We have a directory
                path = Path(config["docs_dir"]) / tree
                if path.is_dir():
                    return get_nested_files(path)
                else:
                    return tree
            else:
                return tree
        else:
            return tree

    config["nav"] = rec(config["nav"])
