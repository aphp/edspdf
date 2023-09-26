import os
from pathlib import Path

import mkdocs.config
import mkdocs.plugins
import mkdocs.structure
import mkdocs.structure.files
import mkdocs.structure.nav
import mkdocs.structure.pages

try:
    from importlib.metadata import entry_points
except ImportError:
    from importlib_metadata import entry_points


def exclude_file(name):
    return name.startswith("assets/fragments/")


# Add the files from the project root

VIRTUAL_FILES = {}
REFERENCE_TEMPLATE = """
# `{ident}`
::: {ident}
    options:
        show_source: false
"""


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

    root = Path("edspdf")
    reference_nav = []
    for path in sorted(root.rglob("*.py")):
        module_path = path.relative_to(root.parent).with_suffix("")
        doc_path = Path("reference") / path.relative_to(root.parent).with_suffix(".md")
        # full_doc_path = Path("docs/reference/") / doc_path
        parts = list(module_path.parts)
        current = reference_nav
        for part in parts[:-1]:
            sub = next((item[part] for item in current if part in item), None)
            if sub is None:
                current.append({part: []})
                sub = current[-1][part]
            current = sub
        if parts[-1] == "__init__":
            parts = parts[:-1]
            doc_path = doc_path.with_name("index.md")
            current.append({"index.md": str(doc_path)})
        elif parts[-1] == "__main__":
            continue
        else:
            current.append({parts[-1]: str(doc_path)})
        ident = ".".join(parts)
        os.makedirs(doc_path.parent, exist_ok=True)
        VIRTUAL_FILES[str(doc_path)] = REFERENCE_TEMPLATE.format(ident=ident)

    for item in config["nav"]:
        if not isinstance(item, dict):
            continue
        key = next(iter(item.keys()))
        if not isinstance(item[key], str):
            continue
        if item[key].strip("/") == "reference":
            item[key] = reference_nav

    VIRTUAL_FILES["contributing.md"] = Path("contributing.md").read_text()
    VIRTUAL_FILES["changelog.md"] = Path("changelog.md").read_text()

    return mkdocs.structure.files.Files(
        [file for file in files if not exclude_file(file.src_path)]
        + [
            mkdocs.structure.files.File(
                file,
                config["docs_dir"],
                config["site_dir"],
                config["use_directory_urls"],
            )
            for file in VIRTUAL_FILES
        ]
    )


def on_nav(nav, config, files):
    def rec(node):
        if isinstance(node, list):
            return [rec(item) for item in node]
        if node.is_section and node.title == "Code Reference":
            return
        if isinstance(node, mkdocs.structure.nav.Navigation):
            return rec(node.items)
        if isinstance(node, mkdocs.structure.nav.Section):
            if (
                len(node.children)
                and node.children[0].is_page
                and node.children[0].is_index
            ):
                first = node.children[0]
                link = mkdocs.structure.nav.Link(
                    title=first.title,
                    url=first.url,
                )
                link.is_index = True
                first.title = "Overview"
                node.children.insert(0, link)
            return rec(node.children)

    rec(nav.items)


def on_page_read_source(page, config):
    if page.file.src_path in VIRTUAL_FILES:
        return VIRTUAL_FILES[page.file.src_path]
    return None


HREF_REGEX = r'href=(?:"([^"]*)"|\'([^\']*)|[ ]*([^ =>]*)(?![a-z]+=))'
# Maybe find something less specific ?
PIPE_REGEX = r"(?<=[^a-zA-Z0-9._-])eds[.][a-zA-Z0-9._-]*(?=[^a-zA-Z0-9._-])"


@mkdocs.plugins.event_priority(-1000)
def on_post_page(
    output: str,
    page: mkdocs.structure.pages.Page,
    config: mkdocs.config.Config,
):
    """
    1. Replace absolute paths with path relative to the rendered page
       This must be performed after all other plugins have run.
    2. Replace component names with links to the component reference

    Parameters
    ----------
    output
    page
    config

    Returns
    -------

    """

    autorefs = config["plugins"]["autorefs"]
    edspdf_factories_entry_points = {
        ep.name: ep.value for ep in entry_points()["edspdf_factories"]
    }

    def get_component_url(name):
        ep = edspdf_factories_entry_points.get(name)
        if ep is None:
            return None
        try:
            url = autorefs.get_item_url(ep.replace(":", "."))
        except KeyError:
            pass
        else:
            return url
        return None

    def get_relative_link(url):
        page_url = os.path.join("/", page.file.url)
        if url.startswith("/"):
            url = os.path.relpath(url, page_url)
        return url

    def replace_component_span(span):
        content = span.text
        if content is None:
            return
        link_url = get_component_url(content.strip("\"'"))
        if link_url is None:
            return
        a = lh.Element("a", href="/" + link_url)
        a.text = content
        span.text = ""
        span.append(a)

    def replace_component_names(root):
        # Iterate through all span elements
        spans = list(root.iter("span", "code"))
        for i, span in enumerate(spans):
            prev = span.getprevious()
            if span.getparent().tag == "a":
                continue
            # To avoid replacing default component name in parameter tables
            if prev is None or prev.text != "DEFAULT:":
                replace_component_span(span)
            # if span.text == "add_pipe":
            #     next_span = span.getnext()
            #     if next_span is None:
            #         continue
            #     next_span = next_span.getnext()
            #     if next_span is None or next_span.tag != "span":
            #         continue
            #     replace_component_span(next_span)
            #     continue
            # tokens = ["@", "factory", "="]
            # while True:
            #     if len(tokens) == 0:
            #         break
            #     if span.text != tokens[0]:
            #         break
            #     tokens = tokens[1:]
            #     span = span.getnext()
            #     while span is not None and (
            #       span.text is None or not span.text.strip()
            #     ):
            #         span = span.getnext()
            # if len(tokens) == 0:
            #     replace_component_span(span)

        # Convert the modified tree back to a string
        return root

    def replace_absolute_links(root):
        # Iterate through all a elements
        for a in root.iter("a"):
            href = a.get("href")
            if href is None or href.startswith("http"):
                continue
            a.set("href", get_relative_link(href))

        # Convert the modified tree back to a string
        return root

    # Replace absolute paths with path relative to the rendered page
    import lxml.html as lh

    root = lh.fromstring(output)
    root = replace_component_names(root)
    root = replace_absolute_links(root)
    return "".join(lh.tostring(e, encoding="unicode") for e in root)
