# _ext/spelling_aliases.py

from sphinx.builders.html import StandaloneHTMLBuilder

SPELLING_MAP = {
    "optimiser": "optimizer",
    "optimising": "optimizing",
    "optimisation": "optimization",
    "colour": "color",
    "organise": "organize",
    "organisation": "organization",
    "analyse": "analyze",
    "behaviour": "behavior",
    "modelled": "modeled",
    "modelling": "modeling",
}


def add_spelling_aliases(app, exception):
    if exception:
        return

    builder = app.builder

    # Only operate on HTML
    if not isinstance(builder, StandaloneHTMLBuilder):
        return

    # ---- CASE 1: Old Sphinx (≤5.x) ----
    # builder.indexer exists
    indexer = getattr(builder, "indexer", None)

    # ---- CASE 2: New Sphinx (≥6.x) ----
    # env.get_search_indexer(...) exists
    if indexer is None:
        get_indexer = getattr(app.env, "get_search_indexer", None)
        if callable(get_indexer):
            indexer = get_indexer(builder)

    if indexer is None:
        app.warn("spelling_aliases: no search indexer available")
        return

    # indexer._index holds the real data
    index = getattr(indexer, "_index", None)
    if index is None:
        return

    terms = index.get("terms", {})

    # Inject aliases
    for brit, amer in SPELLING_MAP.items():
        if brit in terms and amer not in terms:
            terms[amer] = terms[brit]

    # ---- Write index file ----
    # Old Sphinx:
    if hasattr(builder, "write_searchindex"):
        builder.write_searchindex()
    # New Sphinx:
    elif hasattr(indexer, "dump"):
        indexer.dump(app, exception)


def setup(app):
    app.connect("build-finished", add_spelling_aliases)
    return {
        "version": "1.2",
        "parallel_read_safe": True,
    }
