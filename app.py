from flask import Flask, render_template, request
from flask_caching import Cache
import search_logic
import pandas as pd


app = Flask(__name__)
config = {
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": 300,
}
app.config.from_mapping(config)
cache = Cache(app)


@app.route("/")
def hello_world():
    return render_template("search.html")


@app.route("/search", methods=["GET"])
@cache.cached(query_string=True)
def search():
    search = request.args.get("input")
    results = search_logic.search_results(search)

    df = pd.DataFrame(results)
    return render_template(
        "search.html",
        table=df.to_html(index=False, render_links=True, justify="center")
        .replace("\\n", "<br>")
        .replace("\\t", " "),
    )
