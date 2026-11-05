import pathlib
import requests

TITLES = [
    "Apollo 11", "Voyager 1", "Hubble Space Telescope",
    "International Space Station", "James Webb Space Telescope",
    "Sputnik 1", "Artemis program", "Mars Pathfinder",
]

out = pathlib.Path("docs")
out.mkdir(exist_ok=True)

for title in TITLES:
    r = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={"action": "query", "prop": "extracts", "explaintext": 1,
                "titles": title, "format": "json", "redirects": 1},
        headers={"User-Agent": "RAG-assignment/1.0 (student project)"},
        timeout=30,
    )
    page = next(iter(r.json()["query"]["pages"].values()))
    text = page["extract"]
    (out / f"{title.replace(' ', '_')}.txt").write_text(text, encoding="utf-8")
    print(f"Saved {title}: {len(text.split())} words")
