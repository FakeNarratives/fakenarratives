import requests
import json

def get_subclasses(query):
    endpoint_url = "https://query.wikidata.org/sparql"
    params = {
        "query": query,
        "format": "json"
    }
    response = requests.get(endpoint_url, params=params)
    if response.status_code == 200:
        results = response.json()
        return [binding["subclass"]["value"].split("/")[-1] for binding in results["results"]["bindings"]]
    else:
        print(f"Error: {response.status_code}")
        return []

# Define SPARQL queries
queries = {
    "organization": """
    SELECT DISTINCT ?subclass WHERE {
        ?subclass wdt:P279/wdt:P279?/wdt:P279? wd:Q43229.
        ?subclass wikibase:statements ?count.
        FILTER(?count > 10)
    }
    """,
    "occurrence": """
    SELECT DISTINCT ?subclass WHERE {
        ?subclass wdt:P279/wdt:P279? wd:Q1190554.
        ?subclass wikibase:statements ?count.
        FILTER(?count > 10)
    }
    """,
    "event": """
    SELECT DISTINCT ?subclass WHERE {
        ?subclass wdt:P279/wdt:P279? wd:Q1656682.
        ?subclass wikibase:statements ?count.
        FILTER(?count > 10)
    }
    """
}

# Get subclasses for each category
subclasses = {}
for category, query in queries.items():
    subclasses[category] = get_subclasses(query)
    print(f"{category.capitalize()} subclasses: {len(subclasses[category])}")


# Combine occurrence and event subclasses
subclasses["occurrence_event"] = list(set(subclasses["occurrence"] + subclasses["event"]))


# Save subclasses to JSON files
with open("text_analysis/wikidata_ids/organization.json", "w") as f:
    json.dump(subclasses["organization"], f)

with open("text_analysis/wikidata_ids/occurrence_event.json", "w") as f:
    json.dump(subclasses["occurrence_event"], f)
