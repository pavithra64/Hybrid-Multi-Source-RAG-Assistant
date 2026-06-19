from ddgs import DDGS

def search_web(query):

    results = []

    try:

        with DDGS() as ddgs:

            for r in ddgs.text(
                query,
                max_results=5
            ):

                results.append({
                    "title": r["title"],
                    "body": r["body"]
                })

    except Exception as e:

        print(e)

    return results