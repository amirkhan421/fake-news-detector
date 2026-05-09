import requests

API_KEY = "aef4ed6037fc4d6a96f0e7c6a566310b"

def verify_news_online(news_text):

    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={news_text}&"
        f"language=en&"
        f"sortBy=relevancy&"
        f"apiKey={API_KEY}"
    )

    response = requests.get(url)

    data = response.json()

    if data["status"] == "ok":
        total_results = data["totalResults"]

        if total_results > 0:
            return True, total_results
        else:
            return False, 0

    return False, 0