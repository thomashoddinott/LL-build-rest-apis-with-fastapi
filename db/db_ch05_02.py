from datetime import datetime, timedelta
from time import sleep


def query_posts(login, since: datetime):
    """Query posts by login since <since> (default to 7 days ago)"""
    now = datetime.now()

    sleep(0.123)  # Simulate a slow database query
    posts = []
    n = 0
    while since < now:
        n += 1
        posts.append(
            {
                "author": login,
                "time": since,
                "content": f"post #{n}",
            }
        )
        since += timedelta(hours=7.2)

    return posts
