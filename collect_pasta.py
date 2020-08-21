import praw
import json

# Loads Settings
with open("reddit_details.json", 'r') as i:
    settings = json.loads(i.read())

# Creates Reddit object to read submissions
reddit = praw.Reddit(client_id=settings['client_id'],
                     client_secret=settings['client_secret'],
                     user_agent=settings['user_agent'],
                     username=settings['username'],
                     password=settings['password'])

sorting_type = "top"

if sorting_type == "top":
    pastas = reddit.subreddit("copypasta").top(limit=10000)
elif sorting_type == "new":
    pastas = reddit.subreddit("copypasta").new(limit=10000)

text = ""
for pasta in pastas:
    if pasta.is_self:
        text += pasta.selftext

with open("data.txt", 'w+', encoding='utf-8') as output:
    output.write(text)