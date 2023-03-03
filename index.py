import praw
import faiss
import numpy
from sentence_transformers import SentenceTransformer
from praw.models import Comment, MoreComments
import json

def processComment(stack, comment, dates, ids):
    if isinstance(comment, Comment):
        text = stack + " | " + comment.body
        dates.append(int(comment.created_utc))
        ids.append(comment.id)
        yield text
        for reply in comment.replies:
            yield from processComment(text, reply, dates, ids)

def processSubmission(submission, dates, ids):
    for comment in submission.comments:
        yield from processComment("", comment, dates, ids)

def processQuery(query, dates, ids):
    for submission in query:
        yield from processSubmission(submission, dates, ids)

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
reddit = praw.Reddit(
    client_id="",
    client_secret="",
    password="",
    user_agent="",
    username="",
)
reddit.read_only = True
index = faiss.IndexFlatL2(384)
query = reddit.subreddit("realdaytrading").search("title:'Live Day Trading' author:AutoModerator", sort='new', limit=1000)
dates = []
ids = []
comments = list(processQuery(query, dates, ids))
index.add(model.encode(comments))
faiss.write_index(index, "submissions.index")
f = open("dates.json", "w")
f.write(json.dumps(dates))
f.close()
f = open("ids.json", "w")
f.write(json.dumps(ids))
f.close()
f = open("comments.json", "w")
f.write(json.dumps(comments))
f.close()
