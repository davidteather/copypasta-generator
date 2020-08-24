# copypasta-generator
 Code that creates copypasta

# Demo

[![](thumbnail.png?raw=true)](https://youtu.be/rvufIaEFKoQ)

## Installing 

Run the following command (tested on python 3.7.3)
```
pip install -r requirements.txt
```

## Setup

You need to register a Reddit app, which you can do [here](https://www.reddit.com/prefs/apps/). Then fill out reddit_details.json with the following
```
{
    "client_id": "",
    "client_secret": "",
    "user_agent": "",
    "username": "",
    "password": ""
}
```

## Collecting Data 

```
python collect_pasta.py
```

## Training Model
```
python train.py
```

## Evaluating Model

Get direct output
```
python generate.py
```

Post to r/copypasta on a timed basis
```
python timed_reddit_post.py
```
