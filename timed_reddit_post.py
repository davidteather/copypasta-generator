from generate_for_reddit import *
import praw
import time

while True:
    try:
        post_pasta("My parents left me, because ")
    except praw.exceptions.RedditAPIException as e:
        try:
            if "minutes" in e.items[0].message.split("try again in ")[1]:
                multiplier = 60
                phrase = "minutes"
            else:
                multiplier = 1
                phrase = "seconds"
            
            time_delay = e.items[0].message.split("try again in ")[1].split(" minutes.")[0]
            print("sleeping {} {}".format(time_delay, phrase))
            time.sleep(int(time_delay)*60)
        except Exception as e:
            print(e)
            print("No specific time catch")
            time.sleep(10*60)
    except KeyboardInterrupt:
        break

    except OSError:
        continue