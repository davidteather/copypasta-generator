from generate_for_reddit import *
import praw
import random
import time

starting_phrase = ['Somebody once told me', 'My parents disowned me, because', 
                    'I asked someone out and they responded with', "You're welcome", 'I spent years discovering this', 
                    'Some guy on discord just sent this to me', 'My friend thinks this is good copypasta', 
                    'edgy teenagers be like']

while True:
    try:
        post_pasta(random.choice(starting_phrase))
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
            time.sleep(int(time_delay)*multiplier)
        except Exception as e:
            print(e)
            print("No specific time catch")
            time.sleep(10*60)
    except KeyboardInterrupt:
        break

    except OSError:
        print("???")
        continue