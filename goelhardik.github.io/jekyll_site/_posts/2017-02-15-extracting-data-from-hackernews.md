---
layout: post
comments: true
title: Extracting data from HackerNews using Firebase API in Python
message: ../images/ycombinator.png
---


<div class="message">
	This will be a small post where we will learn how to extract stories from HackerNews using the Firebase API in Python.
</div>

This is like a data collection step for a much bigger problem. You might want to do some form of analysis of the stories being posted on HackerNews or see which user has received the most upvotes in the past year. To do anything like that, we need the data first. Since I like Python, I tried to figure out how to do it and am sharing it here.

<h3>Install Firebase</h3>
To get started we will need to install the Firebase Python package. It is easy to do via `pip`. Just follow the steps [here](https://pypi.python.org/pypi/python-firebase/1.2).

<h3>Code</h3>
Once Firebase is installed, we can go ahead and write the code to do our work.

Do the imports:
{% highlight python %}
from firebase import firebase
import pickle
from requests.exceptions import HTTPError, ConnectionError
{% endhighlight %}

We will see below why we need to import the exceptions.

Now define our base HackerNews URL. This is the URL using which HackerNews provides its stories through Firebase.

{% highlight python %}
HN_BASE = "https://hacker-news.firebaseio.com"
{% endhighlight %}

Now we will write our scraper class. We will need to initialize a FirebaseApplication with the URL we defined above.

{% highlight python %}
class HN():
    def __init__(self):
        self.fb = firebase.FirebaseApplication(HN_BASE)

{% endhighlight %}

At this point we need to decide what range of data do we want to get from HackerNews. Let's assume I want the data from the year 2016 - i.e. from <i>1st Jan 2016 00:00:00 GMT to 1st Jan 2017 00:00:00 GMT</i>.
I'll go [here](http://www.epochconverter.com/) and convert these dates into UNIX time.
Once I have the times, I need to find out a starting story index.
The stories in HackerNews are numbered starting from 1 in increments of 1 for each story.
Every post, comment, etc. on HackerNews is a story.
I could start from 1 and filter out only the ones I want - but that will take eternity to reach 2016's stories. Instead, with some trial and error you could figure out a reasonable starting index by filling in the last field in <i>https://hacker-news.firebaseio.com/v0/item/[story-id]</i> and finding a story with time pretty close to our starting time.

Once we have done that, we can define some variables as below. The values are just for illustration and do not really mean anything.
{% highlight python %}
hn = HN()
startind = 11142100
start_time = 1451606400
end_time = 1483228800
f = 'stories_2016.pickle'
hn.fetch_stories(startind, start_time, end_time, f)
{% endhighlight %}

Now for the main functions that do the fetching work. The first one is `fetch_stories`, which given a start story index, start time and end time, will fetch all the stories that fall within the specified time period. It will also keep saving the results every 100 stories into the filename we provide in the last argument - a safety measure in case your script has to stop for some reason.
{% highlight python %}
    def fetch_stories(self, startind, start, end, f):
        stories = []
        # found via trial and error
        i = startind
        while (1):
            print("Getting story", i)
            story = self.get_item(i)
            if (story is None or 'time' not in story):
                i += 1
                continue
            if (story['time'] >= start and story['time'] <= end):
                stories.append(story)
            elif (story['time'] > end):
                break
            i += 1
            # save every 100 stories
            if (i % 100 == 0): 
                with open(f, 'wb') as af: 
                    pickle.dump(stories, af) 
                print("Last dumped story ", i-1, ' Total stories', len(stories))
        with open(f, 'wb') as af: 
            pickle.dump(stories, af) 
        return stories
{% endhighlight %}

The `fetch_stories` function calls a `get_item` function, which actually gets a particular story from Firebase.

{% highlight python %}
    def get_item(self, num=1):
        while True:
            try:
                item = self.fb.get('/v0/item', num)
                break
            except HTTPError:
                print("HTTPError! Retrying!")
            except ConnectionError:
                print("ConnectionError! Retrying!")
        return item
{% endhighlight %}

The exception handling in the above code is pretty useful. It happens a lot of time that while requesting a story an `HTTPError` or a `ConnectionError` occurs. If we don't handle the exception here, our script will die and we'll need to restart.
By handling the exceptions in a `while` loop, our script never really dies. It will just keep trying until it gets through to the story and then will move forward.
This handling turned out to be really useful in writing a robust data collector.

<br/>

That is the entire code and our data collector is now ready! It still takes a lot of time to gather data for just 1 year. But once we have it, we can have all sorts of fun with it!
