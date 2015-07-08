from twitter import *

OAUTH_TOKEN     = '575700682-2QKg3hxxoPUjRMiW2tSMDLOGXKGSNAqVIvMMqGaP'
OAUTH_SECRET    = 'yC9OxcJq0GeIQoqft1kfSQDCBPrghiJhtKajDgFP4Y'
CONSUMER_KEY    = 'TmkRRiVZYaGPTVqKLvLhg'
CONSUMER_SECRET = 'LhKzNZK3sCM5y3aVMLONq3oKXTrfOB5cdDcTa2xzk0'

t = Api(consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET, access_token_key=OAUTH_TOKEN, access_token_secret=OAUTH_SECRET)

status = 'script works!'

t.PostUpdates(status)

statuses = t.GetUserTimeline('LukeMSchmidt')
print [s.text for s in statuses]



#MY_TWITTER_CREDS = os.path.expanduser('~/.my_app_credentials')
#if not os.path.exists(MY_TWITTER_CREDS):
#    oauth_dance("My App Name", CONSUMER_KEY, CONSUMER_SECRET,
#                MY_TWITTER_CREDS)

#oauth_token, oauth_secret = read_token_file(MY_TWITTER_CREDS)

#twitter = Twitter(auth=OAuth(
#    oauth_token, oauth_secret, CONSUMER_KEY, CONSUMER_SECRET))

## Now work with Twitter
#twitter.statuses.update('Hello, world!')