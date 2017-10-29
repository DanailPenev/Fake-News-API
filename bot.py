import botometer, keys
mashape_key = keys.get_key()
twitter_app_auth = keys.get_twitter_auth()

bom = botometer.Botometer(wait_on_ratelimit=True,
                          mashape_key=mashape_key,
                          **twitter_app_auth)

def test_user(user):
	return bom.check_account(user)