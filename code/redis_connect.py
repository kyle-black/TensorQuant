import redis
#import prediction_oopEUR

# Replace these values with your actual Redis server credentials
host = 'localhost'
port = 6379
db = 0
decode_responses = True

# Connect to the Redis server
redis_connection = redis.Redis(host=host, port=port, db=db)




'''
res = r.set("bike:1", "Process 134")
print(res)
# >>> True

res = r.get("bike:1")
print(res)
print(prediction_oopEUR.output)
'''


