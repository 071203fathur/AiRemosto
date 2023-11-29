import requests
import random
from time import sleep
from datetime import datetime

def faker():
    now =  datetime.now().timestamp()
    values={
            "gender": {
                "created_at" : now,
                "pria": random.randrange(0, 50),
                "wanita" : random.randrange(0, 75)
            },
            "age" : {
                "created_at" : now,
                "anak" : random.randrange(0, 75),
                "remaja" : random.randrange(0, 75),
                "dewasa" : random.randrange(0, 75),
                "lansia" : random.randrange(0, 75)
            },
            "race" : {
                "created_at" : now,
                "negroid" : random.randrange(0, 75),
                "east_asian" : random.randrange(0, 75),
                "indian" : random.randrange(0, 75),
                "latin" : random.randrange(0, 75),
                "middle_eastern" : random.randrange(0, 75),
                "south_east_asian" : random.randrange(0, 75),
                "kauskasia" : random.randrange(0, 75),
            },
            "luggage" : {
                "created_at" : now,
                "manusia" : random.randrange(0, 75),
                "besar" : random.randrange(0, 75),
                "sedang" : random.randrange(0, 75),
                "kecil": random.randrange(0, 75),
            },
            "expression" : {
                "created_at" : now,
                "marah" : random.randrange(0, 75),
                "risih" : random.randrange(0, 75),
                "takut" : random.randrange(0, 75),
                "senyum" : random.randrange(0, 75),
                "netral" : random.randrange(0, 75),
                "sedih" : random.randrange(0, 75),
                "terkejut" : random.randrange(0, 75)
            }
        }
    print(values)
    now =  datetime.now().timestamp()
    t = requests.post("https://remosto-fe.vercel.app/api/camera", json=values)
    print(t.json())
    now2 =  datetime.now().timestamp()
    deltaTime = now2-now
    print("waktu:", deltaTime)

while True:
    faker()
    sleep(1)
