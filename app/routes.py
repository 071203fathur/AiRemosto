from app import app, db
from flask import request, jsonify
from app.models import Expression, Race, Age, Gender, Luggage
from datetime import datetime

# emotion_dict = ["marah", "risih", "takut", "senyum", "netral", "sedih", "terkejut"]
# classes = ['negroid','east_asian','indian','latin','middle_eastern','south_east_asian','kaukasia']
@app.route('/', methods=['GET', 'POST'])
def index():
    return "Welcome to REMOSTO-API"

@app.route('/expression', methods=['POST'])
def expression():
   data = Expression(
       marah=request.json["marah"],
       risih=request.json["risih"],
       takut=request.json["takut"],
       senyum=request.json["senyum"],
       netral=request.json["netral"],
       sedih=request.json["sedih"],
       terkejut=request.json["terkejut"]
       )
   db.session.add(data)
   db.session.commit()
   return jsonify(request.json)

@app.route('/race', methods=['POST'])
def race():
   data = Race(
       negroid=request.json["negroid"],
       east_asian=request.json["east_asian"],
       indian=request.json["indian"],
       latin=request.json["latin"],
       middle_eastern=request.json["middle_eastern"],
       south_east_asian=request.json["south_east_asian"],
       kaukasia=request.json["kaukasia"]
       )
   db.session.add(data)
   db.session.commit()
   return jsonify(request.json)

@app.route('/age', methods=['POST'])
def age():
   data = Age(
      created_at=int(datetime.utcnow().timestamp())+25200,
       anak=int(request.json["anak"]),
       remaja=int(request.json["remaja"]),
       dewasa=int(request.json["dewasa"]),
       lansia=int(request.json["lansia"]),
       )
   db.session.add(data)
   db.session.commit()
   return jsonify(request.json)

@app.route('/gender', methods=['POST'])
def gender():
   data = Gender(
      created_at=int(datetime.utcnow().timestamp())+25200,
       pria=int(request.json["pria"]),
       wanita=int(request.json["wanita"]),
       )
   db.session.add(data)
   db.session.commit()
   return jsonify(request.json)

@app.route('/luggage', methods=['POST'])
def luggage():
   data = Luggage(
      created_at=int(datetime.utcnow().timestamp())+25200,
       manusia=int(request.json["manusia"]),
       besar=int(request.json["besar"]),
       sedang=int(request.json["sedang"]),
       kecil=int(request.json["kecil"]),
       )
   db.session.add(data)
   db.session.commit()
   return jsonify(request.json)

@app.route('/get/<int:number>/', methods=['GET','POST'])
def getdata(number):
    data_expression = [i.response() for i in Expression.query.all()[:number]]
    data_race = [i.response() for i in Race.query.all()[:number]]
    response = {"status": 200, "time_response": int(datetime.utcnow().timestamp()+25200), "expression": data_expression, "race": data_race}

    return response