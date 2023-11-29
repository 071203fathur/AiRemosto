from app import db
from sqlalchemy import func
from datetime import datetime
# emotion_dict = ["marah", "risih", "takut", "senyum", "netral", "sedih", "terkejut"]
# classes = ['negroid','east_asian','indian','latin','middle_eastern','south_east_asian','kaukasia']

class Expression(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.Integer, nullable=False, default=int(datetime.utcnow().timestamp() + 25200))
    marah = db.Column(db.Integer, index=True)
    risih = db.Column(db.Integer, index=True)
    takut = db.Column(db.Integer, index=True)
    senyum = db.Column(db.Integer, index=True)
    netral = db.Column(db.Integer, index=True)
    sedih = db.Column(db.Integer, index=True)
    terkejut = db.Column(db.Integer, index=True)

    def __repr__(self):
        return f'<time {self.created_at}>, <Id {self.id}>, {self.marah}, {self.risih}, {self.takut}, {self.senyum}, {self.netral}, {self.sedih}, {self.terkejut}' 
    def response(self):
        return {"time":self.created_at,"marah": self.marah, "risih": self.risih, "takut": self.takut, "senyum": self.senyum, "netral":self.netral, "sedih": self.sedih, "terkejut": self.terkejut} 

class Race(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.Integer, nullable=False, default=int(datetime.utcnow().timestamp() + 25200))
    negroid = db.Column(db.Integer, index=True)
    east_asian = db.Column(db.Integer, index=True)
    indian = db.Column(db.Integer, index=True)
    latin = db.Column(db.Integer, index=True)
    middle_eastern = db.Column(db.Integer, index=True)
    south_east_asian = db.Column(db.Integer, index=True)
    kaukasia = db.Column(db.Integer, index=True)

    def __repr__(self):
        return f'<time {self.created_at}>, <Id {self.id}>, {self.negroid}, {self.east_asian}, {self.indian}, {self.latin}, {self.middle_eastern}, {self.south_east_asian}, {self.kaukasia}' 
    def response(self):
        return {"time":self.created_at,"negroid": self.negroid, "east_asian": self.east_asian, "indian": self.indian, "latin": self.latin, "middle_eastern":self.middle_eastern, "south_east_asian": self.south_east_asian, "kaukasia": self.kaukasia} 

class Age(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.Integer, index=True)
    anak = db.Column(db.Integer, index=True)
    remaja = db.Column(db.Integer, index=True)
    dewasa = db.Column(db.Integer, index=True)
    lansia = db.Column(db.Integer, index=True)

    def __repr__(self):
        return f'<time {self.created_at}>, <Id {self.id}>, {self.anak}, {self.remaja}, {self.dewasa}, {self.lansia}' 
    def response(self):
        return {"time":self.created_at,"anak": self.anak, "remaja": self.remaja, "dewasa": self.dewasa, "lansia": self.lansia} 

class Gender(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.Integer, index=True)
    pria = db.Column(db.Integer, index=True)
    wanita = db.Column(db.Integer, index=True)

    def __repr__(self):
        return f'<time {self.created_at}>, <Id {self.id}>, {self.pria}, {self.wanita}' 
    def response(self):
        return {"time":self.created_at,"pria": self.pria, "wanita": self.wanita} 
    
class Luggage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.Integer, index=True)
    manusia = db.Column(db.Integer, index=True)
    besar = db.Column(db.Integer, index=True)
    sedang = db.Column(db.Integer, index=True)
    kecil = db.Column(db.Integer, index=True)

    def __repr__(self):
        return f'<time {self.created_at}>, <Id {self.id}>, {self.manusia}, {self.besar}, {self.sedang}, {self.kecil}' 
    def response(self):
        return {"time":self.created_at,"manusia": self.manusia, "besar": self.besar, "sedang": self.sedang, "kecil": self.kecil} 