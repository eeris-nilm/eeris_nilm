import dill
import pymongo
from eeris_nilm.algorithms import livehart

mclient = pymongo.MongoClient("mongodb://localhost:27017")
mdb = mclient['eeris']
inst_doc = mdb.models.find_one({"meterId": "5e05d5c83e442d4f78db036f"})
model = dill.loads(inst_doc['modelHart'])
