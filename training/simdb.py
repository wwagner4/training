import datetime as dt
from abc import ABC
from dataclasses import dataclass

import pymongo
from bson import ObjectId
from dataclasses_json import dataclass_json

SIM_STATUS_RUNNING = "running"
SIM_STATUS_FINISHED = "finished"
SIM_STATUS_ERROR = "error"


@dataclass_json
@dataclass
class Simulation:
    port: int
    started_at: dt.datetime = dt.datetime.now()
    status: str = SIM_STATUS_RUNNING
    message: str = ""


def create_client() -> ABC:
    # noinspection PyTypeChecker
    return pymongo.MongoClient("mongodb://localhost:27017/")


def _sim_collection(client: pymongo.MongoClient):
    db = client["sumosim"]
    return db["simulations"]


def insert(client: pymongo.MongoClient, sim_dict: dict) -> str:
    sims = _sim_collection(client)
    result = sims.insert_one(sim_dict)
    str_id = str(result.inserted_id)
    return str_id


def find(client: pymongo.MongoClient, id: str) -> dict:
    raise NotImplementedError()


def find_all(client: pymongo.MongoClient) -> dict:
    sims = _sim_collection(client)
    return list(sims.find())


def update_status_error(client: pymongo.MongoClient, doc_id: str, message: str):
    sims = _sim_collection(client)
    update_document = {"$set": {"status": SIM_STATUS_ERROR, "message": message}}
    obj_id = ObjectId(doc_id)
    sims.update_one({"_id": obj_id}, update_document)


def update_status_finished(client: pymongo.MongoClient, doc_id: str, events: dict):
    sims = _sim_collection(client)
    update_document = {"$set": {"status": SIM_STATUS_FINISHED, "events": events}}
    obj_id = ObjectId(doc_id)
    sims.update_one({"_id": obj_id}, update_document)


def find_running(
    client: pymongo.MongoClient, status: str, base_port: int
) -> list[dict]:
    sims = _sim_collection(client)
    query = {
        "status": status,
        "base_port": base_port,
    }
    return list(sims.find(query))


################## query #####################


def count_running():
    with create_client() as client:
        sims = _sim_collection(client)
        query = {
            "status": SIM_STATUS_RUNNING,
        }
        cnt = sims.count_documents(query)
        print(f"{cnt} simulations are currently running")


def list_running():
    with create_client() as client:
        sims = _sim_collection(client)
        query = {
            "status": SIM_STATUS_RUNNING,
        }
        for i, r in enumerate(sims.find(query)):
            print(f"{i} {r}")


def list_latest():
    with create_client() as client:
        sims = _sim_collection(client)
        for i, r in enumerate(sims.find().sort({"started_at": -1}).limit(10)):
            print(f"{i} {r}")


def delete_old_running():
    with create_client() as client:
        sims = _sim_collection(client)
        now = dt.datetime.now()
        diff = dt.timedelta(days=1)
        minus = now - diff

        query = {"status": SIM_STATUS_RUNNING, "started_at": {"$lt": minus}}
        answer = sims.delete_many(query)
        if not answer.acknowledged:
            raise RuntimeError(f"Could not delete from mongo db {query}")
        print(f"deleted {answer.deleted_count} using {query}")
