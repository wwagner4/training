import training.simdb as db


def main():
    client = db.create_client()
    all = db.find_all(client)
    for d in all[-1:]:
        print(d.keys())
        print(d["started_at"])
        for s in d["states"][0:50]:
            print(s["robot1"]["xpos"])
