from pathlib import Path
import training.simdb as db
import json
from datetime import datetime, date


def adapt_for_serialize(obj: any) -> any:
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return obj


def create_entity_data(
    database_data: dict, name_part: str, description: str, nr: int
) -> dict:
    keys = [y for y in database_data.keys() if y != "_id"]
    data_dict = dict([(key, adapt_for_serialize(database_data[key])) for key in keys])
    data_dict["nr"] = nr
    return data_dict


def export_simulations(name_part: str, description: str, db_host: str, db_port: int):
    out_dir = Path.home("/tmp")
    print(f"Exporting simulations containing '{name_part}' to {out_dir}")
    with db.create_client(db_host, db_port) as client:
        all_database_data = db.list_match_name_part(client, name_part)
        if len(all_database_data) < 1:
            print(f"ERROR: Found no simulations matching '{name_part}'")
        entity_data = [
            create_entity_data(x, name_part, description, i)
            for i, x in enumerate(all_database_data)
        ]
        export_data = {
            "name_part": name_part,
            "description": description,
            "entities": entity_data,
        }
        out_dir.mkdir(exist_ok=True, parents=True)
        out_file = out_dir / f"{name_part}.json"
        with out_file.open("w") as file:
            json.dump(export_data, file, indent=2)
        print(f"SUCCESS: Exported data to {out_file}")
