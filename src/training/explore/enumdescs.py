from pathlib import Path
import yaml


def extract_enumdescs(enumkeys: list[str]) -> list[str]:
    path = Path(__file__).parent.parent.parent.parent / "resources" / "enumdescs.yml"
    # print(f"### path {path} {path.exists()}")
    with path.open() as f:
        data = yaml.safe_load(f)
    # pprint(data)

    def find_enumdesc(class_name: str, value: str) -> str:
        found = [
            entry
            for entry in data
            if entry["class_name"] == class_name and entry["value"] == value
        ]
        if not found:
            raise ValueError(
                f"found no description for enum desc '{class_name}' '{value}'"
            )
        return found[0]

    def extract_enumdesc(enumkey: str) -> str:
        split_key = enumkey.split(".")
        class_name = ".".join(split_key[0:-1])
        value = split_key[-1]
        # print(f"## split key {class_name} {value}")
        desc_obj = find_enumdesc(class_name, value)
        return desc_obj["desc"]

    return [extract_enumdesc(enumkey) for enumkey in enumkeys]
