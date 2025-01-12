from pathlib import Path
import yaml
import dominate
import dominate.tags as dt
import dominate.util as du
import shutil
import markdown as md
from dataclasses import dataclass
import training.helper as hlp
import re

from dominate.dom_tag import dom_tag


@dataclass(frozen=True)
class Resources:
    videos: list[Path]
    q_values_heat: list[Path]
    boxplots: list[Path]


@dataclass
class ResourceLinks:
    text: str
    boxplots: list[str]
    q_values: list[str]
    videos: list[str]


def extract_combis(lead_resources: list[Path], prefix: str) -> list[str]:
    keys_set = set()
    for resource in lead_resources:
        if resource.name.startswith(prefix):
            name_rest = resource.stem[len(prefix) + 1 :]
            split_name = name_rest.split("-")
            keys_set.add(split_name[0])
    return sorted(list(keys_set))


def sumosim_report_path(dir: str, must_exists: bool = True) -> Path:
    p = Path(dir)
    if not p.is_absolute():
        p = Path.home() / p
    if must_exists and not p.exists():
        raise ValueError(f"path {dir} does not exit")
    return p


def collect_resources(results_dirs: list[Path], result_name: str) -> list[Path]:
    out = []
    for result_path in results_dirs:
        out += [
            r.absolute() for r in result_path.iterdir() if r.stem.endswith(result_name)
        ]
    return sorted(out)


def create_report(reports_data: dict, result_dir_paths: list[Path], out_dir: str):
    videos = collect_resources(result_dir_paths, "sumosim-video")
    q_values = collect_resources(result_dir_paths, "q-values-heat")
    boxplots = collect_resources(result_dir_paths, "boxplot")
    resources = Resources(videos=videos, q_values_heat=q_values, boxplots=boxplots)

    out_path = sumosim_report_path(out_dir, must_exists=False)
    out_path.mkdir(parents=True, exist_ok=True)

    style_path = Path(__file__).parent.parent.parent.parent / "resources" / "styles.css"
    style_target = out_path / "styles.css"
    shutil.copy(style_path, style_target)
    # print(f"Created style {style_target}")

    create_report_index(reports_data, out_path, resources)

    print(f"Created reports in {out_path.absolute()}")


# noinspection DuplicatedCode
def create_report_index(report_dict: dict, out_path: Path, resources: Resources):
    out_file = out_path / "index.html"
    method_tuples = [
        create_report_method(method_dict, i, out_path, resources)
        for i, method_dict in enumerate(report_dict["methods"])
    ]
    doc = dominate.document(title=report_dict["title"])
    with doc.head:
        dt.meta(name="viewport", content="width=device-width")
        dt.link(rel="stylesheet", href="styles.css")
    with doc:
        dt.h1().add(report_dict["title"])
        dt.p().add(du.raw(md.markdown(report_dict["description"])))
        for text, link in method_tuples:
            dt.a(text, href=link)
            dt.br()
    with out_file.open("w") as f:
        f.write(str(doc))


# noinspection DuplicatedCode
def create_report_method(
    method_dict: dict, index: int, out_path: Path, resources: Resources
) -> tuple[str, str]:
    out_file_name = f"method-{index:02d}.html"
    out_file = out_path / out_file_name
    training_tuples = [
        create_report_training(method_dict, i, out_path, resources)
        for i, method_dict in enumerate(method_dict["trainings"])
    ]

    doc = dominate.document(title=method_dict["title"])

    with doc.head:
        dt.meta(name="viewport", content="width=device-width")
        dt.link(rel="stylesheet", href="styles.css")

    with doc:
        dt.h1().add(method_dict["title"])
        dt.p().add(du.raw(md.markdown(method_dict["description"])))
        for training_tuple in training_tuples:
            if training_tuple is not None:
                text, link = training_tuple
                dt.a(text, href=link)
                dt.br()

    with out_file.open("w") as f:
        f.write(str(doc))
    return method_dict["abstract"], out_file_name


def create_report_training(
    training_dict: dict, index: int, out_path: Path, resources: Resources
) -> tuple[str, str] | None:
    prefix = training_dict["prefix"]

    def tags_for_combis(
        resources: Resources, combis: list[str], out_path: Path
    ) -> dom_tag:
        def filter_copy_resource(
            resources: list[Path], prefix: str, out_path: Path
        ) -> list[str]:
            res_path = out_path / prefix
            res_path.mkdir(parents=True, exist_ok=True)

            filtered_res = [res for res in resources if res.stem.startswith(prefix)]
            links = []
            for res in filtered_res:
                target_path = res_path / res.name
                if not target_path.exists():
                    shutil.copy(res, target_path)
                    print(f"### Copied {res.name} to {res_path}")
                links.append(f"{prefix}/{res.name}")
            return links

        def match_resource_name(name: str, combi: str) -> bool:
            name_rest = name[len(prefix) + 1 :]
            split_name = name_rest.split("-")
            return name.startswith(prefix) and split_name[0] == combi

        def filter_and_sort_videos(combi: str) -> list[Path]:
            _videos = [
                _res
                for _res in resources.videos
                if match_resource_name(_res.name, combi)
            ]
            return sorted(_videos)

        def tags_for_video_links(index: int, link: str) -> dom_tag:
            if index > 0 and index % 4 == 0:
                return  (
                    dt.br(),
                    dt.a(f"video {index}", href=link)
                )
            return (
                dt.a(f"video {index}", href=link)
            )



        def tags_for_resource(resourceLinks: ResourceLinks) -> dom_tag:
            return dt.div(
                resourceLinks.text,
                dt.br(),
                [
                    dt.a(dt.img(src=link, width=400), href=link)
                    for link in resourceLinks.boxplots
                ],
                dt.br(),
                [dt.a("q-values", href=link) for link in resourceLinks.q_values],
                dt.br(),
                [
                    tags_for_video_links(i, link)
                    for i, link in enumerate(resourceLinks.videos)
                ],
                _class="box",
            )

        def create_resource_links(combi: str) -> ResourceLinks:
            text = f"Results for: {prefix} {combi}"
            boxplots = [
                res
                for res in resources.boxplots
                if match_resource_name(res.name, combi)
            ]
            q_values = [
                res
                for res in resources.q_values_heat
                if match_resource_name(res.name, combi)
            ]
            _videos = filter_and_sort_videos(combi)
            return ResourceLinks(
                text=text,
                boxplots=filter_copy_resource(boxplots, prefix, out_path),
                q_values=filter_copy_resource(q_values, prefix, out_path),
                videos=filter_copy_resource(_videos, prefix, out_path),
            )

        links = [create_resource_links(combi) for combi in combis]
        return dt.div([tags_for_resource(cv) for cv in links], _class="container")

    out_file_name = f"training-{index:02d}.html"
    out_file = out_path / out_file_name
    _combis = extract_combis(resources.boxplots, prefix)

    if not _combis:
        return None

    doc = dominate.document(title=training_dict["title"])

    with doc.head:
        dt.meta(name="viewport", content="width=device-width")
        dt.link(rel="stylesheet", href="styles.css")

    with doc.body:
        dt.h1().add(training_dict["title"])
        dt.p().add(du.raw(md.markdown(training_dict["description"].strip())))
        tags_for_combis(resources, _combis, out_path)

    with out_file.open("w") as f:
        f.write(str(doc))
    return training_dict["abstract"], out_file_name


def create_final_ressources(reports_data: dict, result_dir_paths: list[Path]):
    def create(result_dir_path: Path, prefix: str):
        all_resources = list(
            [r for r in result_dir_path.iterdir() if r.stem.startswith(prefix)]
        )
        for r in all_resources:
            if r.name.startswith(prefix) and r.name.endswith("mp4"):
                pass
                # print("### ", r.name)
        lead_ressources = [r for r in all_resources if r.stem.endswith("boxplot")]
        if lead_ressources:
            combis = extract_combis(lead_ressources, prefix)
            print(
                f"-- create final ressources {len(all_resources)} {prefix} in {result_dir_path}"
            )
            for combi in combis:
                create_heat_video(all_resources, combi, prefix, result_dir_path)
                create_simulation_video(all_resources, combi, prefix, result_dir_path)

    def create_heat_video(all_resources, combi, prefix, result_dir_path):
        name = f"{prefix}-{combi}-q-values-heat.mp4"
        created_heat = [r for r in all_resources if r.name == name]
        if created_heat:
            pass
            # print(f"EXISTS {name} -> Nothing to do")
        else:
            video_path = result_dir_path / name
            print(f"CREATE {name}")
            images_dir = result_dir_path / "q-value-heat"
            image_prefix = f"{prefix}-{combi}"
            image_count = len(
                list(
                    [i for i in images_dir.iterdir() if i.name.startswith(image_prefix)]
                )
            )
            image_pattern = f"{image_prefix}*.png"
            frame_rate = image_count // 5
            cmd = [
                "ffmpeg",
                "-framerate",
                str(frame_rate),
                "-pattern_type",
                "glob",
                "-i",
                f"{image_pattern}",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(video_path.absolute()),
            ]
            print(f"calling '{' '.join(cmd)}'\ncwd: {images_dir}")
            ok, stdout = hlp.call1(cmd, work_path=images_dir)
            if not ok:
                print(f"ERROR calling {cmd}\n{stdout}")

    def create_simulation_video(all_resources, combi, prefix, result_dir_path):
        name_regex = f"{prefix}-{combi}-.*sumosim-video.mp4"
        is_created = len(
            list([r for r in all_resources if re.match(name_regex, r.name)])
        )
        if is_created:
            pass
            # print(f"EXISTS {name_regex} -> Nothing to do")
        else:
            simulation_prefix = f"{prefix}-{combi}"
            cmd = [
                "sumo",
                "video",
                "-p",
                simulation_prefix,
                "-o",
                str(result_dir_path.absolute()),
            ]
            print(f"CREATE simulation video '{' '.join(cmd)}'")
            ok, stdout = hlp.call1(cmd)
            if not ok:
                print(f"ERROR calling {cmd}\n{stdout}")

    for m in reports_data["methods"]:
        for t in m["trainings"]:
            prefix = t["prefix"]
            for p in result_dir_paths:
                create(p, prefix)


def report(results_dir_list: list[str], out_dir: str):
    report_path = (
        Path(__file__).parent.parent.parent.parent / "resources" / "report.yml"
    )
    with report_path.open() as f:
        reports_data = yaml.safe_load(f)
    # pprint(reports_data)
    result_dir_paths = [sumosim_report_path(dir) for dir in results_dir_list]
    create_final_ressources(reports_data, result_dir_paths)
    create_report(reports_data, result_dir_paths, out_dir)
