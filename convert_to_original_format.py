import json
import pathlib
import sys
# generated data format:
# 3677,3681,3837,3681,3838,3844,4001,4005,0,0,0,0,0,0,0,0,0,0,0,0,0

def convert_to_original_format(path):
    # load id_to_edge
    with open(pathlib.Path(path).parent / "id_to_edge.json", "r") as f:
        id_to_edge = json.load(f)
    id_to_edge = {int(k):v for k,v in id_to_edge.items()}

    # load data
    trajs = []
    with open(pathlib.Path(path), "r") as f:
        for line in f:
            traj = [int(vocab) for vocab in line.split(",")]
            # remove 0s at the end
            traj = [vocab for vocab in traj if vocab != 0]
            trajs.append(traj)

    new_trajs = []
    for traj in trajs:
        new_traj = []
        for i in range(len(traj)):
            if i == 0:
                new_traj.append(id_to_edge[traj[i]][0])
            else:
                new_traj.append(id_to_edge[traj[i]][1])
        new_trajs.append(new_traj)

    return new_trajs


if __name__ == "__main__":
    # find generated data named samples_*.txt from directory given by the argument
    path = pathlib.Path(sys.argv[1])
    save_dir = pathlib.Path(sys.argv[2])
    save_dir.mkdir(parents=True, exist_ok=True)
    files = [file for file in path.iterdir() if file.name.startswith("samples_")]
    for i, file in enumerate(files):
        trajs = convert_to_original_format(file)
        with open(save_dir / f"generated_{i+1}.csv", "w") as f:
            for traj in trajs:
                f.write(",".join([str(vocab) for vocab in traj]) + "\n")