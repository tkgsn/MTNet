import json
import pathlib
import sys
# generated data format:
# 3677,3681,3837,3681,3838,3844,4001,4005,0,0,0,0,0,0,0,0,0,0,0,0,0

def convert_to_original_format(training_data_dir, path):
    # load id_to_edge
    print(training_data_dir / "id_to_edge.json")
    with open(training_data_dir / "id_to_edge.json", "r") as f:
        id_to_edge = json.load(f)
    id_to_edge = {int(k):v for k,v in id_to_edge.items()}

    # load data
    trajs = []
    with open(pathlib.Path(path), "r") as f:
        for line in f:
            traj = [int(vocab) for vocab in line.split(",")]
            # traj = [int(vocab) for vocab in line.strip().split(" ")]
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
    training_data_dir = pathlib.Path(sys.argv[1])
    save_dir = pathlib.Path(sys.argv[2])
    save_dir.mkdir(parents=True, exist_ok=True)
    files = [file for file in save_dir.iterdir() if file.name.startswith("samples_")]
    # sort
    files.sort(key=lambda x: int(x.name.split("_")[1].split(".")[0]))
    for file in files:
        # get the id
        id = file.name.split("_")[1].split(".")[0]
        print("convert to original format: ", file)
        trajs = convert_to_original_format(training_data_dir, file)
        print("save to", save_dir / f"generated_{id}.csv")
        with open(save_dir / f"generated_{id}.csv", "w") as f:
            for traj in trajs:
                f.write(",".join([str(vocab) for vocab in traj]) + "\n")