
import sys
import pathlib
import json
import numpy as np

sys.path.append('../../priv_traj_gen')
from my_utils import load
from grid import Grid


def run(path, save_path):
    save_path = pathlib.Path(save_path)
    # load
    with open(pathlib.Path(path).parent / "params.json", "r") as f:
        param = json.load(f)
    n_bins = param["n_bins"]
    lat_range = param["lat_range"]
    lon_range = param["lon_range"]
    distance_matrix = np.load(pathlib.Path(path).parent.parent.parent / f"distance_matrix_bin{n_bins}.npy")

    edges, edges_properties, adjs = make_edge_properties(lat_range, lon_range, n_bins, distance_matrix)
    # make edge_property file
    with open(save_path / "edge_property.txt", "w") as f:
        for i in range(1, len(edges_properties)+1):
            from_lat = edges_properties[i-1][3][0][0]
            from_lon = edges_properties[i-1][3][0][1]
            to_lat = edges_properties[i-1][3][1][0]
            to_lon = edges_properties[i-1][3][1][1]
            f.write(f'{i},{edges_properties[i-1][0]},{edges_properties[i-1][1]},{edges_properties[i-1][2]},LINESTRING"({from_lat} {from_lon},{to_lat} {to_lon})"\n')
    
    # make id_to_edge file
    id_to_edge = {}
    for i in range(1, len(edges_properties)+1):
        id_to_edge[i] = edges[i-1]
    with open(save_path / "id_to_edge.json", "w") as f:
        json.dump(id_to_edge, f)
    edge_to_id = {v:k for k,v in id_to_edge.items()}

    # make adjs file
    max_n_adjs = 4
    with open(save_path / "edge_adj.txt", "w") as f:
        for edge in edges:
            end_location = edge[-1]
            if len(edge) == 1:
                adj_edges = adjs[end_location]
            else:
                adj_edges = adjs[end_location]
                # remove the edge that reverse the direction
                # adj_edges = [adj_edge for adj_edge in adj_edges if adj_edge != (edge[1],edge[0])]
            
            adj_edge_ids = [edge_to_id[adj_edge] for adj_edge in adj_edges]
            # padding with -1
            adj_edge_ids.extend([-1]*(max_n_adjs-len(adj_edge_ids)))
            f.write(f',{",".join([str(adj_edge_id) for adj_edge_id in adj_edge_ids])}\n')

    # make trajectory file
    trajectories = load(path)
    trajs = convert(trajectories, edge_to_id, n_bins)
    with open(save_path / "trajs_demo.csv", "w") as f:
        for traj in trajs:
            f.write(" ".join([str(vocab) for vocab in traj] + [str(0)])+"\n")

    # make time file
    time_trajectories = load(pathlib.Path(path).parent / "training_data_time.csv")
    time_trajs = convert_time(time_trajectories)
    with open(save_path / "tstamps_demo.csv", "w") as f:
        for traj in time_trajs:
            f.write(" ".join([str(vocab) for vocab in traj])+"\n")

def convert(trajectories, edge_to_id, n_bins):
    
    new_trajectories = []
    # we first convert a trajectory to a list of edges
    for traj in trajectories:
        # edge_traj has a special vocab in the first place that represents the start place of the trajectory
        new_traj = [edge_to_id[(traj[0],)]]
        # compliment and convert
        edge_traj = convert_traj_to_edges(traj, n_bins)
        for edge in edge_traj:
            if edge_to_id[edge] != new_traj[-1]:
                new_traj.append(edge_to_id[edge])
        
        new_trajectories.append(new_traj)
    
    return new_trajectories

def convert_time(time_trajectories):
    new_trajectories = []
    for traj in time_trajectories:
        new_traj = [0]
        for i in range(len(traj)-1):
            new_traj.append(traj[i+1]-traj[i])
        new_trajectories.append(new_traj)
    return new_trajectories

def convert_traj_to_edges(traj, n_bins):
    new_traj = []
    for i in range(len(traj)-1):
        from_state = traj[i]
        to_state = traj[i+1]
        edge = (from_state,to_state)
        new_edges = compliment_edge(edge, n_bins)

        new_traj.extend(new_edges)
    return new_traj


def compliment_edge(edge, n_bins):
    '''
    In the case, the edge is not neighboring, we compliment the edge by adding the edges between the two states
    the route is the hamming way, from the direction of x-axis to the direction of y-axis
    '''

    from_state = edge[0]
    to_state = edge[1]
    from_x = from_state % (n_bins+2)
    from_y = from_state // (n_bins+2)
    to_x = to_state % (n_bins+2)
    to_y = to_state // (n_bins+2)
    x_sign = 1 if from_x < to_x else -1
    y_sign = 1 if from_y < to_y else -1

    edges = []
    for x in range(from_x, to_x, x_sign):
        edges.append((from_y*(n_bins+2)+x, from_y*(n_bins+2)+x+x_sign))
    
    for y in range(from_y, to_y, y_sign):
        edges.append((y*(n_bins+2)+to_x, (y+y_sign)*(n_bins+2)+to_x))

    return edges


def make_edge_properties(lat_range, lon_range, n_bins, distance_matrix):

    print("make grid of ", lat_range, lon_range, n_bins)
    ranges = Grid.make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, n_bins)
    grid = Grid(ranges)
    edges, adjs = make_edges(n_bins)
    aux_infos = [add_aux_info_to_edge(edge, distance_matrix, grid.state_to_center_latlon) for edge in edges]

    return edges, aux_infos, adjs

def make_edges(n_bins):
    # edges are made in the order of state
    edges = []
    adjs = {state:[] for state in range((n_bins+2)**2)}
    for y in range(n_bins+2):
        for x in range(n_bins+2):
            state = y * (n_bins+2) + x
            edges.append((state,))
            if x -1 >= 0:
                edge = (state,state-1)
                if edge not in edges:
                    edges.append(edge)
                adjs[state].append(edge)
            if y -1 >= 0:
                edge = (state,state-n_bins-2)
                if edge not in edges:
                    edges.append(edge)
                adjs[state].append(edge)
            if x <= n_bins:
                edge = (state,state+1)
                if edge not in edges:
                    edges.append(edge)
                adjs[state].append(edge)
            if y <= n_bins:
                edge = (state,state+n_bins+2)
                if edge not in edges:
                    edges.append(edge)
                adjs[state].append(edge)

    return edges, adjs

def add_aux_info_to_edge(edge, distance_matrix, state_to_latlon):
    # All WKTs have two length
    # 2 road types (start, move)
    # 3 types of headding, north, east south, we consider the headding of the start vocab as 0
    # length is the Euclidian distance of the two centers of the grids
    n_locations_in_x = int(np.sqrt(distance_matrix.shape[0]))
    if len(edge) == 1:
        from_latlon = state_to_latlon(edge[0])
        to_latlon = state_to_latlon(edge[0]) 
        heading = 0
        road_type = 0
        length = 0      
    elif len(edge) == 2:
        from_latlon = state_to_latlon(edge[0])
        to_latlon = state_to_latlon(edge[1])
        if edge[0] == edge[1] - 1:
            heading = 90
        elif edge[0] == edge[1] + 1:
            heading = 270
        elif edge[0] == edge[1] + n_locations_in_x:
            heading = 180
        elif edge[0] == edge[1] - n_locations_in_x:
            heading = 0
        else:
            print(edge)
            raise ValueError("edge must be neighboring")
        road_type = 1
        length = distance_matrix[edge[0]][edge[1]]
    else:
        raise ValueError("edge length must be 1 or 2")
    return length, road_type, heading, (from_latlon, to_latlon)

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])