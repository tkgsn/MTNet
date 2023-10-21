# update
The purpose of the update is comparison experiments on Geolife dataset, which includes moving ways other than road networks (e.g., train, subway)
We implemented make_training_data.py, which converts our training data to that following the format of MTNet.
This is realized by virtually interpreting that we have road networks that connect every two adjacent grids.
For example, if we have 16 grids that split a map, we have 24 edges.

'''
format of the edge_property.txt from PIDX = {'id': 0, 'len': 1, 'road_type': 2, 'heading': 3, 'WKT': 4}
WKT: stands for "Well-Known Text". It is a text markup language used for representing vector geometry objects.
heading: 0 -> north, 90 -> east, 180 -> south, 270 -> west
'''
All WKTs have two length
1 road type (virtually)
2 types of headding, north, east (south, west are identical to north, east respectively)
length is the Euclidian distance of the two centers of the grids

This includes virtual edges that represent the start location
That is, the number of virutal edgese is identical to the number of lcoations
Note that the start vocab only appears in the first location

For the Chengdu dataset, we use the provided configure data