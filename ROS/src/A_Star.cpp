#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Path.h>
#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <sstream>
#include <algorithm>
#include <memory>
#include <geometry_msgs/PoseStamped>

// using common std commands
using std::cout;
using std::endl;

struct Node {
    int x, y;
    double cost;
    double heuristic;
    Node* parent;

    Node(int x_, int y_, double cost_, double heuristic_, Node* parent_) :
            x(x_), y(y_), cost(cost_), heuristic(heuristic_), parent(parent_) {}

    double totalCost() const {
        return cost + heuristic;
    }
};

// structure to compare nodes 
struct CompareNodes {
    bool operator()(const Node* lhs, const Node* rhs) const {
        return lhs->totalCost() > rhs->totalCost();
    }
};

// Function to calculate Euclidean distance heuristic
double calculateHeuristic(int x1, int y1, int x2, int y2) {
    return std::sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

std::vector<Node*> astar(Node* start, Node* goal, const std::vector<std::vector<int>>& map, int width, int height) {
    std::priority_queue<Node*, std::vector<Node*>, CompareNodes> openSet;
    std::vector<std::vector<bool>> visited(height, std::vector<bool>(width, false));
    openSet.push(start);

    while (!openSet.empty()) {
        Node* current = openSet.top();
        openSet.pop();

        if (current->x == goal->x && current->y == goal->y) {
            std::vector<Node*> path;
            while (current != nullptr) {
                path.push_back(current);
                current = current->parent;
            }
            std::reverse(path.begin(), path.end());
            return path;
        }

        visited[current->y][current->x] = true;

        // Generate neighbor nodes
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                if (dx == 0 && dy == 0) continue; // Skip current node
                int nx = current->x + dx;
                int ny = current->y + dy;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height && map[ny][nx] == 0 && !visited[ny][nx]) {
                    double newCost = current->cost + 1; // Assuming uniform cost for all movements
                    Node* neighbor = new Node(nx, ny, newCost, calculateHeuristic(nx, ny, goal->x, goal->y), current);
                    openSet.push(neighbor);
                }
            }
        }
    }

    return {}; // No path found
}

void callBack(const nav_msgs::OccupancyGrid::ConstPtr& data) {
    int width = data->info.width;
    int height = data->info.height;
    double resolution = data->info.resolution;
    cout << "width: " << width << endl;
    cout << "height: " << height << endl;
    std::vector<std::vector<int>> map(height, std::vector<int>(width));

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int index = i * width + j;
            map[i][j] = data->data[index];
        }
    }

    Node* start = new Node(10, 10, 0, 0, nullptr); // Assuming start point coordinates
    Node* goal = new Node(50, 50, 0, 0, nullptr); // Assuming goal point coordinates

    std::vector<Node*> path = astar(start, goal, map, width, height);

    ros::NodeHandle nh;
    ros::Publisher path_pub = nh.advertise<nav_msgs::Path>("path", 1);
    nav_msgs::Path path_msg;
    path_msg.header.frame_id = "map";

    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        geometry_msgs::PoseStamped pose;
        pose.pose.position.x = (*it)->x * resolution;
        pose.pose.position.y = (*it)->y * resolution;
        pose.pose.orientation.w = 1.0;
        path_msg.poses.push_back(pose);
    }

    path_pub.publish(path_msg);

    // Clean up allocated memory
    for (Node* node : path) {
        delete node;
    }
}

int main(int argc, char ** argv) {
    // initialise ros
    ros::init(argc, argv, "AStar");
    ros::NodeHandle n;
    ros::Subscriber map = n.subscribe("map", 1, callBack); // subscribing to the map topic
    ros::spin();
    return 0;
}
