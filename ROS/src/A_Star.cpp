#include <iostream>
#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <vector>
#include <cmath>
#include <queue>
#include <limits>
#include <unordered_set>
#include <memory>

// Define your Node structure for A* algorithm
struct Node {
    int i, j;
    float f, g, h;
    Node* parent;

    Node(int _i, int _j) : i(_i), j(_j), f(0), g(std::numeric_limits<float>::infinity()), h(0), parent(nullptr) {}

    float dist(const Node& other) const {
        return std::sqrt(std::pow(i - other.i, 2) + std::pow(j - other.j, 2));
    }

    bool operator==(const Node& other) const {
        return i == other.i && j == other.j;
    }
};

// Hash function for Node to use in unordered_set
namespace std {
template<>
struct hash<Node> {
    size_t operator()(const Node& node) const {
        return std::hash<int>()(node.i) ^ std::hash<int>()(node.j);
    }
};
}

// Structure to compare nodes for priority queue
struct CompareNodes {
    bool operator()(const Node& a, const Node& b) const {
        return a.f > b.f;
    }
};

// A* algorithm implementation
class AStarPathFinder {
public:
    AStarPathFinder(const std::vector<std::vector<int>>& _map, const Node& _start, const Node& _end, bool _allowDiagonals)
        : map(_map), start(_start), end(_end), allowDiagonals(_allowDiagonals) {}

    std::vector<Node> findPath() {
        std::vector<Node> openSet;
        std::unordered_set<Node> closedSet;

        openSet.push_back(start);

        int maxIterations = 1000; // Example maximum number of iterations
        int iterations = 0;

        while (!openSet.empty() && iterations < maxIterations) {
            auto current = openSet.back();
            openSet.pop_back();

            if (current == end) {
                auto path = reconstructPath(&current);
                return path;
            }

            closedSet.insert(current);

            for (auto& neighbor : getNeighbors(current)) {
                if (closedSet.find(neighbor) != closedSet.end()) continue;

                float tentativeG = current.g + current.dist(neighbor);

                auto it = std::find(openSet.begin(), openSet.end(), neighbor);
                if (it == openSet.end() || tentativeG < it->g) {
                    neighbor.g = tentativeG;
                    neighbor.h = neighbor.dist(end);
                    neighbor.f = neighbor.g + neighbor.h;
                    neighbor.parent = &current;
                    openSet.push_back(neighbor);
                }
            }

            iterations++;
        }

        return {}; // No path found within the maximum iterations
    }


private:
    const std::vector<std::vector<int>>& map;
    Node start;
    Node end;
    bool allowDiagonals;

    std::vector<Node> reconstructPath(Node* current) {
        std::vector<Node> path;
        while (current != nullptr) {
            path.emplace_back(*current);
            current = current->parent;
        }
        std::reverse(path.begin(), path.end());
        return path;
    }

    std::vector<Node> getNeighbors(const Node& node) const {
        std::vector<Node> neighbors;
        int di[8] = {1, 1, 0, -1, -1, -1, 0, 1};
        int dj[8] = {0, 1, 1, 1, 0, -1, -1, -1};
        int numDirs = allowDiagonals ? 8 : 4;

        for (int d = 0; d < numDirs; ++d) {
            int ni = node.i + di[d];
            int nj = node.j + dj[d];
            if (ni >= 0 && ni < map.size() && nj >= 0 && nj < map[0].size() && map[ni][nj] == 0) {
                neighbors.push_back(Node(ni, nj));
            }
        }

        return neighbors;
    }
};

// Global variables for storing map data
std::vector<std::vector<int>> mapData;
int mapWidth, mapHeight;

// Function to publish the path as a nav_msgs::Path message
void publishPath(const std::vector<Node>& path) {
    ros::NodeHandle nh;
    ros::Publisher path_pub = nh.advertise<nav_msgs::Path>("path", 1);

    nav_msgs::Path path_msg;
    path_msg.header.frame_id = "map";

    for (const auto& node : path) {
        geometry_msgs::PoseStamped pose;
        pose.pose.position.x = node.i;
        pose.pose.position.y = node.j;
        pose.pose.orientation.w = 1.0;
        path_msg.poses.push_back(pose);
    }

    path_pub.publish(path_msg);
}

// Function to check if a position is valid
bool isValidPosition(const Node& position) {
    return position.i >= 0 && position.i < mapHeight && position.j >= 0 && position.j < mapWidth;
}

// Callback function for map data
void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr& data) {
    mapWidth = data->info.width;
    mapHeight = data->info.height;
    mapData.clear();
    mapData.resize(mapHeight, std::vector<int>(mapWidth));

    // Copy map data to vector
    for (int i = 0; i < mapHeight; ++i) {
        for (int j = 0; j < mapWidth; ++j) {
            mapData[i][j] = data->data[i * mapWidth + j];
        }
    }

    // Example start and end nodes
    Node start(10,10);
    Node end(40,40);

    // Check if the start and end points are within the map boundaries
    if (!isValidPosition(start) || !isValidPosition(end)) {
        ROS_WARN("Invalid start or end point position");
        return;
    }

    // Print the coordinates of the start and end points
    ROS_INFO("Start point: (%d, %d)", start.i, start.j);
    ROS_INFO("End point: (%d, %d)", end.i, end.j);

    // Create AStarPathFinder object and find path
    AStarPathFinder pathFinder(mapData, start, end, true);
    std::vector<Node> path = pathFinder.findPath();

    // Check if path is found
    if (path.empty()) {
        ROS_WARN("No path found");
    } else {
        // Publish the path
        publishPath(path);
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "path_planner");
    ros::NodeHandle nh;
    ros::Subscriber map_sub = nh.subscribe("map", 1, mapCallback);
    ros::spin();
    return 0;
}
