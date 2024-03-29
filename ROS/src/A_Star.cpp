#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <queue>
#include <vector>
#include <cmath>
#include <memory> // Include for smart pointers

struct Node {
    int x, y;
    double g_cost;
    double h_cost;
    double f_cost;
    std::shared_ptr<Node> parent; // Smart pointer

    Node(int x, int y) : x(x), y(y), g_cost(0), h_cost(0), f_cost(0), parent(nullptr) {}

    bool operator<(const Node& other) const {
        return f_cost > other.f_cost; // For priority queue
    }
};

class AStarPlanner {
public:
    AStarPlanner(ros::NodeHandle& nh) : nh_(nh), map_received_(false) {
        map_sub_ = nh_.subscribe("/map", 1, &AStarPlanner::mapCallback, this);
        path_pub_ = nh_.advertise<nav_msgs::Path>("/path", 1);
    }

    void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg) {
        map_ = *msg;
        map_received_ = true;
        ROS_INFO("Map received!");
        nav_msgs::Path path = planPath();
        path_pub_.publish(path);
    }

    bool isMapReceived() const {
        return map_received_;
    }


private:
    ros::NodeHandle nh_;
    ros::Subscriber map_sub_;
    ros::Publisher path_pub_;
    nav_msgs::OccupancyGrid map_;
    bool map_received_;

    double euclideanDistance(int x1, int y1, int x2, int y2) {
        return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
    }

    bool isValid(int x, int y) {
        return x >= 0 && x < map_.info.width && y >= 0 && y < map_.info.height &&
               map_.data[y * map_.info.width + x] == 0;
    }

    nav_msgs::Path planPath() {
    nav_msgs::Path path;
    path.header = map_.header;

    if (!isMapReceived()) {
        ROS_WARN("Map has not been received yet!");
        return path;
    }

    int map_width = map_.info.width;
    int map_height = map_.info.height;

    Node start(0, 0);
    Node goal(20,10);

    ROS_INFO("Start node: (%d, %d)", start.x, start.y);
    ROS_INFO("Goal node: (%d, %d)", goal.x, goal.y);

    std::priority_queue<Node> open_set;
    std::vector<std::vector<bool>> closed_set(map_width, std::vector<bool>(map_height, false));

    open_set.push(start);

    while (!open_set.empty()) {
        Node current = open_set.top();
        open_set.pop();

        ROS_INFO("Exploring node: (%d, %d)", current.x, current.y);

        if (current.x == goal.x && current.y == goal.y) {
            ROS_INFO("Path found!");
            reconstructPath(current, path);
            return path;
        }

        closed_set[current.x][current.y] = true;

        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                if (dx == 0 && dy == 0)
                    continue;

                int new_x = current.x + dx;
                int new_y = current.y + dy;

                if (!isValid(new_x, new_y) || closed_set[new_x][new_y])
                    continue;

                double tentative_g_cost = current.g_cost + euclideanDistance(current.x, current.y, new_x, new_y);
                Node neighbor(new_x, new_y);
                neighbor.g_cost = tentative_g_cost;
                neighbor.h_cost = euclideanDistance(new_x, new_y, goal.x, goal.y);
                neighbor.f_cost = neighbor.g_cost + neighbor.h_cost;
                neighbor.parent = std::make_shared<Node>(current); // Use smart pointer

                ROS_INFO("Adding neighbor: (%d, %d) to open set with f_cost: %f", neighbor.x, neighbor.y, neighbor.f_cost);

                open_set.push(neighbor);
            }
        }
    }

    ROS_WARN("Failed to find path!");
    return path;
}

    void reconstructPath(const Node& goal_node, nav_msgs::Path& path) {
        Node current = goal_node;
        while (current.parent != nullptr) {
            geometry_msgs::PoseStamped pose;
            pose.pose.position.x = current.x * map_.info.resolution + map_.info.origin.position.x;
            pose.pose.position.y = current.y * map_.info.resolution + map_.info.origin.position.y;
            path.poses.insert(path.poses.begin(), pose);
            current = *(current.parent); // Dereference smart pointer
        }
        geometry_msgs::PoseStamped start_pose;
        start_pose.pose.position.x = current.x * map_.info.resolution + map_.info.origin.position.x;
        start_pose.pose.position.y = current.y * map_.info.resolution + map_.info.origin.position.y;
        path.poses.insert(path.poses.begin(), start_pose);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "path_planner_node");
    ros::NodeHandle nh;

    AStarPlanner planner(nh);

    ros::spin();

    return 0;
}
