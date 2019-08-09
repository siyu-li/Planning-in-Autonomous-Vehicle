//
// Created by Siyu Li on 2019-08-07.
//

#include <uWS/uWS.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "helpers.h"
#include "json.hpp"
#include "spline.h"
#include <time.h>
#include <algorithm>
#include <float.h>

// for convenience
using namespace std;
using nlohmann::json;
using std::string;
using std::vector;

#define ACTION_NUM 5
#define STATE_NUM 72
#define SAFETY_DIS 30
#define EPSILON 0.3
#define GAMMA 0.95
#define LR 0.1
#define SPEED_LIMIT 45
#define BEST_SPEED 40
#define MIN_SPEED 10
#define COLLISION_DIS 30


vector<vector<double>> Q(STATE_NUM, vector<double>(ACTION_NUM, 0));
//vector<vector<double>>Q{
//    {62.4096,49.3378, 0, 62.6437, 53.9113},
//    {0, 27.3142, 0, 26.5026, 0},
//    {0, 0, 0, 0, 0},
//    {49.1609, 25.342, 0, 36.8408, 24.2823},
//    {0, 21.4178, 0, 20.0726, 0},
//    {0, 0, 0, 0, 0},
//    {-16.3365, 22.7006, 0, 17.2182, -165.136},
//    {0, -227.811, 0, 11.1046, 0},
//    {0,0,0,0,0},
//    {4.9,0,0,0,0},
//    {0,0,0,0,0},
//    {0,0,0,0,0},
//    {19.2198, 0,0,0,0},
//    {0, 3.8, 0,0,0},
//    {0,0,0,0,0},
//    {0,0,0,0,0},
//    {0,0,0,0,0},
//    {0,0,0,0,0},
//    {0,0,0,0,0},
//    {0,0,0,0,0},
//    {0,0,0,0,0},
//    {0,0,0,0,0},
//    {0,0,0,0,0},
//    {7.61366e-311, 0,0,0,0},
//    {60.1391, 25.0835, 44.6279, 37.4168, 43.7603},
//    {0, 21.4626, 32.6877, 16.9756, 0},
//    {0,0,0,0,0},
//    {19.3082, 9.94323, 12.9673, 0, 0},
//    {0, 3.95031, 7.1872, 0, 0},
//    {0,0,0,0,0},
//    {-253.229, -210.564, -90.7626, -237.929, -228.52},
//    {0, -169.514, -187.722, -53.0566, 0},
//    {0,0,0,0,0},
//    {-6.97428, 10.5809, 17.0198, 19.5238, -26.4843},
//    {0, 16.1668, 8.9299, 15.3622, 0},
//    {0,0,0,0,0},
//    {57.7342, 48.0829, 55.9672, 55.4735, 52.738},
//    {0, 29.9366, 19.0126, 22.033, 0},
//    {0,0,0,0,0},
//    {55.6089, 33.7267, 43.1115, 22.311, 23.7991},
//    {0, 28.2706, 19.0899, 5.7533, 0},
//    {0,0,0,0,0},
//    {0,0,0,0,0},
//    {0,0,0,0,0},
//    {0,0,0,0,0},
//    {9.37249, 2.813, 0, 0, 2.04229},
//    {0, 4.12132, 0, 0, 0},
//    {0,0,0,0,0},
//    {63.3848, 55.4097, 56.0504, 0, 55.8415},
//    {0, 38.0685, 27.7487, 0, 0},
//    {0,0,0,0,0},
//    {0,0,0,0,0},
//    {0,0,0,0,0},
//    {0,0,0,0,0},
//    {27.9677, 0, 16.7262, 0, 24.2175},
//    {0, -20.2703, -2.74334, 0, 0},
//    {0,0,0,0,0},
//    {0,0,0,0,0},
//    {0,0,0,0,0},
//    {0,0,0,0,0},
//    {64.9685, 31.2071, 46.5888, 0, 28.8542},
//    {0, -35.2408, -2.62547, 0, 0},
//    {0,0,0,0,0},
//    {0,0,0,0,0},
//    {0,0,0,0,0},
//    {0,0,0,0,0},
//    {-340.897, -176.566, -301.494, 0, -337.411},
//    {0, -126.681, -295.819, 0, 0},
//    {0,0,0,0,0},
//    {0,0,0,0,0},
//    {0, -29.6507, 0, 0, 0}
//};


//get current state
vector<int> get_current_state(int lane, vector<vector<double>>& sensor_fusion, double car_speed, double car_s){
    vector<int> current_state(5,0);
    current_state[0] = lane;
    for(int i=0; i<sensor_fusion.size(); i++){
        float d = sensor_fusion[i][6];
        double check_car_s = sensor_fusion[i][5];
        //car is in my lane
        if(d<8 && d>4){
            //car is not safe
            if(check_car_s - car_s > 0 && check_car_s - car_s < SAFETY_DIS) current_state[2] = 1;
        }else if(d<4 && d>0){
            //car is in the left lane
            if(check_car_s - car_s > 0 && check_car_s - car_s < SAFETY_DIS) current_state[1] = 1;
        }else if(d<12 && d>8){
            //car is in the right lane
            if(check_car_s - car_s > 0 && check_car_s - car_s < SAFETY_DIS) current_state[3] = 1;
        }
    }
    if(car_speed >= SPEED_LIMIT) current_state[4] = 2;
    else if(car_speed <= MIN_SPEED) current_state[4] = 0;
    else current_state[4] = 1;
    return current_state;
}

//get possible action
vector<int> get_possible_action(vector<int>& current_state){
    vector<int> possible_action;
    int lane = current_state[0];
    int velo = current_state[4];
    //lane
    if(lane == 0 && current_state[2] == 0) {
        possible_action.push_back(3);
    }
    if(lane == 1){
        if(current_state[3] == 0) possible_action.push_back(3);
        if(current_state[1] == 0) possible_action.push_back(2);
    }
    if(lane == 2 && current_state[2] == 0)
        possible_action.push_back(2);
    //velo
    if(velo == 0) possible_action.push_back(0);
    else if(velo == 2) possible_action.push_back(1);
    else{
        possible_action.push_back(0);
        possible_action.push_back(1);
        possible_action.push_back(4);
    }
    return possible_action;
}

int get_state_index(vector<int>& current_state){
    int state_index = current_state[0]*24 + current_state[1]*12 + current_state[2]*6 + current_state[3]*3 + current_state[4];
    return state_index;
}

//get the action of max q
int get_max_q_action(int epsilon_greedy, vector<int> possible_action, int state_index){
    int size = possible_action.size();
    srand(time(0));
    double ran_possible = rand()%100/(double)101;
    if(epsilon_greedy && ran_possible < EPSILON){
        return possible_action[(rand() % size)];
    }else{
        vector<double> Q_state = Q[state_index];
        int max_index = -1;
        double max_value = -DBL_MAX;
        for(int i=0; i< Q_state.size(); i++){
            auto it = find(possible_action.begin(), possible_action.end(), i);
            if((it != possible_action.end()) && Q_state[i] > max_value){
                max_index = i;
                max_value = Q_state[i];
            }
        }
        return max_index;
    }
}

//double get_reward1(int last_action, vector<vector<double >>& sensor_fusion, double car_s, int lane, double car_speed, int old_lane) {
//    double r = 1;
//    //speed REWARD
//    if(car_speed>SPEED_LIMIT) r -= (SPEED_LIMIT - car_speed)/6;
//    if(last_action == 0 && car_speed < BEST_SPEED) r += car_speed/6;
//    if(car_speed<MIN_SPEED) r -= 0.5;
//    //decrease reward when not safety
//    for (int i = 0; i < sensor_fusion.size(); i++) {
//        float d = sensor_fusion[i][6];
//        double check_car_s = sensor_fusion[i][5];
//        if (d < (lane * 4 + 4) && d > (lane * 4)) {
//            //car is not safe
//            if (abs(check_car_s - car_s) < COLLISION_DIS) {
//                r -= 10;
//                vector<vector<double>> zeros(STATE_NUM, vector<double>(ACTION_NUM, 0));
//                Q = zeros;
//
//            }
//            return r;
//        }
//    }
//
//}

//double get_reward2(int last_action, vector<vector<double >>& sensor_fusion, double car_s, int lane, double car_speed, int old_lane) {
//    double r = 1;
//    //speed REWARD
//    if(car_speed>SPEED_LIMIT) r -= 1;
//    else if(car_speed<MIN_SPEED) r -= 1;
//    else r += car_speed/SPEED_LIMIT;
//    //OBSTACLE
//    //decrease reward when not safety
//
//    bool flag= true;
//    for (int i = 0; i < sensor_fusion.size(); i++) {
//        float d = sensor_fusion[i][6];
//        double check_car_s = sensor_fusion[i][5];
//        if (d < (lane * 4 + 4) && d > (lane * 4)) {
//            //car is not safe
//            if (abs(check_car_s - car_s) < COLLISION_DIS) {
//                flag= false;
//                break;
//            }
//        }
//    }
//    if(flag) r+=1;
//    else r-=1;
//    return r;
//
//}

double get_reward(vector<int> current_state, double car_speed) {
    double r = 1;
    //speed REWARD
    if(current_state[4] == 0) r -= 1;
    else if(current_state[4] == 2) r -= 1;
    else r += car_speed/SPEED_LIMIT;;

    //OBSTACLE
    //decrease reward when not safety
    if(current_state[current_state[0]+1] == 1) r-=2;
    else r+=2;
    return r;


}

//update Q table
void update_Q_table(double r, int state_index, int action, int new_state_index, vector<int> possible_action){
    // greedy method to get action index of max_q
    int q_newstate_max_index = get_max_q_action(0, possible_action, new_state_index);
    double q_newstate_max = Q[new_state_index][q_newstate_max_index];
    Q[state_index][action] = Q[state_index][action] + LR*(r + GAMMA * q_newstate_max - Q[state_index][action]);
    cout << "update state_index "<<state_index <<" action "<< action<<" with "<<Q[state_index][action] <<endl;

}



int main() {
    uWS::Hub h;

    // Load up map values for waypoint's x,y,s and d normalized normal vectors
    vector<double> map_waypoints_x;
    vector<double> map_waypoints_y;
    vector<double> map_waypoints_s;
    vector<double> map_waypoints_dx;
    vector<double> map_waypoints_dy;

    // Waypoint map to read from
    string map_file_ = "../data/highway_map.csv";
    // The max s value before wrapping around the track back to 0
    double max_s = 6945.554;

    std::ifstream in_map_(map_file_.c_str(), std::ifstream::in);

    string line;
    while (getline(in_map_, line)) {
        std::istringstream iss(line);
        double x;
        double y;
        float s;
        float d_x;
        float d_y;
        iss >> x;
        iss >> y;
        iss >> s;
        iss >> d_x;
        iss >> d_y;
        map_waypoints_x.push_back(x);
        map_waypoints_y.push_back(y);
        map_waypoints_s.push_back(s);
        map_waypoints_dx.push_back(d_x);
        map_waypoints_dy.push_back(d_y);
    }
    //start in lane 1
    int lane = 1;
    //have a reference velocity to target
    double ref_val = 5;
    //initialize
    vector<int> past_state = {lane, 0, 0, 0, 1};
    int past_state_index = get_state_index(past_state);
    int past_action = 0;

    h.onMessage([&past_state_index, &past_state, &past_action, &ref_val,&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,
                        &map_waypoints_dx,&map_waypoints_dy,&lane]
                        (uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                         uWS::OpCode opCode) {
        // "42" at the start of the message means there's a websocket message event.
        // The 4 signifies a websocket message
        // The 2 signifies a websocket event
        if (length && length > 2 && data[0] == '4' && data[1] == '2') {

            auto s = hasData(data);

            if (s != "") {
                auto j = json::parse(s);

                string event = j[0].get<string>();

                if (event == "telemetry") {
                    // j[1] is the data JSON object

                    // Main car's localization Data
                    double car_x = j[1]["x"];
                    double car_y = j[1]["y"];
                    double car_s = j[1]["s"];
                    double car_d = j[1]["d"];
                    double car_yaw = j[1]["yaw"];
                    double car_speed = j[1]["speed"];

                    // Previous path data given to the Planner
                    auto previous_path_x = j[1]["previous_path_x"];
                    auto previous_path_y = j[1]["previous_path_y"];
                    // Previous path's end s and d values
                    double end_path_s = j[1]["end_path_s"];
                    double end_path_d = j[1]["end_path_d"];

                    // Sensor Fusion Data, a list of all other cars on the same side
                    //   of the road.
                    vector<vector<double>> sensor_fusion = j[1]["sensor_fusion"];

                    //previous size
                    int prev_size = previous_path_x.size();

                    //car in current lane
                    int current_lane = car_d/4;
                    int old_lane = past_state[0];
                    //get current state
                    vector<int> current_state = get_current_state(current_lane, sensor_fusion, car_speed, car_s);
                    //get current state index
                    int current_state_index = get_state_index(current_state);
                    //get possible action
                    vector<int> possible_action = get_possible_action(current_state);
                    //using eplison greedy to get action
                    //int current_action = get_max_q_action(1, possible_action, current_state_index);
                    int current_action = get_max_q_action(0, possible_action, current_state_index);
                    //get reward
                    //double r = get_reward(past_action, sensor_fusion, car_s, current_lane, car_speed, old_lane, current_state);
                    double r = get_reward(current_state, car_speed);
                    //update
                    update_Q_table(r, past_state_index, past_action, current_state_index, possible_action);

//          cout << "possible_action is ";
//          for(int i=0; i<possible_action.size(); i++){
//              cout << possible_action[i] << " ";
//          }
//          cout << endl;


                    //here comes sensor fusion
                    if(prev_size > 0) {
                        car_s = end_path_s;
                    }

                    // bool too_close = false;
                    // //find ref_v to use
                    // for(int i=0; i<sensor_fusion.size(); i++){
                    //   //car is in my lane
                    //   float d = sensor_fusion[i][6];
                    //   if(d<(lane*4 +4) && d>(lane*4)){
                    //     double vx = sensor_fusion[i][3];
                    //     double vy = sensor_fusion[i][4];
                    //     double check_speed = sqrt(vx*vx + vy*vy);
                    //     double check_car_s = sensor_fusion[i][5];
                    //     check_car_s += ((double)prev_size*0.02*check_speed);
                    //     //take action is car is close
                    //     if((check_car_s>car_s)&&((check_car_s - car_s) < 30)){
                    //       too_close = true;
                    //       if(lane > 0)
                    //         lane = 0;
                    //     }
                    //   }
                    // }

                    // if(too_close){
                    //   ref_val -= .224;
                    // }else if(ref_val < 49.5){
                    //   ref_val += .224;
                    // }
                    //here ends sensor fusion

                    //control input

                    switch(current_action){
                        case 0:
                            ref_val += 0.5;
                            break;
                        case 1:
                            ref_val -= 0.5;
                            break;
                        case 2:
                            lane = current_lane - 1;
                            break;
                        case 3:
                            lane = current_lane + 1;
                            break;
                        case 4:
                            break;
                        default:
                            break;
                    }
                    if(ref_val < 0) ref_val = 0;
                    //debug
                    cout << "Q table is " << endl;
                    for(int i=0; i<Q.size();i++){
                        cout << i << " ";
                        for(int j=0; j<Q[0].size();j++){
                            cout<<Q[i][j]<< " ";
                        }
                        cout<<endl;
                    }
//          cout << "current_action is "<< current_action << endl;
//          cout << "reference velo is "<< ref_val << endl;
//          cout << "current lane is " << current_lane << endl;
//          cout << "lane is " << lane << endl;
                    //update state and action for next iteration
                    past_state = current_state;
                    past_state_index = current_state_index;
                    past_action = current_action;

                    //create a list of widely spaced (x,y) points, evenly spaces at 30m
                    //later we will interpolate theses waypoints with spline
                    vector<double> ptsx;
                    vector<double> ptsy;

                    //reference x, y, yaw states
                    double ref_x = car_x;
                    double ref_y = car_y;
                    double ref_yaw = deg2rad(car_yaw);

                    //if previous size is almost empty, use the car as starting reference
                    if(prev_size < 2){
                        //use two point that make the path tangent to the car
                        double prev_car_x = car_x - cos(car_yaw);
                        double prev_car_y = car_y - sin(car_yaw);

                        ptsx.push_back(prev_car_x);
                        ptsx.push_back(car_x);

                        ptsy.push_back(prev_car_y);
                        ptsy.push_back(car_y);
                    }
                        //using previous path's and pint as starting reference
                    else
                    {
                        //redefine reference state as previous path and points
                        ref_x = previous_path_x[prev_size-1];
                        ref_y = previous_path_y[prev_size-1];

                        double ref_x_prev = previous_path_x[prev_size-2];
                        double ref_y_prev = previous_path_y[prev_size-2];
                        ref_yaw = atan2(ref_y-ref_y_prev, ref_x-ref_x_prev);

                        //use two points that make the path tangent to he previous path's end point
                        ptsx.push_back(ref_x_prev);
                        ptsx.push_back(ref_x);

                        ptsy.push_back(ref_y_prev);
                        ptsy.push_back(ref_y);

                    }

                    //In frenet add evenly 30m spaced points ahead of the starting reference
                    vector<double> nextwp0 = getXY(car_s+30, 2+4*lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);
                    vector<double> nextwp1 = getXY(car_s+60, 2+4*lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);
                    vector<double> nextwp2 = getXY(car_s+90, 2+4*lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);

                    ptsx.push_back(nextwp0[0]);
                    ptsx.push_back(nextwp1[0]);
                    ptsx.push_back(nextwp2[0]);

                    ptsy.push_back(nextwp0[1]);
                    ptsy.push_back(nextwp1[1]);
                    ptsy.push_back(nextwp2[1]);

                    //shift to car reference
                    for(int i=0; i < ptsx.size(); i++){
                        double shift_x = ptsx[i] - ref_x;
                        double shift_y = ptsy[i] - ref_y;
                        //shift car reference angle to 0 degree
                        ptsx[i] = cos(0-ref_yaw)*shift_x - sin(0-ref_yaw)*shift_y;
                        ptsy[i] = sin(0-ref_yaw)*shift_x + cos(0-ref_yaw)*shift_y;

                    }

                    //define splnie
                    tk::spline s;
                    //set(x, y) to pline
                    s.set_points(ptsx, ptsy);


                    //define actual (x, y) we will use for planner
                    vector<double> next_x_vals;
                    vector<double> next_y_vals;

                    //start with all of the previous path points from the last time
                    for(int i=0; i<previous_path_x.size(); i++){
                        next_x_vals.push_back(previous_path_x[i]);
                        next_y_vals.push_back(previous_path_y[i]);

                    }

                    //calculate how to break up spline points so that we travel at our desired reference velocity
                    double target_x = 30.0;
                    double target_y = s(target_x);
                    double target_dist = sqrt(target_x*target_x + target_y*target_y);

                    double x_add_on = 0;

                    //fill out the rest of path planning
                    for(int i=1; i <= 50-previous_path_x.size(); i++){
                        double N = target_dist/((.02*ref_val)/2.24);
                        double x_point = x_add_on + target_x/N;
                        double y_point = s(x_point);

                        x_add_on = x_point;

                        double x_ref = x_point;
                        double y_ref = y_point;

                        //rotate back to normal after rotate it
                        x_point = cos(ref_yaw)*x_ref - sin(ref_yaw)*y_ref;
                        y_point = sin(ref_yaw)*x_ref + cos(ref_yaw)*y_ref;

                        x_point +=ref_x;
                        y_point +=ref_y;

                        next_x_vals.push_back(x_point);
                        next_y_vals.push_back(y_point);


                    }

                    /**
                     * TODO: define a path made up of (x,y) points that the car will visit
                     *   sequentially every .02 seconds
                     */

                    // double dist_inc = 0.3;
                    // for(int i = 0; i < 50; i++)
                    // {
                    //   double next_s = car_s + (i+1)*dist_inc;
                    //   double next_d = 6;
                    //   vector<double> xy = getXY(next_s, next_d, map_waypoints_s, map_waypoints_x, map_waypoints_y);
                    //   next_x_vals.push_back(xy[0]);
                    //   next_y_vals.push_back(xy[1]);
                    // }

                    //define Json message
                    json msgJson;
                    msgJson["next_x"] = next_x_vals;
                    msgJson["next_y"] = next_y_vals;

                    auto msg = "42[\"control\","+ msgJson.dump()+"]";

                    ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
                }  // end "telemetry" if
            } else {
                // Manual driving
                std::string msg = "42[\"manual\",{}]";
                ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
            }
        }  // end websocket if
    }); // end h.onMessage

    h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
        std::cout << "Connected!!!" << std::endl;
    });

    h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                           char *message, size_t length) {
        ws.close();
        std::cout << "Disconnected" << std::endl;
    });

    int port = 4567;
    if (h.listen(port)) {
        std::cout << "Listening to port " << port << std::endl;
    } else {
        std::cerr << "Failed to listen to port" << std::endl;
        return -1;
    }

    h.run();
}