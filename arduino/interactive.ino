#include <ros.h>
#include <softgrasp_ros/SetGraspState.h>
#include <softgrasp_ros/SetGraspParams.h>

ros::NodeHandle nh;
using softgrasp_ros::SetGraspState;
using softgrasp_ros::SetGraspParams;

// status indicator
const int LED = 13;

// output
const int TRIGGER= 9;
const int MODE_SEL = 5;
const int OPENING = 10;
const int GRIP = 3;

// input
const int PRESSURE_IN = A1;


void grasp(bool state) {
  digitalWrite(TRIGGER, state ? HIGH : LOW);
}

void state_cb(const SetGraspState::Request &req, SetGraspState::Response &res) {
  digitalWrite(LED, req.state ? HIGH : LOW);
  grasp(req.state);
  res.success = true;
  res.message = "done";
}

void setparams(float opening, float grip) {
  // o = 3.1374*opening + 1.5687
  // g = 4.1832*grip    + 1.0458
  // pin = int(round(o*255/5)) = int(o*51 + 0.5)
  opening = 160.0074*opening + 80.5037;
  grip    = 213.3432*grip    + 53.8358;
  analogWrite(OPENING, constrain(int(opening), 0, 255));
  analogWrite(GRIP,    constrain(int(grip), 0, 255));
}

void params_cb(const SetGraspParams::Request &req, SetGraspParams::Response &res) {
  setparams(req.opening_amount, req.grip_strength);
  res.success = true;
  res.message = "done";
}

// service servers
ros::ServiceServer<SetGraspState::Request, SetGraspState::Response> state_server("grasp_state", &state_cb);
ros::ServiceServer<SetGraspParams::Request, SetGraspParams::Response> params_server("grasp_params", &params_cb);

void setup()
{
  pinMode(LED, OUTPUT);
  pinMode(TRIGGER, OUTPUT);
  pinMode(MODE_SEL, OUTPUT);
  pinMode(OPENING, OUTPUT);
  pinMode(GRIP, OUTPUT);
  pinMode(PRESSURE_IN, INPUT);
  

  digitalWrite(MODE_SEL, HIGH);
  grasp(false);
  setparams(0.5, 0.5);
  
  nh.initNode();
  nh.advertiseService(state_server);
  nh.advertiseService(params_server);
}

void loop()
{
  nh.spinOnce();
  delay(33);
}
