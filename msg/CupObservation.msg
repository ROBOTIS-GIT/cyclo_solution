# BODY=1: head-camera visible surface. observation_type=2 on /perception/wrist_cup:
# measured upright RIM center from wrist_left; body_position stores the rim center.
# Invalid messages carry diagnostic codes; never use their position as a target.
uint8 BODY=1
uint8 observation_type
std_msgs/Header header
string source
string cup_id
geometry_msgs/Point body_position
float64 inner_radius
float64 position_error
float64 confidence
bool valid
string detail
