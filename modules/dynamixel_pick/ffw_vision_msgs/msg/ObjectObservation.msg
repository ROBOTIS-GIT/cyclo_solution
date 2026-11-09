# One object found in a source image, as an oriented box in source pixels.

string target_label  # e.g. dynamixel
string pose_label  # task-defined, e.g. lying, standing, unknown

uint8 STATUS_UNSET=0
uint8 STATUS_VALID=1
uint8 STATUS_AMBIGUOUS=3  # e.g. touching objects not separated
uint8 status

float64 u  # box center column, integer = pixel center
float64 v  # box center row
float64 length  # long side in pixels
float64 width  # short side in pixels
float64 angle  # long side direction from +u toward +v, rad, [-pi/2, pi/2)
float64 angle_period  # pi: rectangle, pi/2: near square, 0: no orientation

bool has_score
float32 score  # detection confidence, not position accuracy

float64 top_height  # top face above the array's support_plane, m; NaN if not measured
