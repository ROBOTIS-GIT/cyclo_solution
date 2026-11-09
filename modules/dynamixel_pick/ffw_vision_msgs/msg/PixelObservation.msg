# Pixel observation from one source image.
# Read status first; u and v are NaN unless STATUS_VALID.

std_msgs/Header header  # source image header (capture stamp, optical frame_id)
string camera_id  # head, left_wrist, right_wrist
string image_topic  # stream the coordinates refer to
uint32 image_width
uint32 image_height

string target_label  # may be empty for manual selection
string POINT_TYPE_SELECTED_POINT="selected_point"
string point_type

uint8 SOURCE_UNSET=0
uint8 SOURCE_MANUAL=1
uint8 SOURCE_AUTO=2
uint8 source_mode

uint8 STATUS_UNSET=0
uint8 STATUS_VALID=1
uint8 STATUS_NOT_DETECTED=2
uint8 STATUS_AMBIGUOUS=3
uint8 status

float64 u  # column, + right, integer = pixel center
float64 v  # row, + down
