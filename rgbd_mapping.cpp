#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/range_image/range_image.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/projection_matrix.h>
#include <pcl/search/kdtree.h>
#include <string>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/conversions.h>
#include <opencv/cv.h>
#include <image_geometry/pinhole_camera_model.h>
#include <boost/foreach.hpp>
#include <message_filters/subscriber.h>
#include <math.h>
#include <pcl/surface/marching_cubes_rbf.h>
#include <pcl/surface/marching_cubes_hoppe.h>
#include <pcl/surface/impl/marching_cubes_hoppe.hpp>
#include <pcl/PolygonMesh.h>
#include <pcl/surface/impl/marching_cubes.hpp>
#include <pcl/surface/marching_cubes.h>
#include <pcl/io/pcd_io.h>
#include <cmath> 
#include <pcl/io/obj_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/io/impl/vtk_lib_io.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <visualization_msgs/Marker.h>
#include <pcl/surface/boost.h>
#include <pcl/common/common.h>
#include <pcl/common/vector_average.h>
#include <pcl/Vertices.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Dense>


using namespace pcl;
using namespace std;
using namespace message_filters;
namespace enc = sensor_msgs::image_encodings;


sensor_msgs::PointCloud _voxel_mid_points;
sensor_msgs::PointCloud _v_mid_points_tf;
image_geometry::PinholeCameraModel cam_model_;
ros::Time acquisition_time;
tf::TransformListener* _listener;
PolygonMesh::Ptr triangles(new PolygonMesh);
MarchingCubesHoppe<PointNormal> mc;
ros::Subscriber* imageInfo;
ros::Publisher* pub;
float voxel_length = 0.02;
int grid_size = 256;
int pic = 0;
float delta = 0.7;
float epsilon = 0.025;
std::vector<float> _grid(grid_size*grid_size*grid_size, 0.5f);
std::vector<float> _weight(grid_size*grid_size*grid_size, 0.0f);
std::vector<int> _r(grid_size*grid_size*grid_size, 0);
std::vector<int> _g(grid_size*grid_size*grid_size, 0);
std::vector<int> _b(grid_size*grid_size*grid_size, 0);
std::vector<float> _weight_color(grid_size*grid_size*grid_size, 0.0f);
float weight_color_old = 0.0f;
float grid_old = 0.0f;
float weight_old = 0.0f;
Eigen::Matrix3d _R;
Eigen::Vector3d _t;
Eigen::VectorXd _Xi(6);


// initialize voxel grid -------------------------------------------------------
void getVoxelPointCloud()
{
  int center = grid_size/2;
  _voxel_mid_points.points.resize(grid_size*grid_size*grid_size);
  int counter = 0;
  // fill PointCloud
  for (int i = 0; i < grid_size; i++)
  {
    for (int j = 0; j < grid_size; j++)
    {
      for (int k = 0; k < grid_size; k++)
      {
        _voxel_mid_points.points[counter].x = (i - center) * voxel_length;
        _voxel_mid_points.points[counter].y = (j - center) * voxel_length;
        _voxel_mid_points.points[counter].z = (k - center) * voxel_length;
        counter++;
      }
    }
  }
}

// get cam info ---------------------------------------------------------------
void CamInfoCb( const sensor_msgs::CameraInfoConstPtr& info_msg)
{
  cam_model_.fromCameraInfo(info_msg);
  if (cam_model_.initialized()) {imageInfo->shutdown();}
}

// get absolute value from 3-1 vector -----------------------------------------
float getAbs(Eigen::Vector3d z)
{
  float abs = sqrt(pow(z(0),2) + pow(z(1),2) + pow(z(2),2));
  return abs;
}

// get Xi from R and t through logarithmic mapping ----------------------------
Eigen::VectorXd getXiFromRotationAndTranslation(Eigen::Matrix3d R, Eigen::Vector3d t)
{
  Eigen::VectorXd Xi(6);
  float theta = acos((_R.trace() - 1)/2);
  Eigen::Matrix3d w_bar;
  Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d A_inv;
  Eigen::Vector3d w;
  Eigen::Vector3d v;
  float w_abs;
  if (theta == 0)
  {
    Xi(0) = 0;
    Xi(1) = 0;
    Xi(2) = 0;
    Xi(3) = t(0);
    Xi(4) = t(1);
    Xi(5) = t(2);
  }
  else
  {
    w_bar = theta/(2 * sin(theta)) * (_R - _R.transpose());
    w(0) = w_bar(2,1);
    w(1) = w_bar(0,2);
    w(2) = w_bar(1,0); 
    w_abs = getAbs(w);
    Xi(0) = w(0);
    Xi(1) = w(1);
    Xi(2) = w(2);
    A_inv = I - (0.5 * w_bar) + ((2 * sin(w_abs) - (w_abs * (1 + cos(w_abs))))/(2 * (w_abs * w_abs) * sin(w_abs))) * (w_bar * w_bar);
    v = A_inv * t;
    Xi(3) = v(0);
    Xi(4) = v(1);
    Xi(5) = v(2);
  }
  return Xi;
}

// get R from Xi through exponential mapping ----------------------------------
Eigen::Matrix3d getRotationFromXi(Eigen::VectorXd Xi)
{
  Eigen::Matrix3d R;
  Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  Eigen::Vector3d w;
  w(0) = Xi(0);
  w(1) = Xi(1);
  w(2) = Xi(2);
  float theta = getAbs(w);
  if (theta == 0)
  {
    R = I;
  }
  else
  {
    Eigen::Matrix3d w_bar;
    w_bar << 0, -w(2), w(1),
             w(2), 0, -w(0),
             -w(1), w(0), 0;
    R = I +  (sin(theta)/theta) * w_bar + ((1 - cos(theta))/pow(theta,2)) * (w_bar * w_bar);
  }
  return R;
}

// get t from Xi through exponential mapping -----------------------------------
Eigen::Vector3d getTranslationFromXi(Eigen::VectorXd Xi)
{
  Eigen::Vector3d t;
  Eigen::Matrix3d A;
  Eigen::Vector3d w;
  Eigen::Vector3d v;
  Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  w(0) = Xi(0);
  w(1) = Xi(1);
  w(2) = Xi(2);
  v(0) = Xi(3);
  v(1) = Xi(4);
  v(2) = Xi(5);
  float theta = getAbs(w);
  if (theta == 0)
  {
    t(0) = Xi(3);
    t(1) = Xi(4);
    t(2) = Xi(5);
  }
  else
  {
    Eigen::Matrix3d w_bar;
    w_bar << 0, -w(2), w(1),
             w(2), 0, -w(0),
             -w(1), w(0), 0;
    A = I + ((1 - cos(theta))/(pow(theta,2))) * w_bar + ((theta - sin(theta))/(pow(theta,3))) * (w_bar * w_bar);
    t = A * v;
  }
  return t;
}

// SDF value read out --------------------------------------------------------
float getSDFValueOfXi(Eigen::VectorXd Xi, cv::Point3d p)
{
  Eigen::Vector3d t = getTranslationFromXi(Xi);
  Eigen::Matrix3d R = getRotationFromXi(Xi);
  Eigen::Vector3d point;
  point(0) = p.x;
  point(1) = p.y;
  point(2) = p.z;
  // cout << "local frame" << endl;
  // cout << point << endl;
  // cout << "local point" << endl;
  // cout << i << " " << j << " " << " " << k << endl;
  // cout << _grid[i * grid_size * grid_size + j * grid_size + k] << endl;
  point = (R * point + t);
  // cout << "world frame" << endl;
  // cout << point << endl;
  // cout << "Xi:" << endl;
  // cout << Xi << endl;
  // cout << "R and t" << endl;
  // cout << R << endl;
  // cout << t << endl;
  int x = (point(0)/voxel_length + grid_size/2) + 0.5;
  int y = (point(1)/voxel_length + grid_size/2) + 0.5;
  int z = (point(2)/voxel_length + grid_size/2) + 0.5;
  // cout << "world point" << endl;
  // cout << x << " " << y << " " << z << endl;
  // cout << _grid[x * grid_size * grid_size + y * grid_size + z] << endl;
  if (x > 255 || x < 0 || y > 255 ||  y < 0 || z > 255 || z < 0)
  {
    return 1000;
  }
  else
  {
    return _grid[x * grid_size * grid_size + y * grid_size + z];
  }
}

// function for partial numerical differentiation -----------------------------
Eigen::VectorXd getParDiff(Eigen::VectorXd Xi, cv::Point3d p, float h)
{
  Eigen::VectorXd Xi_pos(6);
  Eigen::VectorXd Xi_neg(6);
  Eigen::VectorXd diff_vector(6);
  float diff_pos;
  float diff;
  for (int x = 0; x < 6; x++)
  {
    Xi_pos = Xi;
    Xi_neg = Xi;
    Xi_pos(x) = Xi(x) + h;
    Xi_neg(x) = Xi(x) - h;
    diff_pos = getSDFValueOfXi(Xi_pos, p);
    diff = getSDFValueOfXi(Xi, p);
    if (diff_pos == 1000 || diff == 1000)
    {
      diff_vector << 0,0,0,0,0,0;
    }
    else
    {
      diff_vector(x) = (diff_pos - diff)/(h);
      if (diff_pos != 0.5 || diff != 0.5)
      {
      // cout << diff_pos << " " << diff << endl;
      }
      else {}
    }
  }
  return diff_vector;

}

// Transform voxel grid in camera coordinate system ---------------------------
void transformCloud(cv_bridge::CvImagePtr cv_ptr)
{
  if (pic == 0) { }
  else
  {
    cv::Mat &mat = cv_ptr->image;
    cv::Point2d pixel;
    cv::Point3d point;
    Eigen::VectorXd Xi(6);
    Xi = getXiFromRotationAndTranslation(_R, _t);
    Eigen::MatrixXd A(6, 6);
    Eigen::MatrixXd A_temp(6, 6);
    Eigen::VectorXd b(6);
    float h = 0.02;
    Eigen::VectorXd diff_vector(6);
    for (int iteration = 0; iteration < 7; iteration++)
    {
      cout << "iteration:" << iteration << endl;
      A << 0,0,0,0,0,0,
           0,0,0,0,0,0,
           0,0,0,0,0,0,
           0,0,0,0,0,0,
           0,0,0,0,0,0,
           0,0,0,0,0,0;
      b << 0,0,0,0,0,0;
      for (int i = 0; i < 480; i++)
      {
        for (int j = 0; j < 640; j++)
        {
          pixel.x = i;
          pixel.y = j;
          point = cam_model_.projectPixelTo3dRay(pixel);
          point.z = point.z * mat.at<float>(i, j);
          if(isnan(point.z)) { }
          else
          {
            diff_vector = getParDiff(Xi, point, h);
            A_temp = diff_vector * diff_vector.transpose();
            if (A_temp.determinant() == 0) { }
            else
            {
            A = A + A_temp;
            b = b + getSDFValueOfXi(Xi, point) * diff_vector;
            }
          }
        }
      }
      Xi = Xi - A.inverse() * b;
    }
  }
  std::string target_frame = "/openni_depth_optical_frame";
  _voxel_mid_points.header.frame_id = "/world";
   printf("wait for tf frame\n");
  _listener->waitForTransform( target_frame, _voxel_mid_points.header.frame_id.c_str(), acquisition_time, ros::Duration(2.0));
  try
  {
  _listener->TransformListener::transformPointCloud (target_frame, _voxel_mid_points, _v_mid_points_tf);
  }
  catch (tf::TransformException ex)
  {
    ROS_ERROR("%s",ex.what());
  } 
}

// Function for trilinear interpolation
void interpolate( geometry_msgs::Point point, std_msgs::ColorRGBA &color)
{
  float x = point.x;
  float y = point.y;
  float z = point.z;
  if (x < 0 || y < 0 || z < 0 || x > grid_size - 1 || y > grid_size - 1 || z > grid_size - 1)
  {
  color.r = 0;
  color.g = 0;
  color.b = 0;
  }
  else
  {
  // get edge points
  int x_0 = x;
  int x_1 = x + 1;
  int y_0 = y;
  int y_1 = y + 1;
  int z_0 = z;
  int z_1 = z + 1;
  
  int r1 = _r[x_0 * grid_size * grid_size + y_0 * grid_size + z_0];
  int r2 = _r[x_1 * grid_size * grid_size + y_0 * grid_size + z_0];
  int r3 = _r[x_0 * grid_size * grid_size + y_1 * grid_size + z_0];
  int r4 = _r[x_1 * grid_size * grid_size + y_1 * grid_size + z_0];
  int r5 = _r[x_0 * grid_size * grid_size + y_0 * grid_size + z_1];
  int r6 = _r[x_1 * grid_size * grid_size + y_0 * grid_size + z_1];
  int r7 = _r[x_0 * grid_size * grid_size + y_1 * grid_size + z_1];
  int r8 = _r[x_1 * grid_size * grid_size + y_1 * grid_size + z_1];

  int g1 = _g[x_0 * grid_size * grid_size + y_0 * grid_size + z_0];
  int g2 = _g[x_1 * grid_size * grid_size + y_0 * grid_size + z_0];
  int g3 = _g[x_0 * grid_size * grid_size + y_1 * grid_size + z_0];
  int g4 = _g[x_1 * grid_size * grid_size + y_1 * grid_size + z_0];
  int g5 = _g[x_0 * grid_size * grid_size + y_0 * grid_size + z_1];
  int g6 = _g[x_1 * grid_size * grid_size + y_0 * grid_size + z_1];
  int g7 = _g[x_0 * grid_size * grid_size + y_1 * grid_size + z_1];
  int g8 = _g[x_1 * grid_size * grid_size + y_1 * grid_size + z_1];

  int b1 = _b[x_0 * grid_size * grid_size + y_0 * grid_size + z_0];
  int b2 = _b[x_1 * grid_size * grid_size + y_0 * grid_size + z_0];
  int b3 = _b[x_0 * grid_size * grid_size + y_1 * grid_size + z_0];
  int b4 = _b[x_1 * grid_size * grid_size + y_1 * grid_size + z_0];
  int b5 = _b[x_0 * grid_size * grid_size + y_0 * grid_size + z_1];
  int b6 = _b[x_1 * grid_size * grid_size + y_0 * grid_size + z_1];
  int b7 = _b[x_0 * grid_size * grid_size + y_1 * grid_size + z_1];
  int b8 = _b[x_1 * grid_size * grid_size + y_1 * grid_size + z_1];
  

  if (r1 || r2 || r3 || r4 || r5 || r6 || r7 || r8 || g1 || g2 || g3 || g4 || g5 || g6 || g7 || g8 || b1 || b2 || b3 || b4 || b5 || b6 || b7 || b8 == 0)
  {
  cout << r1 << "  " << r2 << "  " << r3 << "  " << r4 << "  " << r5 << "  " << r6 << "  " << r7 << "  " << r8 << endl;
  }
  else 
  {
  float x_d = (x - x_0)/(x_1 - x_0);
  float y_d = (y - y_0)/(y_1 - y_0);
  float z_d = (z - z_0)/(z_1 - z_0);
  std_msgs::ColorRGBA c00;
  c00.r = r1 * (1 - x_d) + r2 * x_d;
  c00.g = _g[x_0 * grid_size * grid_size + y_0 * grid_size + z_0] * (1 - x_d) + _g[x_1 * grid_size * grid_size + y_0 * grid_size + z_0] * x_d;
  c00.b = _b[x_0 * grid_size * grid_size + y_0 * grid_size + z_0] * (1 - x_d) + _b[x_1 * grid_size * grid_size + y_0 * grid_size + z_0] * x_d;
  std_msgs::ColorRGBA c10;
  c10.r = _r[x_0 * grid_size * grid_size + y_1 * grid_size + z_0] * (1 - x_d) + _r[x_1 * grid_size * grid_size + y_1 * grid_size + z_0] * x_d;
  c10.g = _g[x_0 * grid_size * grid_size + y_1 * grid_size + z_0] * (1 - x_d) + _g[x_1 * grid_size * grid_size + y_1 * grid_size + z_0] * x_d;
  c10.b = _b[x_0 * grid_size * grid_size + y_1 * grid_size + z_0] * (1 - x_d) + _b[x_1 * grid_size * grid_size + y_1 * grid_size + z_0] * x_d;
  std_msgs::ColorRGBA c01;
  c01.r = _r[x_0 * grid_size * grid_size + y_0 * grid_size + z_1] * (1 - x_d) + _r[x_1 * grid_size * grid_size + y_0 * grid_size + z_1] * x_d;
  c01.g = _g[x_0 * grid_size * grid_size + y_0 * grid_size + z_1] * (1 - x_d) + _g[x_1 * grid_size * grid_size + y_0 * grid_size + z_1] * x_d;
  c01.b = _b[x_0 * grid_size * grid_size + y_0 * grid_size + z_1] * (1 - x_d) + _b[x_1 * grid_size * grid_size + y_0 * grid_size + z_1] * x_d;
  std_msgs::ColorRGBA c11;
  c11.r = _r[x_0 * grid_size * grid_size + y_1 * grid_size + z_1] * (1 - x_d) + _r[x_1 * grid_size * grid_size + y_1 * grid_size + z_1] * x_d;
  c11.g = _g[x_0 * grid_size * grid_size + y_1 * grid_size + z_1] * (1 - x_d) + _g[x_1 * grid_size * grid_size + y_1 * grid_size + z_1] * x_d;
  c11.b = _b[x_0 * grid_size * grid_size + y_1 * grid_size + z_1] * (1 - x_d) + _b[x_1 * grid_size * grid_size + y_1 * grid_size + z_1] * x_d;
  std_msgs::ColorRGBA c0;
  c0.r = c00.r * (1 - y_d) + c10.r * y_d;
  c0.g = c00.g * (1 - y_d) + c10.g * y_d;
  c0.b = c00.b * (1 - y_d) + c10.b * y_d;
  std_msgs::ColorRGBA c1;
  c1.r = c01.r * (1 - y_d) + c11.r * y_d;
  c1.g = c01.g * (1 - y_d) + c11.g * y_d;
  c1.b = c01.b * (1 - y_d) + c11.b * y_d;
  color.r = c0.r * (1 - z_d) + c1.r * z_d;
  color.g = c0.g * (1 - z_d) + c1.g * z_d;
  color.b = c0.b * (1 - z_d) + c1.b * z_d;
  }
  }
}

// Image callback function --------------------------------------------------
void imageCb(const sensor_msgs::ImageConstPtr& image, const sensor_msgs::ImageConstPtr& image_color)
{
  cout << "Editing picture " << pic << endl;
  if ( !cam_model_.initialized())
  {
    return;
  }
  else
  {
  }
  cv_bridge::CvImagePtr cv_ptr;
  cv_bridge::CvImagePtr cv_ptr_c;
  cv_ptr = cv_bridge::toCvCopy(image, enc::TYPE_32FC1);
  cv_ptr_c = cv_bridge::toCvCopy(image_color, enc::BGR8);
  if (pic == 0)
  {
     _v_mid_points_tf.header.stamp = image->header.stamp;
  }
  else { }
  _voxel_mid_points.header.stamp = image->header.stamp;
  acquisition_time = image->header.stamp;
  transformCloud(cv_ptr);
  cv::Mat &mat = cv_ptr->image;
  cv::Mat &mat_color = cv_ptr_c->image;
  int width = mat.cols;
  int height = mat.rows;
  int i_pixel;
  int j_pixel;
  float i_d;
  float d;
  float z;
  float r;
  float g;
  float b;
  float _r_old;
  float _g_old;
  float _b_old;
  cv::Vec3b color;

  printf("generate SDF...");
  for (int x = 0; x < grid_size*grid_size*grid_size; x++)
  {
    // ------------------------------------------
    // Cut out voxels which are behind the camera
    // ------------------------------------------
    if (_v_mid_points_tf.points[x].z < 0) 
    {
    }
    else
    {
      cv::Point3d pt_cv(_v_mid_points_tf.points[x].y, _v_mid_points_tf.points[x].x , _v_mid_points_tf.points[x].z);
      cv::Point2d uv;
      uv = cam_model_.project3dToPixel(pt_cv);
      if ( uv.x > 480 || uv.y > 640 || uv.x < 0 || uv.y < 0) 
      {
      }
      else
      {
        i_pixel = uv.x + 0.5;
        j_pixel = uv.y + 0.5;
        color = mat_color.at<cv::Vec3b>(i_pixel, j_pixel);
        // --------------------------------------------------------------------------------------
        // set projective point-to-point distance "d" as difference of the depth of the voxel "z"
        // and and the observed path "i_d" of pixel "i" and "j"
        // --------------------------------------------------------------------------------------
        i_d = mat.at<float>(i_pixel, j_pixel);
        z = _v_mid_points_tf.points[x].z;
        d = - (z - i_d);
        //save old weight and grid-values for weighting
        weight_old = _weight[x];
        grid_old = _grid[x];
        weight_color_old = _weight_color[x];
        _r_old = _r[x];
        _g_old = _g[x];
        _b_old = _b[x];
        // --------------------------------------------------
        // get color of voxels which are close to the surface
        // --------------------------------------------------
        if (d >= - 1.5 * epsilon && d <= delta)
        {
          _r[x] = 255 - color[2];
          _g[x] = 255 - color[1];
          _b[x] = 255 - color[0];
        }
        // ----------------------------------------------------
        // Store point-to-point distance "d" in the voxel grid
        // if distance is between -delta & delta, else truncate 
        // values to -delta and deltas
        // ----------------------------------------------------
        if (d <= delta && d >= - 0.1 )
        {
          _grid[x] = d;
        }
        else if ( d > delta)
        {
          _grid[x] = delta;
        }
        else 
        {
        }
        // ---------------------------------------------------------
        // Apply a linear weighting for each voxel in the grid
        // ---------------------------------------------------------
        if (d > - epsilon)
        {
          _weight[x] = 1;
        }
        else if(d <= - epsilon && d >= -0.1)
        {
          _weight[x] = ( 0.1 - d)/( 0.1 - epsilon);
        }
        else
        {
          _weight[x] = 0;
        }
        // ------------------------------------------------------------
        // Weight algorithms are:
        //
        // D_i(x) = (W_i-1(x)D_i-1(x) + W_i(x)D_i(x))/(W_i(x)+W_i-1(x))
        //
        // W_i(x) = W_i + W_i-1(x)
        //
        // Make also sure you don't divide through zero
        // ------------------------------------------------------------
        if (_weight[x] + weight_old == 0)
        {
        }
        else
        { 
          _grid[x] = (((weight_old * grid_old) + (_weight[x] * _grid[x])) / ( _weight[x] + weight_old));
        }
        _weight[x] = _weight[x] + weight_old;
        // ----------------
        // Color weighting
        // ----------------
        _r[x] = (weight_old * _r_old + _weight[x] * _r[x]) / (_weight[x] + weight_old);
        _g[x] = (weight_old * _g_old + _weight[x] * _g[x]) / (_weight[x] + weight_old);
        _b[x] = (weight_old * _b_old + _weight[x] * _b[x]) / (_weight[x] + weight_old);
      }
    }
  }
  cv::Mat &mat2 = cv_ptr->image;
  cv::Point2d pixel;
  cv::Point3d point;
  Eigen::Vector3d p;
  float norm;
  int wrongs = 0;
  int rights = 0;
  for (int i = 0; i < 480; i++)
  {
    for (int j = 0; j < 640; j++)
    {
      pixel.x = i;
      pixel.y = j;
      point = cam_model_.projectPixelTo3dRay(pixel);
      p(0) = point.x;
      p(1) = point.y;
      p(2) = point.z;
      norm = p.norm();
      i_d = mat2.at<float>(i, j);
      if (isnan(i_d)) { }
      else
      {
        p = p * mat2.at<float>(i, j);
        if (_grid[p(0) * grid_size * grid_size + p(1) * grid_size + p(2)] == 0)
        {
          rights++;
        }
        else {wrongs++;}
        // cout << _grid[p(0) * grid_size * grid_size + p(1) * grid_size + p(2)] << endl;
      }
    }
  }
  cout << "rights: " << rights << endl;
  cout << "wrongs: " << wrongs << endl;
  printf ("finished\n");
  if (pic >= 3)
  {
    ros::shutdown();
    pic++;
  }
  else 
  {
    pic++;
  }
  // ----------------------------------------------------
  // create a mesh with marching cubes every 10th picture
  // ----------------------------------------------------
  if (pic%10 == 0)
  {
    cout << "begin marching cubes reconstruction" << endl;
    mc.reconstruct (_grid, *triangles);
    cout << triangles->polygons.size() << " triangles created" << endl;
    // -----------------------------------------
    // use a publisher to visualize mesh to rviz
    // -----------------------------------------
    pcl::PointCloud<pcl::PointXYZ> cloud2;
    pcl::fromPCLPointCloud2(triangles->cloud, cloud2);
    visualization_msgs::Marker marker;
    marker.header.frame_id = "/world";
    marker.header.stamp = ros::Time::now();
    marker.ns = "Triangle_List";
    marker.id = 1;
    marker.type = visualization_msgs::Marker::TRIANGLE_LIST;
    marker.action = visualization_msgs::Marker::ADD;
    marker.scale.x = 1.0;
    marker.scale.y = 1.0;
    marker.scale.z = 1.0;
    marker.color.g = 1.0;
    marker.color.a = 1.0;

    int x = 0;
    int y = 1;
    int z = 2;
    for (int k = 0; k < cloud2.size(); k++)
    {
       geometry_msgs::Point p;
       p.x = cloud2[k].x;
       p.y = cloud2[k].y;
       p.z = cloud2[k].z;

       std_msgs::ColorRGBA c;
       // TODO: make interpolate function work
       // interpolate(p, c);
       int x = cloud2[k].x + 0.5;
       int y = cloud2[k].y + 0.5;
       int z = cloud2[k].z + 0.5;
       c.b = _b[x * grid_size * grid_size + y * grid_size + z];
       c.g = _g[x * grid_size * grid_size + y * grid_size + z];
       c.r = _r[x * grid_size * grid_size + y * grid_size + z];
       c.a = 1;
       p.x = cloud2[k].x * 0.02 -2.55;
       p.y = cloud2[k].y * 0.02 -2.55;
       p.z = cloud2[k].z * 0.02 -2.55;

       marker.points.push_back(p);
       marker.colors.push_back(c);
    }
  pub->publish (marker);
  }
  else
  {
  }
}


int main(int argc, char** argv)
{
  printf("initialize\n");
  _R << 1, 0, 0,
        0, 1, 0,
        0, 0, 1;
  _t << 0, 0, 0;
  ros::init(argc, argv, "image_listener");
  _listener = new tf::TransformListener;
  getVoxelPointCloud();
  ros::NodeHandle nh;
  ros::NodeHandle n;
  ros::NodeHandle node;
  // ---------------------------------------
  // set marching cubes bounding conditions
  // ---------------------------------------
  mc.setGridResolution(grid_size, grid_size, grid_size);
  mc.setIsoLevel(0.1);
  mc.setBoundingBox(grid_size, voxel_length);
  // -----------------------------
  // use subscriber to get camInfo
  // -----------------------------
  imageInfo = new ros::Subscriber;
  *imageInfo = n.subscribe("/camera/depth/camera_info", 1, CamInfoCb);
  image_transport::ImageTransport it(node);
  // --------------------------------------------------------
  // use two synced subscribers to get depth and color images
  // --------------------------------------------------------
  message_filters::Subscriber<sensor_msgs::Image> image_sub(node, "/camera/depth/image", 5);
  message_filters::Subscriber<sensor_msgs::Image> image_sub2(node, "/camera/rgb/image_color", 5);
  typedef sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
  Synchronizer<MySyncPolicy> sync(MySyncPolicy(1), image_sub, image_sub2);
  sync.registerCallback(boost::bind(&imageCb, _1, _2));
  // ---------------------------------
  // use publisher to get mesh to rviz
  // ---------------------------------
  pub = new ros::Publisher;
  *pub = n.advertise<visualization_msgs::Marker>("marker", 1);

  ros::spin();

  // save file.

  // io::savePolygonFilePLY ("test.ply", *triangles);

  return 0;
}
