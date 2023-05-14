#include "softgrasp_ros/visualization_utils.h"

namespace o3dv = open3d::visualization;
namespace o3dg = open3d::geometry;
using namespace std;

void show_with_axes(vector<std::shared_ptr<const o3dg::Geometry>> geoms,
                    const string &window_name,
                    const vector<Eigen::Affine3d> &axes_poses) {
  for (const auto &axes_pose : axes_poses) {
    auto axes = o3dg::TriangleMesh::CreateCoordinateFrame(0.2);
    axes->Transform(axes_pose.matrix());
    geoms.push_back(axes);
  }
  o3dv::DrawGeometries(geoms, window_name);
}

void show_seg_with_axes(o3dg::PointCloud pc, const vector<size_t> idxs,
                        const string &window_name,
                        const vector<Eigen::Affine3d> &axes_poses) {
  pc.PaintUniformColor(Eigen::Vector3d(0, 0.651, 0.929));
  for (size_t idx: idxs) pc.colors_[idx] = Eigen::Vector3d(1, 0.706, 0);
  if (!pc.HasNormals()) pc.EstimateNormals();
  show_with_axes({std::make_shared<o3dg::PointCloud>(pc)}, window_name,
                 axes_poses);
}