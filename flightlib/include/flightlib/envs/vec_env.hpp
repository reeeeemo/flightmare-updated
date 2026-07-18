//
// This is inspired by RaiGym, thanks.
//
#pragma once

// std
#include <memory>

// openmp
#include <omp.h>

#include <opencv2/core/core.hpp>

// flightlib
#include "flightlib/bridges/unity_bridge.hpp"
#include "flightlib/common/logger.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/envs/env_base.hpp"
#include "flightlib/envs/quadrotor_env/quadrotor_env.hpp"
#include "flightlib/sensors/rgb_camera.hpp"

namespace flightlib {

template<typename EnvBase>
class VecEnv {
 public:
  VecEnv();
  VecEnv(const std::string& cfgs, const bool from_file = true);
  VecEnv(const YAML::Node& cfgs_node);
  ~VecEnv();

  // - public OpenAI-gym style functions for vectorized environment
  bool reset(Ref<MatrixRowMajor<>> obs, const bool domrand = false);
  bool step(Ref<MatrixRowMajor<>> act, Ref<MatrixRowMajor<>> obs,
            Ref<Vector<>> reward, Ref<BoolVector<>> done,
            Ref<MatrixRowMajor<>> extra_info);
  bool stepUnity(Ref<MatrixRowMajor<>> act, Ref<MatrixRowMajor<>> obs,
                 Ref<Vector<>> reward, Ref<BoolVector<>> done,
                 Ref<MatrixRowMajor<>> extra_info, uint64_t send_id);
  void close();

  // public set functions
  void setSeed(const int seed);
  void increaseRotMult(const Scalar num);
  void decreaseRotMult(const Scalar num);
  void addRGBCamera();
  void modifyResetPositions(Ref<Vector<>> pos);
  void setLowestZ(const Scalar num);

  // public get functions
  void getObs(Ref<MatrixRowMajor<>> obs);
  size_t getEpisodeLength(void);
  void getRGBImages(std::vector<cv::Mat>& imgs);

  // - auxiliary functions
  void isTerminalState(Ref<BoolVector<>> terminal_state);
  void testStep(Ref<MatrixRowMajor<>> act, Ref<MatrixRowMajor<>> obs,
                Ref<Vector<>> reward, Ref<BoolVector<>> done,
                Ref<MatrixRowMajor<>> extra_info);
  void curriculumUpdate();

  // flightmare (visualization)
  bool setUnity(bool render);
  bool connectUnity();
  void disconnectUnity();
  void addGate(Ref<MatrixRowMajor<>> positions, Ref<MatrixRowMajor<>> rotations);

  // public functions
  inline int getSeed(void) { return seed_; };
  inline SceneID getSceneID(void) { return scene_id_; };
  inline bool getUnityRender(void) { return unity_render_; };
  inline int getObsDim(void) { return obs_dim_; };
  inline int getActDim(void) { return act_dim_; };
  inline int getExtraInfoDim(void) { return extra_info_names_.size(); };
  inline int getNumOfEnvs(void) { return envs_.size(); };
  inline std::vector<std::string>& getExtraInfoNames() {
    return extra_info_names_;
  };

  // friend std::ostream& operator<<(std::ostream& os,
  //                                 const VecEnv<EnvBase>& vec_env);

 private:
  // initialization
  void init(void);
  // step every environment
  void perAgentStep(int agent_id, Ref<MatrixRowMajor<>> act,
                    Ref<MatrixRowMajor<>> obs, Ref<Vector<>> reward,
                    Ref<BoolVector<>> done, Ref<MatrixRowMajor<>> extra_info);
  // create objects
  Logger logger_{"VecEnv"};
  std::vector<std::unique_ptr<EnvBase>> envs_;
  std::vector<std::string> extra_info_names_;

  // Flightmare(Unity3D)
  std::shared_ptr<UnityBridge> unity_bridge_ptr_;
  SceneID scene_id_{UnityScene::WAREHOUSE};
  bool unity_ready_{false};
  bool unity_render_{false};
  bool unity_camera_{false};
  RenderMessage_t unity_output_;
  uint16_t receive_id_{0};
  int gateCounter;

  // rl objects
  bool domrand_{false};

  // auxiliar variables
  int seed_, num_envs_, obs_dim_, act_dim_;
  Matrix<> obs_dummy_;

  // yaml configurations
  YAML::Node cfg_;
};

}  // namespace flightlib
