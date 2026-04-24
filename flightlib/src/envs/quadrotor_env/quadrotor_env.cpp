#include "flightlib/envs/quadrotor_env/quadrotor_env.hpp"

namespace flightlib {

QuadrotorEnv::QuadrotorEnv()
  : QuadrotorEnv(getenv("FLIGHTMARE_PATH") +
                 std::string("/flightlib/configs/quadrotor_env.yaml")) {}

QuadrotorEnv::QuadrotorEnv(const std::string &cfg_path)
  : EnvBase(),
    rot_scale_(1.0),
    rot_mult_(0.1),
    goal_state_((Vector<quadenv::kNObs>() << 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0)
                  .finished()) {
  // load configuration file
  YAML::Node cfg_ = YAML::LoadFile(cfg_path);

  quadrotor_ptr_ = std::make_shared<Quadrotor>();
  // update dynamics
  QuadrotorDynamics dynamics;
  dynamics.updateParams(cfg_);
  quadrotor_ptr_->updateDynamics(dynamics);

  // define a bounding box
  world_box_ << -100, 100, -100, 100, 0, 100;
  if (!quadrotor_ptr_->setWorldBox(world_box_)) {
    logger_.error("cannot set wolrd box");
  };

  // define input and output dimension for the environment
  obs_dim_ = quadenv::kNObs;
  act_dim_ = quadenv::kNAct;

  Scalar mass = quadrotor_ptr_->getMass();
  act_mean_ = Vector<quadenv::kNAct>::Ones() * (-mass * Gz) / 4;
  act_std_ = Vector<quadenv::kNAct>::Ones() * (-mass * 2 * Gz) / 4;

  // load parameters
  loadParam(cfg_);
}

QuadrotorEnv::~QuadrotorEnv() {}

void QuadrotorEnv::addRGBCamera() {
  // setup cam
  rgb_camera_ = std::make_shared<RGBCamera>();
  Vector<3> B_r_BC(0.0, 0.0, 0.3);
  // Matrix<3, 3> R_BC = Quaternion(1.0, 0.0, 0.0, 0.0).toRotationMatrix();
  // 45 degree upwards tilt to the camera 
  Matrix<3, 3> R_BC = Quaternion(cos(M_PI/8), sin(M_PI/8), 0.0, 0.0).toRotationMatrix();
  rgb_camera_->setFOV(90);
  rgb_camera_->setWidth(320);
  rgb_camera_->setHeight(320);
  rgb_camera_->setRelPose(B_r_BC, R_BC);
  quadrotor_ptr_->addRGBCamera(rgb_camera_);
}

void QuadrotorEnv::modifyResetPositions(Ref<Vector<>> pos) {
  x_dist = std::uniform_real_distribution<Scalar>(pos[0], pos[1]);
  y_dist = std::uniform_real_distribution<Scalar>(pos[2], pos[3]);
  z_dist = std::uniform_real_distribution<Scalar>(pos[4], pos[5]);
}

bool QuadrotorEnv::reset(Ref<Vector<>> obs, const bool random) {
  quad_state_.setZero();
  quad_act_.setZero();

  if (random) {
    // randomly reset the quadrotor state
    quad_state_.x(QS::POSX) = x_dist(random_gen_); //uniform_dist_(random_gen_);
    quad_state_.x(QS::POSY) = y_dist(random_gen_); //uniform_dist_(random_gen_);
    quad_state_.x(QS::POSZ) = z_dist(random_gen_); //uniform_dist_(random_gen_) + 5;
    if (quad_state_.x(QS::POSZ) < -0.0)
      quad_state_.x(QS::POSZ) = -quad_state_.x(QS::POSZ);
    // reset linear velocity
    quad_state_.x(QS::VELX) = uniform_dist_(random_gen_);
    quad_state_.x(QS::VELY) = uniform_dist_(random_gen_);
    quad_state_.x(QS::VELZ) = uniform_dist_(random_gen_);
    // reset orientation
    // (w, x, y, z)
    // how far is the quadcopter tilted (1.0~=0, 0.0~=180)
    quad_state_.x(QS::ATTW) = rot_scale_;

    // x, y, z vector components of quaternion for rot matrix
    quad_state_.x(QS::ATTX) = uniform_dist_(random_gen_) * rot_mult_;
    quad_state_.x(QS::ATTY) = uniform_dist_(random_gen_) * rot_mult_;
    quad_state_.x(QS::ATTZ) = uniform_dist_(random_gen_) * rot_mult_;
    quad_state_.qx /= quad_state_.qx.norm();
  }
  // reset quadrotor with random states
  quadrotor_ptr_->reset(quad_state_);

  // reset control command
  cmd_.t = 0.0;
  cmd_.thrusts.setZero();

  // obtain observations
  getObs(obs);
  return true;
}

bool QuadrotorEnv::getObs(Ref<Vector<>> obs) {
  quadrotor_ptr_->getState(&quad_state_);

  // convert quaternion to euler angle
  Vector<3> euler_zyx = quad_state_.q().toRotationMatrix().eulerAngles(2, 1, 0);
  // quaternionToEuler(quad_state_.q(), euler);
  quad_obs_ << quad_state_.p, euler_zyx, quad_state_.v, quad_state_.w;

  obs.segment<quadenv::kNObs>(quadenv::kObs) = quad_obs_;
  return true;
}

bool QuadrotorEnv::getRGBImage(cv::Mat& img) {
  return rgb_camera_->getRGBImage(img);
}

Scalar QuadrotorEnv::step(const Ref<Vector<>> act, Ref<Vector<>> obs) {
  // ---------- gets collective thrust and roll, pitch, yaw rates
  cmd_.t += sim_dt_;
  cmd_.thrusts.fill(NAN);
  cmd_.collective_thrust = 9.81 * (1.0 + act[0]);
  cmd_.omega = act.segment<3>(1) * 6.0;

  // simulate quadrotor
  quadrotor_ptr_->run(cmd_, sim_dt_);

  // update observations
  getObs(obs);

  return 0.0;
}

bool QuadrotorEnv::isTerminalState(Scalar &reward) {
  if (quad_state_.x(QS::POSZ) <= 0.02) {
    reward = -10.0;
    return true;
  }
  if (cmd_.t >= max_t_) {
    reward = 0.0;
    return true;
  }
  reward = 0.0;
  return false;
}

bool QuadrotorEnv::loadParam(const YAML::Node &cfg) {
  if (cfg["quadrotor_env"]) {
    sim_dt_ = cfg["quadrotor_env"]["sim_dt"].as<Scalar>();
    max_t_ = cfg["quadrotor_env"]["max_t"].as<Scalar>();
    rot_mult_ = cfg["quadrotor_env"]["rotation_mult"].as<Scalar>();
    rot_scale_ = cfg["quadrotor_env"]["rotation_scale"].as<Scalar>();
  } else {
    return false;
  }

  return true;
}

void QuadrotorEnv::increaseRotMult(const Scalar num) {
  rot_mult_ = std::min(rot_mult_ + num, (float)1.0);
}

void QuadrotorEnv::decreaseRotMult(const Scalar num) {
  rot_mult_ = std::max(rot_mult_ - num, (float)0.0);
}

bool QuadrotorEnv::getAct(Ref<Vector<>> act) const {
  if (cmd_.t >= 0.0 && quad_act_.allFinite()) {
    act = quad_act_;
    return true;
  }
  return false;
}

bool QuadrotorEnv::getAct(Command *const cmd) const {
  if (!cmd_.valid()) return false;
  *cmd = cmd_;
  return true;
}

void QuadrotorEnv::addObjectsToUnity(std::shared_ptr<UnityBridge> bridge) {
  bridge->addQuadrotor(quadrotor_ptr_);
}

std::ostream &operator<<(std::ostream &os, const QuadrotorEnv &quad_env) {
  os.precision(3);
  os << "Quadrotor Environment:\n"
     << "obs dim =            [" << quad_env.obs_dim_ << "]\n"
     << "act dim =            [" << quad_env.act_dim_ << "]\n"
     << "sim dt =             [" << quad_env.sim_dt_ << "]\n"
     << "max_t =              [" << quad_env.max_t_ << "]\n"
     << "act_mean =           [" << quad_env.act_mean_.transpose() << "]\n"
     << "act_std =            [" << quad_env.act_std_.transpose() << "]\n"
     << "obs_mean =           [" << quad_env.obs_mean_.transpose() << "]\n"
     << "obs_std =            [" << quad_env.obs_std_.transpose() << std::endl;
  os.precision();
  return os;
}

}  // namespace flightlib