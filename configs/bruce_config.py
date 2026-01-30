from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class BruceRoughCfg(LeggedRobotCfg):
    class init_state( LeggedRobotCfg.init_state):
        


        pos = [0.0, 0.0, 0.48]      # x,y,z [m]
        randomize = False

        #rot = [0.0, 0.0, 0.0, 1.0]
        #ang_vel = [0.0, 0.0, 0.0]
#         right foot initial state: [-0.00824264  0.46926757  0.01822493 -0.94714788  0.47781332]
# left foot initial state: [ 0.00824264  0.46926757 -0.01822493 -0.94714788  0.47781332]

        default_joint_angles = { # = target angles [rad] when action = 0.0
           'hip_yaw_r' : -0.00824264,   
           'hip_roll_r' : 0.01822493,    

           'hip_pitch_r' :  0.26926757,   #'hip_pitch_r' :  0.46926757,         
           'knee_pitch_r' : -0.94714788,       
           'ankle_pitch_r' : 0.180,#'ankle_pitch_r' : 0.47781332,     
        
           'hip_yaw_l' : 0.00824264, 
           'hip_roll_l' : -0.01822493, 

           'hip_pitch_l' : 0.26926757,#'hip_pitch_l' : 0.46926757,                                       
           'knee_pitch_l' : -0.94714788,                                             
           'ankle_pitch_l' : 0.180,#'ankle_pitch_l' : 0.47781332,                                     
        
           # Arm default positions: Removed for now due to tensor mismatch
           # Tensor mismatch occurs because simulation removes arms, this
           # issue must be further investigates if the arms are to be used
            # 'shoulder_pitch_l': 0.0,
            # 'shoulder_roll_l': 0.0,
            # 'elbow_pitch_l': 0.0,
            # 'shoulder_pitch_r': 0.0,
            # 'shoulder_roll_r': 0.0,
            # 'elbow_pitch_r': 0.0,

           }
    
    class env(LeggedRobotCfg.env):
        num_observations = 41
        num_privileged_obs = 44
        num_actions = 10

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        add_kf_noise = False
        add_noise = False
        randomize_motor_strength = False
        randomize_com = False

        imperfect_imu = False
        randomize_com = False


        friction_range = [0.4, 1.20]
        randomize_base_mass = False
        added_mass_range = [-0.2, 0.2]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 0.2

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip_yaw': 265.0, 'hip_roll': 80.0, 'hip_pitch': 150.0,
                     'knee': 80.0,
                     'ankle': 30.0,
                     
                     # 'shoulder': 10.0, 'elbow': 10.0
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 1.0,
                     'hip_roll': 0.8,
                     'hip_pitch': 2.3,
                     'knee': 0.8,
                     'ankle': 0.003,
                     # 'shoulder': 0.5, 'elbow':0.5,
                     }

        # PD gains from BRUCE_macros.py (manufacturer values)
        # pos_kp = 60.0, pos_kd = 1.0 for all joints
        # stiffness = {'hip_yaw': 70.0, 'hip_roll': 70.0, 'hip_pitch': 85.0,
        #              'knee': 80.0,
        #              'ankle': 95.0,
                     
        #              # 'shoulder': 60.0, 'elbow': 60.0
        #              }  # [N*m/rad]
        # damping = {  'hip_yaw': 2.0,
        #              'hip_roll': 2.5,
        #              'hip_pitch': 2.0,
        #              'knee': 3.0,
        #              'ankle': 2.0,
        #              # 'shoulder': 1.0, 'elbow': 1.0,
        #              }

        # action scale: target angle = actionScale * action + defaultAngle
        # increased from 0.25 to 3.25 as BRUCE is much higher frequency than g1
        action_scale = 0.50
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 6

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/bruce/urdf/bruce2.urdf'
        name = "bruce"
        foot_name = "ankle_pitch"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["base_link", "knee"]
        disable_gravity = False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False


        # fix_base_link = True 
        # override_dof_limits = True


    class noise( LeggedRobotCfg.noise ):
        # Domain randomisation (not policy randomisation)
        add_noise = False
        noise_level = 0.0

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9

        ### CHANGED FROM 0.5 to 0.44 as per BRUCE specs, 0.445 is base height wil legs fully extended ### 
        base_height_target = 0.4499020000000001
        ### END OF CHANGE ###

        class scales( LeggedRobotCfg.rewards.scales ):
            # tracking_lin_vel = 2.8
            # tracking_ang_vel = 2.25
            # lin_vel_z = -0.5
            # ang_vel_xy = -0.05

            # # orientation reduced from -10 to -3.0 as 
            # #init orientation might not be optimal orientation
            # orientation = -3.0
            # base_height = -1.0
            
            # #dof_acc = -2.5e-7
            # dof_acc = 0
            
            # feet_air_time = 0.0
            # collision = -1.3
            # action_rate = -0.0002
            # torques = 0.0
            # dof_pos_limits = -0.001
            # alive = 17.85
            # hip_pos = -0.5
            # contact_no_vel = -0.2
            # feet_swing_height = -0.05
            # contact = 0.18
            # termination = -200

            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 0.0
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.18
            termination = -10

class BruceRoughCfgPPO( LeggedRobotCfgPPO ):

    class policy( LeggedRobotCfgPPO.policy ):
        init_noise_std = 0.8
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1


    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.001
        # learning_rate = 3e-4
        # desired_kl = 0.02

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
        run_name = ''
        experiment_name = 'bruce'        

